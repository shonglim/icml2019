import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class KernelRL {
	
	//neighbor/weight(un-normalized) lookup only
	static Pair<int[],double[]> rbf_kernel_cached(int rbf_K,double rbf_sigma,KdTree2.SqrEuclid<Integer> rbf_tree,
			double [] query,Map<double[],Pair<int[],double[]>> cache,boolean update_cache) {
		Pair<int[],double[]> p=null;
		if(cache!=null) p=cache.get(query);
		if(p==null) {
			List<KdTree2.Entry<Integer>> list=rbf_tree.nearestNeighbor(query, rbf_K, false);
			int K=list.size(); //just in case
			double d_min=Double.POSITIVE_INFINITY;
			double s2=2*rbf_sigma*rbf_sigma;
			for(int i=0;i<K;i++) {
				double d2=list.get(i).distance/s2;
				if(d2==0) { //rbf centers get exact value
					int [] idx=new int[1];
					double [] w=new double[1];
					idx[0]=list.get(i).value;
					w[0]=1.0;
					p=new Pair<>(idx,w);
					if(cache!=null && update_cache) cache.put(query, p);
					break;
				}
				if(d2<d_min) d_min=d2;
			}
			if(p==null) {
				int [] idx=new int[K];
				double [] w=new double[K];
				for(int i=0;i<K;i++) {
					double d2=list.get(i).distance/s2;
					w[i]=Math.exp(-d2+d_min);
					idx[i]=list.get(i).value;
				}
				p=new Pair<>(idx,w);
				if(cache!=null && update_cache) cache.put(query, p);
			}
		}
		return p;
	}
	static double rbf_kernel_query(int rbf_K,double rbf_sigma,KdTree2.SqrEuclid<Integer> rbf_tree,double [] query,double [] vals,
			Map<double[],Pair<int[],double[]>> cache,boolean update_cache) {
		Pair<int[],double[]> nn=rbf_kernel_cached(rbf_K,rbf_sigma,rbf_tree,query,cache,update_cache);
		int[] idx=nn.x;
		double[] weight=nn.y;
		double total_w=0;
		double value=0;
		for(int j=0;j<weight.length;j++) {
			value+=weight[j]*vals[idx[j]];
			total_w+=weight[j];
		}
		return value/total_w;
	}
	//minimize <p',u> over L1-ball of size epsilon centered on p
	//NOTE: assume p sum to 1
	static double min_1norm(double [] p,double [] u,double epsilon) {
		if(epsilon<=0) {
			double v=0;
			for(int i=0;i<u.length;i++) {
				v+=p[i]*u[i];
			}
			return v;
		}
		ArrayList<Pair<Double,Double>> arr=new ArrayList<>();
		for(int i=0;i<u.length;i++)
			arr.add(new Pair<>(p[i],u[i]));
		Collections.sort(arr,new Comparator<Pair<Double,Double>>() {
			public int compare(Pair<Double,Double> a,Pair<Double,Double> b) {
				if(a.y>b.y) return 1;
				if(a.y<b.y) return -1;
				return 0;
			}
		});
		arr.get(0).x=Math.min(1, arr.get(0).x+0.5*epsilon);
		double sum=0;
		for(int i=0;i<u.length;i++) {
			sum+=arr.get(i).x;
		}
		int j=u.length-1;
		while(sum>1 && j>=0) {
			sum-=arr.get(j).x;
			arr.get(j).x=Math.max(0, 1-sum);
			sum+=arr.get(j).x;
			j--;
		}
		double v=0;
		for(int i=0;i<u.length;i++) {
			v+=arr.get(i).x*arr.get(i).y;
		}
		return v;
	}
	static double norm_inf(double [] x,double []y) {
		double d=0;
		for(int i=0;i<x.length;i++) d=Math.max(d,Math.abs(x[i]-y[i]));
		return d;
	}
	
	/** 
	 * Generate training samples from policy
	 * 
	 * @param task
	 * @param feat (optional, null is not used)
	 * @param phi_tree (optional, null for random policy)
	 * @param rbf_K (KNN K)
	 * @param rbf_sigma
	 * @param gamma
	 * @param n_par (par over n_episodes)
	 * @param n_episodes
	 * @param episode_len
	 * @param exploration_p (0<=0<=1)
	 * @param train_tree (optional, null for random policy)
	 * @param pol_tr_map (optional, null for random policy)
	 * @param w (not used if random policy)
	 * @param robust_delta
	 * @param merged_phi_cache (optional, null for random policy)
	 * @return
	 * @throws Exception
	 */
	static HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>>
	get_training_set_pol(Task3 task,int feat_type,
			KdTree2.SqrEuclid<Integer> phi_tree,int rbf_K,double rbf_sigma,int phi_K,double phi_sigma,
			double gamma,int n_par,String out_iterx,int n_episodes,int episode_len,double exploration_p,
			KdTree2.SqrEuclid<Integer> train_tree,
			HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>> pol_tr_map,
			double [] w,double robust_delta,Map<double[],Pair<int[],double[]>> merged_phi_cache,
			ArrayList<String> log_all) 
					throws Exception {
		int n_action=task.numActions();
		int [] reach_goal=new int[n_episodes];

		HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>> all_map=new HashMap<>();
		for(int a=1;a<=n_action;a++) all_map.put(a, new ArrayList<>());
		
		ExecutorService exec=Executors.newFixedThreadPool(n_par);
		ArrayList<Future<HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>>>> arr_cache=new ArrayList<>();
		
		ArrayList<ArrayList<String>> logs=new ArrayList<>();
		for(int i=0;i<n_par;i++) {
			if(log_all!=null) logs.add(new ArrayList<>());
			arr_cache.add(exec.submit((new Callable<HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>>>() {
				int tr_idx;
				Callable<HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>>> init(int idx) {
					tr_idx=idx;
					return this;
				}
				public HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>> call() throws Exception {
					ArrayList<String> log=(log_all==null)?null:logs.get(tr_idx);
					StringBuilder str=null;
					HashMap<double[],Pair<int[],double[]>> phi_cache=(merged_phi_cache==null)?null:new HashMap<>(merged_phi_cache); 
					int tr_N=n_episodes;
					int st_i=(int)Math.ceil(tr_idx*tr_N/(double)n_par);
					int ed_i=(int)Math.ceil((tr_idx+1)*tr_N/(double)n_par);
					HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>> tr_map=new HashMap<>();
					for(int a=1;a<=n_action;a++) {
						ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>> sa_out=new ArrayList<>();
						tr_map.put(a, sa_out);
					}
					for(int i=st_i;i<ed_i;i++) {
						String id=String.format("%s %d",out_iterx,i);
						Task3 this_task=task.copy(id.hashCode());
						Random randgen=new Random(id.hashCode());
						if(log!=null) str=new StringBuilder(id);
						EuState raw_s=this_task.getInitialState();
						EuState s=this_task.extract_feature(raw_s,feat_type);
						boolean seen_goal=false;
						int max_a=0;
						for(int step=0;step<episode_len;step++) {
							if(log!=null) { 
								for(int j=0;j<s.x.length;j++) str.append(String.format(" %f", s.x[j]));
							}
							if(this_task.isGoal(raw_s)) {
								if(seen_goal) { 
									if(log!=null) str.append(String.format(" %d %f", max_a,0.0));
									continue;
								}
								reach_goal[i]=step+1;
								seen_goal=true;
							}
							max_a=randgen.nextInt(n_action)+1;
							if(train_tree!=null && randgen.nextDouble()<exploration_p) {
								Pair<int[],double[]> nn=rbf_kernel_cached(rbf_K,rbf_sigma,train_tree,s.x,null,false);
								int[] idx=nn.x;
								double[] weight=nn.y;
								double [] u=new double[weight.length];
								double total_w=0;
								for(int j=0;j<weight.length;j++) total_w+=weight[j];
								for(int j=0;j<weight.length;j++) weight[j]/=total_w;
								double max_q=Double.NEGATIVE_INFINITY;
								for(int a=1;a<=n_action;a++) {
									for(int j=0;j<idx.length;j++) {
										Pair<Double,EuState> outcome=pol_tr_map.get(a).get(idx[j]).y.get(0); //training state-action outcome
										u[j]=outcome.x + gamma*rbf_kernel_query(phi_K,phi_sigma,phi_tree,outcome.y.x, w, phi_cache,true);
									}
									//double q=min_1norm(weight,u,robust_delta);
									double q=min_1norm(weight,u,0);
									if(q>max_q) {
										max_q=q;
										max_a=a;
									}
								}
							}
							for(int a=1;a<=n_action;a++) { //try all actions
								ArrayList<Pair<Double,EuState>> outcome=new ArrayList<>();
								Pair<Double,EuState> next = this_task.nextState(raw_s, a);
								next.y=this_task.extract_feature(next.y,feat_type);
								outcome.add(next);
								tr_map.get(a).add(new Pair<>(s,outcome));
							}
							Pair<Double,EuState> next = this_task.nextState(raw_s, max_a);
							if(log!=null) str.append(String.format(" %d %f", max_a,next.x));
							raw_s=next.y;
							s=this_task.extract_feature(raw_s,feat_type);
						}
						if(log!=null) log.add(str.toString());
					}
					return tr_map;
				}
			}).init(i) ));
		}
		exec.shutdown();
		for(int g_i=0;g_i<arr_cache.size();g_i++) {
			HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>> map= arr_cache.get(g_i).get();
			for(int a=1;a<=n_action;a++) {
				all_map.get(a).addAll(map.get(a));
			}
		}
		if(log_all!=null) {
			for(int i=0;i<n_par;i++) log_all.addAll(logs.get(i));
		}
		//count goals
		int n_goal=0;
		for(int i=0;i<reach_goal.length;i++) {
			if(reach_goal[i]>0) n_goal++;
		}
		if(n_goal>0) System.out.println(String.format("Goal reached: %d out of %d",n_goal,n_episodes));
		return all_map;
	}

	//NOTE: ASSUME SINGLE OUTCOME SAMPLE PER sample_state
	// kernel + phi version
	// w[] holds V of each phi_state
	static double [] do_bellman_kernel_phi_cached(double [] w,double robust_delta,
			HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>> tr_map,
			int rbf_K,double rbf_sigma,int phi_K,double phi_sigma,KdTree2.SqrEuclid<Integer> phi_tree,double[][] phi_states,
			KdTree2.SqrEuclid<Integer> train_tree,
			double gamma,
			int n_par,HashMap<Integer,HashMap<double[],Pair<int[],double[]>>> all_cache,
			HashMap<Integer,HashMap<double[],Pair<int[],double[]>>> all_phi_cache) throws Exception {
		ExecutorService exec=Executors.newFixedThreadPool(n_par);
		ArrayList<Future<double []>> arr_cache=new ArrayList<>();
		int N=w.length;
		double [] nextw=new double[N];
		
		for(int i=0;i<n_par;i++) {
			arr_cache.add(exec.submit((new Callable<double[]>() {
				int tr_idx;
				Callable<double[]> init(int idx) {
					tr_idx=idx;
					return this;
				}
				public double[] call() throws Exception {
					HashMap<double[],Pair<int[],double[]>> cache=all_cache.get(tr_idx);
					HashMap<double[],Pair<int[],double[]>> phi_cache=all_phi_cache.get(tr_idx);
					int st_i=(int)Math.ceil(tr_idx*N/(double)n_par);
					int ed_i=(int)Math.ceil((tr_idx+1)*N/(double)n_par);
					for(int i=st_i;i<ed_i;i++) {
						Pair<int[],double[]> nn=rbf_kernel_cached(rbf_K,rbf_sigma,train_tree,phi_states[i],cache,true);
						int[] idx=nn.x;
						double[] weight=nn.y;
						double [] u=new double[weight.length];
						double total_w=0;
						for(int j=0;j<weight.length;j++) total_w+=weight[j];
						for(int j=0;j<weight.length;j++) weight[j]/=total_w;
						double max_q=Double.NEGATIVE_INFINITY; //V is tracked here
						for(int k=0;k<tr_map.size();k++) { //actions
							for(int j=0;j<idx.length;j++) {
								Pair<Double,EuState> outcome=tr_map.get(k+1).get(idx[j]).y.get(0); //training state-action outcome
								u[j]=outcome.x + gamma*rbf_kernel_query(phi_K,phi_sigma,phi_tree,outcome.y.x,w, phi_cache,true);
							}
							double q=min_1norm(weight,u,robust_delta);
							max_q=Math.max(max_q, q);
						}
						nextw[i]=max_q; //a from 1 ...
					}
					return nextw;
				}
			}).init(i) ));
		}
		exec.shutdown();
		for(int g_i=0;g_i<arr_cache.size();g_i++) arr_cache.get(g_i).get();
		return nextw;
	}
	
	//kernel+phi version
	static HashMap<Double,ArrayList<Pair<double[],String>>> test_policy_kernel_phi(Task3 ref_task,int feat_type,
			double [] tst_noise_rng,int max_step,String out_iterx,int tst_iter,
			int rbf_K,double rbf_sigma,int phi_K,double phi_sigma,KdTree2.SqrEuclid<Integer> phi_tree,
			KdTree2.SqrEuclid<Integer> train_tree,double gamma,double [] w,double robust_delta,
			HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>> tr_map,int n_par,
			Map<double[],Pair<int[],double[]>> merged_phi_cache) throws Exception {
		
		ExecutorService exec=Executors.newFixedThreadPool(n_par);
		ArrayList<Future<Map<Double,ArrayList<Pair<double [],String>>>>> arr=new ArrayList<>();
		
		//for(int tn_i=0;tn_i<tst_noise_rng.length;tn_i++) {
			//for(int i=0;i<tst_iter;i++) {
		for(int i=0;i<n_par;i++) {
			arr.add(exec.submit((new Callable<Map<Double,ArrayList<Pair<double[],String>>>>() {
				int tst_idx;
				Callable<Map<Double,ArrayList<Pair<double[],String>>>> init(int it) {
					tst_idx=it;
					return this;
				}
				public Map<Double,ArrayList<Pair<double[],String>>> call() throws Exception {
					HashMap<double[],Pair<int[],double[]>> phi_cache=new HashMap<>(merged_phi_cache);
					HashMap<Double,ArrayList<Pair<double[],String>>> res_map =new HashMap<>();
					
					int tst_N=tst_iter;
					int st_i=(int)Math.ceil(tst_idx*tst_N/(double)n_par);
					int ed_i=(int)Math.ceil((tst_idx+1)*tst_N/(double)n_par);
					for(int tn_i=0;tn_i<tst_noise_rng.length;tn_i++) {
						double tst_noise=tst_noise_rng[tn_i];
						ArrayList<Pair<double[],String>> res=new ArrayList<>();
						res_map.put(tst_noise, res);
						for(int i=st_i;i<ed_i;i++) {
							Task3 task=ref_task.copy(String.format("%s %d %f",out_iterx,i,tst_noise).hashCode());
							task.set_noise(tst_noise);

							StringBuilder str=new StringBuilder(String.format("%s %d %.2f", out_iterx,i,tst_noise));
							EuState raw_s=task.getInitialState();
							EuState s=task.extract_feature(raw_s,feat_type);
							int n_action=task.numActions();
							double reward=0,dreward=0;
							boolean seen_goal=false;
							int max_a=0;
							for(int step=0;step<max_step;step++) {
								for(int j=0;j<s.x.length;j++) str.append(String.format(" %f", s.x[j]));
								if(task.isGoal(raw_s)) {
									if(seen_goal) {
										str.append(String.format(" %d %f", max_a,0.0));
										continue;
									}
									seen_goal=true;
								}
								/////
								Pair<int[],double[]> nn=rbf_kernel_cached(rbf_K,rbf_sigma,train_tree,s.x,null,false);
								int[] idx=nn.x;
								double[] weight=nn.y;
								double [] u=new double[weight.length];
								double total_w=0;
								for(int j=0;j<weight.length;j++) total_w+=weight[j];
								for(int j=0;j<weight.length;j++) weight[j]/=total_w;
								max_a=0;
								double max_q=Double.NEGATIVE_INFINITY;
								for(int a=1;a<=n_action;a++) {
									for(int j=0;j<idx.length;j++) {
										Pair<Double,EuState> outcome=tr_map.get(a).get(idx[j]).y.get(0); //training state-action outcome
										u[j]=outcome.x + gamma*rbf_kernel_query(phi_K,phi_sigma,phi_tree,outcome.y.x, w, phi_cache,true);
									}
									//double q=min_1norm(weight,u,robust_delta);
									double q=min_1norm(weight,u,0);
									if(q>max_q) {
										max_q=q;
										max_a=a;
									}
								}
								Pair<Double,EuState> next = task.nextState(raw_s, max_a);
								reward+=next.x;
								dreward+=Math.pow(gamma,step)*next.x;

								str.append(String.format(" %d %f", max_a,next.x));

								raw_s=next.y;
								s=task.extract_feature(raw_s,feat_type);
							}
							double [] c=new double[2];
							c[0]=reward; c[1]=dreward;
							res.add(new Pair<>(c,str.toString()));
						}
					}
					return res_map;
				}
			}).init(i) ));
		}
		exec.shutdown();
		
		HashMap<Double,ArrayList<Pair<double[],String>>> map =new HashMap<>();
		for(int g_i=0;g_i<arr.size();g_i++) { 
			Map<Double,ArrayList<Pair<double[],String>>> p=arr.get(g_i).get();
			for(double noise : p.keySet()) {
				if(!map.containsKey(noise)) map.put(noise, p.get(noise));
				else map.get(noise).addAll(p.get(noise));
			}
		}
		return map;
	}
	
	static String toHexString(byte [] digest) {
		StringBuilder s=new StringBuilder();
		for(int i=0;i<digest.length;i++) {
			s.append(String.format("%02X",digest[i]&0xff));
		}
		return s.toString();
	}
	static String getSha1(double [][] X) throws Exception {
		MessageDigest md=MessageDigest.getInstance("SHA-1");
		byte[] buffer=TextUtil.double2str(X).getBytes();
		md.update(buffer,0,buffer.length);
		return(toHexString(md.digest()));
	}
	
	// training set generated by policy, phi generated by k-means
	static ArrayList<String> learn_and_test_policy(int n_policy,int policy_offset,
			int N_OUT_ITER,boolean tr_replace,double vi_epsilon,int vi_max_iter,
			int train_episodes,int train_episode_len,double exploration_p,int rbf_n_states,int k_means_seed,
			Task3 task,int feat_type,int n_dim,int rbf_K,double rbf_sigma,int phi_K,double phi_sigma,double gamma,int n_par,double robust_delta,
			double [] tst_noise_rng,int tst_max_step,int tst_iter,
			ArrayList<String> tr_log,ArrayList<String> phi_log,ArrayList<String> tst_log,
			HashMap<String,KMeans> kmeans_cache
			) throws Exception {
		int iter=-1;
		ArrayList<String> res_log=new ArrayList<>();

		for(int p_i=policy_offset;p_i<n_policy;p_i++) {
			long st_pol=System.currentTimeMillis();
			
			KdTree2.SqrEuclid<Integer> phi_tree=null;

			double [] w=null; //new double[rbf_n_states];
			KdTree2.SqrEuclid<Integer> train_tree=null;
			HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>> tr_map=null;
			Map<double[],Pair<int[],double[]>> merged_phi_cache=null;
			for(int out_iter=0;out_iter<N_OUT_ITER;out_iter++) {
				//training set
				long st=System.currentTimeMillis();
				HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>> new_tr_map =
						get_training_set_pol(task,feat_type,phi_tree,rbf_K,rbf_sigma,phi_K,phi_sigma,gamma, 
								n_par,String.format("%d %d",p_i,out_iter), train_episodes,train_episode_len,exploration_p,
								train_tree,tr_map,w,robust_delta,merged_phi_cache,tr_log);
				if(tr_replace || tr_map==null) tr_map=new_tr_map;
				else {
					for(int a : tr_map.keySet()) tr_map.get(a).addAll(new_tr_map.get(a));
					new_tr_map.clear(); new_tr_map=null;
				}
				train_tree=new KdTree2.SqrEuclid<Integer>(n_dim,tr_map.get(1).size()+10);
				ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>> a1_arr=tr_map.get(1);
				HashSet<String> train_set=new HashSet<>();
				for(int i=0;i<a1_arr.size();i++) { 
					train_tree.addPoint(a1_arr.get(i).x.x, i);
					train_set.add(TextUtil.double2str(a1_arr.get(i).x.x));
					train_set.add(TextUtil.double2str(a1_arr.get(i).y.get(0).y.x));
				}
				System.out.println(String.format("(pol %d/%d)(out iter %d/%d) Done train set (%.02f secs)",
						p_i+1,n_policy,out_iter+1,N_OUT_ITER,
						(System.currentTimeMillis()-st)/1000.0));
				
				//phi states
				st=System.currentTimeMillis();
				double [][] train_states=new double[train_set.size()][];
				int tr_i=0;
				for(String s : train_set) train_states[tr_i++]=TextUtil.str2vec(s);
				KMeans km_res=null;
				int km_hash=String.format("%d %d %d", k_means_seed,p_i,out_iter).hashCode();
				String km_key=String.format("%s %d %d",getSha1(train_states),Math.min(rbf_n_states,train_states.length),km_hash); 
				if(kmeans_cache==null || !kmeans_cache.containsKey(km_key)) {
					km_res=KMeans.kmeans_mult(train_states, Math.min(rbf_n_states,train_states.length), 
							1,Integer.MAX_VALUE, new Random(km_hash), n_par);
					if(kmeans_cache!=null) kmeans_cache.put(km_key, km_res);
				}
				else km_res=kmeans_cache.get(km_key);
				double [][] phi_states=km_res.centers;
				if(phi_log!=null) {
					StringBuilder str=new StringBuilder(String.format("%d %d", p_i,out_iter));
					for(int i=0;i<phi_states.length;i++) {
						for(int j=0;j<phi_states[i].length;j++) str.append(String.format(" %f", phi_states[i][j]));
					}
					phi_log.add(str.toString());
				}
				w=new double[phi_states.length];
				phi_tree=new KdTree2.SqrEuclid<Integer>(n_dim,phi_states.length+10);
				for(int i=0;i<phi_states.length;i++) phi_tree.addPoint(phi_states[i], i);
				System.out.println(String.format("(pol %d/%d)(out iter %d/%d) Done k-means (%d -> %d)(%d iters)(%.02f secs)",
						p_i+1,n_policy,out_iter+1,N_OUT_ITER,train_set.size(),phi_states.length,km_res.iter,
						(System.currentTimeMillis()-st)/1000.0));

				st=System.currentTimeMillis();
				HashMap<Integer,HashMap<double[],Pair<int[],double[]>>> all_cache=new HashMap<>();
				for(int i=0;i<n_par;i++) all_cache.put(i, new HashMap<>());
				HashMap<Integer,HashMap<double[],Pair<int[],double[]>>> all_phi_cache=new HashMap<>();
				for(int i=0;i<n_par;i++) all_phi_cache.put(i, new HashMap<>());
				double err=0;
				for(iter=0;iter<vi_max_iter;iter++) {
					double [] nextw = do_bellman_kernel_phi_cached(w,robust_delta,tr_map,rbf_K,rbf_sigma,phi_K,phi_sigma,
							phi_tree,phi_states,train_tree,gamma,n_par,all_cache,all_phi_cache);

					err=norm_inf(nextw,w);
					w=nextw;
					if (err<vi_epsilon) break;	
				}
				System.out.println(String.format("Done value iteration err=%f #iter=%d (%.02f secs)",err,iter,(System.currentTimeMillis()-st)/1000.0));

				//test
				st=System.currentTimeMillis();
				//merge all phi cache
				merged_phi_cache=new HashMap<>(); //Collections.synchronizedMap(new HashMap<>());
				for(int i=0;i<n_par;i++) merged_phi_cache.putAll(all_phi_cache.get(i));
				HashMap<Double,ArrayList<Pair<double[],String>>> res=test_policy_kernel_phi(task,feat_type,tst_noise_rng,
						tst_max_step,String.format("%d %d", p_i,out_iter),tst_iter,rbf_K,rbf_sigma,phi_K,phi_sigma,
						phi_tree,train_tree,gamma,w,robust_delta,tr_map,n_par,merged_phi_cache);
				for(int tn_i=0;tn_i<tst_noise_rng.length;tn_i++) {
					ArrayList<Pair<double[],String>> res_arr=res.get(tst_noise_rng[tn_i]);
					double total_reward=0, total_dreward=0, total_reward2=0;
					for(Pair<double[],String> p2 : res_arr) {
						total_reward+=p2.x[0]; total_dreward+=p2.x[1];
						total_reward2+=p2.x[0]*p2.x[0];
						if(tst_log!=null) tst_log.add(p2.y);
					}
					double mean=total_reward/tst_iter;
					double std = Math.sqrt((total_reward2/tst_iter-mean*mean)*tst_iter/(tst_iter-1));

					res_log.add(String.format("%d %d %d %.2f %g %f %f %d %f",
							p_i,out_iter,iter, tst_noise_rng[tn_i], err, mean,total_dreward/tst_iter,tst_iter,std));
					System.out.println(res_log.get(res_log.size()-1));
				}
				System.out.println(String.format("Done testing err=%f #iter=%d (%.02f secs)",err,iter,(System.currentTimeMillis()-st)/1000.0));
			}
			System.out.println(String.format("(pol %d/%d) Done (%.02f secs)",
					p_i+1,n_policy,(System.currentTimeMillis()-st_pol)/1000.0));
					
		}
		return res_log;
	}
	
	static double [] random_vector(int dim,Random randgen) {
		double [] v=new double[dim];
		double n=0;
		for(int i=0;i<dim;i++) {
			v[i]=randgen.nextGaussian();
			n+=v[i]*v[i];
		}
		n=Math.sqrt(n);
		for(int i=0;i<dim;i++) v[i]/=n;
		return v;
	}
	
	//randgen needed only when noise>0
	static Pair<double[][],KdTree2.SqrEuclid<Integer>> get_rbf_tree(int n_dim,int rbf_n,double noise,Random randgen) {
		int N=(int)Math.round(Math.pow(rbf_n, n_dim));
		double [][] sample_states=new double[N][n_dim];
		KdTree2.SqrEuclid<Integer> rbf_tree=new KdTree2.SqrEuclid<>(n_dim, N+10);
		double width=1./rbf_n;
		for(int d=0;d<n_dim;d++) {
			int dups=(int)Math.round(Math.pow(rbf_n, n_dim-1-d));
			int reps=N/(rbf_n*dups);
			for(int k=0;k<reps;k++) {
				int offset=k*rbf_n*dups;
				for(int i=0;i<rbf_n;i++) {
					double x=i*width+0.5*width;
					for(int j=0;j<dups;j++) {
						sample_states[offset+i*dups+j][d]=x;
					}
				}
			}
		}
		for(int i=0;i<N;i++) {
			if(noise>0) {
				double []v=random_vector(n_dim,randgen);
				double mag=randgen.nextDouble();
				for(int k=0;k<n_dim;k++) { 
					sample_states[i][k]+= v[k]*noise*mag;
				}
			}
			rbf_tree.addPoint(sample_states[i], i);
		}
		return new Pair<>(sample_states,rbf_tree);
	}
	
	//Note: no feat_type, assume sample_states are raw states
	static HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>>
	get_training_set(Task3 task,double [][] sample_states,
			int tr_rep,int n_par) throws Exception {
		int n_action=task.numActions();

		HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>> all_map=new HashMap<>();
		for(int a=1;a<=n_action;a++) all_map.put(a, new ArrayList<>());
		
		ExecutorService exec=Executors.newFixedThreadPool(n_par);
		ArrayList<Future<HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>>>> arr_cache=new ArrayList<>();
		
		for(int i=0;i<n_par;i++) {
			arr_cache.add(exec.submit((new Callable<HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>>>() {
				int tr_idx;
				Callable<HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>>> init(int idx) {
					tr_idx=idx;
					return this;
				}
				public HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>> call() throws Exception {
					Task3 this_task=task.copy(tr_idx);
					int tr_N=sample_states.length;
					int st_i=(int)Math.ceil(tr_idx*tr_N/(double)n_par);
					int ed_i=(int)Math.ceil((tr_idx+1)*tr_N/(double)n_par);
					HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>> tr_map=new HashMap<>();
					for(int a=1;a<=n_action;a++) {
						ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>> sa_out=new ArrayList<>();
						tr_map.put(a, sa_out);
						for(int i=st_i;i<ed_i;i++) {
							EuState this_s=new EuState(sample_states[i]);
							ArrayList<Pair<Double,EuState>> outcome=new ArrayList<>();
							for(int k=0;k<tr_rep;k++) {
								outcome.add(this_task.nextState(this_s,a));
							}
							sa_out.add(new Pair<>(this_s,outcome));
						}
					}
					
					return tr_map;
				}
			}).init(i) ));
		}
		exec.shutdown();
		for(int g_i=0;g_i<arr_cache.size();g_i++) {
			HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>> map= arr_cache.get(g_i).get();
			for(int a=1;a<=n_action;a++) {
				all_map.get(a).addAll(map.get(a));
			}
		}
		return all_map;
	}
	
	
	// training set generated by grid, phi generated by grid
	static ArrayList<String> learn_and_test_policy_grid(int n_policy,
			double vi_epsilon,int vi_max_iter,int rand_seed,
			int train_grid_n,double train_grid_noise,int phi_grid_n,double phi_grid_noise,
			Task3 task,int feat_type,int n_dim,int rbf_K,double rbf_sigma,int phi_K,double phi_sigma,double gamma,int n_par,double robust_delta,
			double [] tst_noise_rng,int tst_max_step,int tst_iter,
			ArrayList<String> tr_log,ArrayList<String> phi_log,ArrayList<String> tst_log
			) throws Exception {
		int iter=-1;
		ArrayList<String> res_log=new ArrayList<>();

		for(int p_i=0;p_i<n_policy;p_i++) {
			long st_pol=System.currentTimeMillis();
			
			Random randgen=new Random(String.format("%d %d",rand_seed,p_i).hashCode());

			//training set
			long st=System.currentTimeMillis();
			Pair<double[][],KdTree2.SqrEuclid<Integer>> p_tr=get_rbf_tree(n_dim,train_grid_n,train_grid_noise,randgen);
			double[][] train_states=p_tr.x;
			KdTree2.SqrEuclid<Integer> train_tree=p_tr.y;
			HashMap<Integer,ArrayList<Pair<EuState,ArrayList<Pair<Double,EuState>>>>> tr_map=
					get_training_set(task,train_states, 1, n_par);
			if(tr_log!=null) {
				StringBuilder str=new StringBuilder(String.format("%d", p_i));
				for(int i=0;i<train_states.length;i++) {
					for(int j=0;j<train_states[i].length;j++) str.append(String.format(" %f", train_states[i][j]));
				}
				tr_log.add(str.toString());
			}
			System.out.println(String.format("(pol %d/%d) Done train set (%.02f secs)",
					p_i+1,n_policy,(System.currentTimeMillis()-st)/1000.0));

			//phi states
			st=System.currentTimeMillis();
			Pair<double[][],KdTree2.SqrEuclid<Integer>> p=get_rbf_tree(n_dim,phi_grid_n,phi_grid_noise,randgen);
			double[][] phi_states=p.x;
			KdTree2.SqrEuclid<Integer> phi_tree=p.y;
			if(phi_log!=null) {
				StringBuilder str=new StringBuilder(String.format("%d", p_i));
				for(int i=0;i<phi_states.length;i++) {
					for(int j=0;j<phi_states[i].length;j++) str.append(String.format(" %f", phi_states[i][j]));
				}
				phi_log.add(str.toString());
			}
			double [] w=new double[phi_states.length];
			System.out.println(String.format("(pol %d/%d) Done phi (%d)(%.02f secs)",
					p_i+1,n_policy,phi_states.length,(System.currentTimeMillis()-st)/1000.0));
					

			st=System.currentTimeMillis();
			HashMap<Integer,HashMap<double[],Pair<int[],double[]>>> all_cache=new HashMap<>();
			for(int i=0;i<n_par;i++) all_cache.put(i, new HashMap<>());
			HashMap<Integer,HashMap<double[],Pair<int[],double[]>>> all_phi_cache=new HashMap<>();
			for(int i=0;i<n_par;i++) all_phi_cache.put(i, new HashMap<>());
			double err=0;
			for(iter=0;iter<vi_max_iter;iter++) {
				double [] nextw = do_bellman_kernel_phi_cached(w,robust_delta,tr_map,rbf_K,rbf_sigma,phi_K,phi_sigma,
						phi_tree,phi_states,train_tree,gamma,n_par,all_cache,all_phi_cache);

				err=norm_inf(nextw,w);
				w=nextw;
				if (err<vi_epsilon) break;	
			}
			System.out.println(String.format("Done value iteration err=%f #iter=%d (%.02f secs)",err,iter,(System.currentTimeMillis()-st)/1000.0));

			//test
			st=System.currentTimeMillis();
			//merge all phi cache
			Map<double[],Pair<int[],double[]>> merged_phi_cache=new HashMap<>(); //Collections.synchronizedMap(new HashMap<>());
			for(int i=0;i<n_par;i++) merged_phi_cache.putAll(all_phi_cache.get(i));
			HashMap<Double,ArrayList<Pair<double[],String>>> res=test_policy_kernel_phi(task,feat_type,tst_noise_rng,
					tst_max_step,String.format("%d", p_i),tst_iter,rbf_K,rbf_sigma,phi_K,phi_sigma,phi_tree,train_tree,gamma,w,robust_delta,tr_map,n_par,
					merged_phi_cache);
			for(int tn_i=0;tn_i<tst_noise_rng.length;tn_i++) {
				ArrayList<Pair<double[],String>> res_arr=res.get(tst_noise_rng[tn_i]);
				double total_reward=0, total_dreward=0, total_reward2=0;
				for(Pair<double[],String> p2 : res_arr) {
					total_reward+=p2.x[0]; total_dreward+=p2.x[1];
					total_reward2+=p2.x[0]*p2.x[0];
					if(tst_log!=null) tst_log.add(p2.y);
				}
				double mean=total_reward/tst_iter;
				double std = Math.sqrt((total_reward2/tst_iter-mean*mean)*tst_iter/(tst_iter-1));

				res_log.add(String.format("%d %d %.2f %g %f %f %d %f",
						p_i,iter, tst_noise_rng[tn_i], err, mean,total_dreward/tst_iter,tst_iter,std));
				System.out.println(res_log.get(res_log.size()-1));
			}
			System.out.println(String.format("Done testing err=%f #iter=%d (%.02f secs)",err,iter,(System.currentTimeMillis()-st)/1000.0));
			
			System.out.println(String.format("(pol %d/%d) Done (%.02f secs)",
					p_i+1,n_policy,(System.currentTimeMillis()-st_pol)/1000.0));
					
		}
		return res_log;
	}
	
}
