import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class KMeans {
	int [] labels;
	int [] count;
	double sum_d; //sum of distance^2
	double [][] centers;
	int iter;
	double [][] compute_mean(double [][] X) {
		int n=X.length;
		int p=X[0].length;
		int K=count.length;
		double [][] M=new double[K][p];
		int i,j,c_i;
		for(j=0;j<n;j++) {
			c_i=labels[j];
			for(i=0;i<p;i++) M[c_i][i]+=X[j][i];
		}
		for(c_i=0;c_i<K;c_i++) {
			for(i=0;i<p;i++) M[c_i][i]/=count[c_i];
		}
		return M;
	}
	double [][] compute_std(double [][] X) {
		double [][] M=compute_mean(X);
		int n=X.length;
		int p=X[0].length;
		int K=count.length;
		double [][] S=new double[K][p];
		int i,j,c_i;
		for(j=0;j<n;j++) {
			c_i=labels[j];
			for(i=0;i<p;i++) S[c_i][i]+=(X[j][i]*X[j][i]);
		}
		for(c_i=0;c_i<K;c_i++) {
			for(i=0;i<p;i++) S[c_i][i]=Math.sqrt(S[c_i][i]/count[c_i]-M[c_i][i]*M[c_i][i]);
		}
		
		return S;
	}
	int disagreement(double [][] X,double epsilon) {
		int D=0;
		int i,j,k;
		int n=X.length;
		int p=X[0].length;
		double epsilon2=epsilon*epsilon;
		double dist;
		for(j=0;j<n;j++) {
			for(k=j+1;k<n;k++) {
				dist=0;
				for(i=0;i<p;i++) dist+=(X[j][i]-X[k][i])*(X[j][i]-X[k][i]);
				if(dist>epsilon2) {
					if(labels[j]==labels[k]) D++;
				}
				else {
					if(labels[j]!=labels[k]) D++;
				}
			}
		}
		return D;
	}
	static double [][] init_M(double [][] X,int K,Random randgen) {
		int n=X.length;
		int p=X[0].length;
		double [][] M=new double[K][p];
		int idx=(int)(randgen.nextDouble()*n); //random first center
		int i,j,c_i;
		for(i=0;i<p;i++) M[0][i]=X[idx][i];
		//kmeans++
		double [] min_dist=new double[n];
		for(j=0;j<n;j++) {
			min_dist[j]=0;
			for(i=0;i<p;i++) min_dist[j]+=(M[0][i]-X[j][i])*(M[0][i]-X[j][i]);
		}
		for(c_i=1;c_i<K;c_i++) {
			j=sample(min_dist,randgen);
			for(i=0;i<p;i++) M[c_i][i]=X[j][i]; //new center
			//update min_dist
			for(j=0;j<n;j++) {
				double dist=0;
				for(i=0;i<p;i++) dist+=(M[c_i][i]-X[j][i])*(M[c_i][i]-X[j][i]);
				min_dist[j]=(dist<min_dist[j])?dist:min_dist[j];
			}
		}
		return M;
	}
	static int sample(double [] w,Random randgen) {
		int j;
		int n=w.length; 
		double [] cum_d=new double[n];
		cum_d[0]=w[0];
		for(j=1;j<n;j++) cum_d[j]=cum_d[j-1]+w[j];
		double chosen_d=randgen.nextDouble()*cum_d[n-1];
		for(j=0;j<n;j++) {
			if(chosen_d<cum_d[j]) return j;
		}
		return n-1; //should never reach this!
	}
	static int [] label(double [][] X,double [][] M) {
		//assign label
		int K=M.length;
		int n=X.length;
		int p=X[0].length;
		int i,j,c_i;
		int [] labels=new int[n];
		double [] min_dist=new double[n];
		for(j=0;j<n;j++) {
			min_dist[j]=Double.MAX_VALUE;
			int min_c_i=0;
			for(c_i=0;c_i<K;c_i++) {
				double dist=0;
				for(i=0;i<p;i++) dist+=(M[c_i][i]-X[j][i])*(M[c_i][i]-X[j][i]);
				if(dist<min_dist[j]) {
					min_dist[j]=dist;
					min_c_i=c_i;
				}
			}
			labels[j]=min_c_i;
		}
		return labels;
	}
	static KMeans assign(double [][] X,double [][] M,Random randgen) {
		//assign label
		int K=M.length;
		int n=X.length;
		int p=X[0].length;
		int i,j,c_i;
		KMeans res=new KMeans();
		res.labels=new int[n];
		res.count=new int[K];
		res.sum_d=0;
		double [] min_dist=new double[n];
		for(j=0;j<n;j++) {
			min_dist[j]=Double.MAX_VALUE;
			int min_c_i=0;
			for(c_i=0;c_i<K;c_i++) {
				double dist=0;
				for(i=0;i<p;i++) dist+=(M[c_i][i]-X[j][i])*(M[c_i][i]-X[j][i]);
				if(dist<min_dist[j]) {
					min_dist[j]=dist;
					min_c_i=c_i;
				}
			}
			res.labels[j]=min_c_i;
			res.sum_d+=min_dist[j];
			res.count[min_c_i]++;
		}
		//enforce non-empty cluster
		for(c_i=0;c_i<K;c_i++) {
			if(res.count[c_i]==0) { //random sample if no member
				for(;;) {
					j=sample(min_dist,randgen);
					if(res.count[res.labels[j]]>1) break;
					min_dist[j]=0; //suppress single-member
				}
				res.count[res.labels[j]]--;
				res.labels[j]=c_i;
				res.count[c_i]=1;
				min_dist[j]=0;
			}
		}
		return res;
	}
	static KMeans assign(double [][] X,double [][] M,Random randgen,int n_par) throws Exception {
		//assign label
		int K=M.length;
		int n=X.length;
		int p=X[0].length;
		int j,c_i; //i
		KMeans res=new KMeans();
		res.labels=new int[n];
		res.count=new int[K];
		res.sum_d=0;
		double [] min_dist=new double[n];
		
		ExecutorService exec=Executors.newFixedThreadPool(n_par);
		ArrayList<Future<String>> arr_cache=new ArrayList<>();
		for(int n_i=0;n_i<n_par;n_i++) {
			arr_cache.add(exec.submit((new Callable<String>() {
				int tr_idx;
				Callable<String> init(int idx) {
					tr_idx=idx;
					return this;
				}
				public String call() throws Exception {
					int st_i=(int)Math.ceil(tr_idx*n/(double)n_par);
					int ed_i=(int)Math.ceil((tr_idx+1)*n/(double)n_par);
					int i,c_i;
					for(int j=st_i;j<ed_i;j++) {
						min_dist[j]=Double.MAX_VALUE;
						int min_c_i=0;
						for(c_i=0;c_i<K;c_i++) {
							double dist=0;
							for(i=0;i<p;i++) dist+=(M[c_i][i]-X[j][i])*(M[c_i][i]-X[j][i]);
							if(dist<min_dist[j]) {
								min_dist[j]=dist;
								min_c_i=c_i;
							}
						}
						res.labels[j]=min_c_i;
					}
					return "";
				}
			}).init(n_i) ));
		}
		exec.shutdown();
		for(int g_i=0;g_i<arr_cache.size();g_i++) {
			arr_cache.get(g_i).get();
		}
		
		for(j=0;j<n;j++) {
			res.sum_d+=min_dist[j];
			res.count[res.labels[j]]++;
		}
		
		//enforce non-empty cluster
		for(c_i=0;c_i<K;c_i++) {
			if(res.count[c_i]==0) { //random sample if no member
				for(;;) {
					j=sample(min_dist,randgen);
					if(res.count[res.labels[j]]>1) break;
					min_dist[j]=0; //suppress single-member
				}
				res.count[res.labels[j]]--;
				res.labels[j]=c_i;
				res.count[c_i]=1;
				min_dist[j]=0;
			}
		}
		return res;
	}
	/***********
	 *  X must not contain duplicates 
	 */
	static KMeans kmeans(double [][] X,int K) {
		return kmeans(X,K,100,new Random());
	}
	static KMeans kmeans(double [][] X,int K,int max_iter,Random randgen) {
		int n=X.length;
		int j;
		
		
//		if(is_patho(X)) {
//			
//		}
		
		
		double [][] M=init_M(X,K,randgen);
		KMeans res=assign(X,M,randgen);
		for(int iter=0;iter<max_iter;iter++) {
			M=res.compute_mean(X);
			KMeans new_res=assign(X,M,randgen);
			new_res.iter=iter+1;
			for(j=0;j<n;j++) if(res.labels[j]!=new_res.labels[j]) break;
			if(j==n) {
				new_res.centers=M;
				return new_res; //no change, done!
			}
			res=new_res;
		}
		return res;
	}
	/***********
	 *  X must not contain duplicates 
	 */
	static KMeans kmeans(double [][] X,int K,int max_iter,Random randgen,int n_par) throws Exception {
		int n=X.length;
		int j;
		
		
//		if(is_patho(X)) {
//			
//		}
		
		
		double [][] M=init_M(X,K,randgen);
		KMeans res=assign(X,M,randgen,n_par);
		for(int iter=0;iter<max_iter;iter++) {
			M=res.compute_mean(X);
			KMeans new_res=assign(X,M,randgen,n_par);
			new_res.iter=iter+1;
			for(j=0;j<n;j++) if(res.labels[j]!=new_res.labels[j]) break;
			if(j==n) {
				new_res.centers=M;
				return new_res; //no change, done!
			}
			res=new_res;
		}
		return res;
	}
//	static KMeans kmeans_mult(double [][] X,int K,int n_rep,Random randgen,int n_par) throws Exception {
//		KMeans min_res=kmeans(X,K,100,randgen,n_par);
//		for(int i=1;i<n_rep;i++) {
//			KMeans res=kmeans(X,K,100,randgen,n_par);
//			if(res.sum_d<min_res.sum_d) {
//				min_res=res;
//			}
//		}
//		return min_res;
//	}
	static KMeans kmeans_mult(double [][] X,int K,int n_rep,int max_iter,Random randgen,int n_par) throws Exception {
		KMeans min_res=kmeans(X,K,max_iter,randgen,n_par);
		for(int i=1;i<n_rep;i++) {
			KMeans res=kmeans(X,K,max_iter,randgen,n_par);
			if(res.sum_d<min_res.sum_d) {
				min_res=res;
			}
		}
		return min_res;
	}
}
