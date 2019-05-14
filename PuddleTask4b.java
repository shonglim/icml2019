import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;


//based on PuddleTask4, for Task3

class PuddleTask4b implements Task3 {
	java.util.Random randgen;
	double [][] puddles;
	double [] initial_state;
	double NOISE_LEVEL;
	public PuddleTask4b(double [][]p,double []init_s,long seed,double noise) {
		NOISE_LEVEL=noise;
		randgen=new java.util.Random(seed);
		puddles = p;
		initial_state = init_s;
	}
	//shallow copy params
	public PuddleTask4b copy(long seed) {
		return new PuddleTask4b(puddles,initial_state,seed,NOISE_LEVEL);
	}
	public EuState getInitialState() {
		EuState s=new EuState(new double[initial_state.length]);
		int i;
		for(i=0;i<initial_state.length;i++) {
			s.x[i] = initial_state[i];
		}
		return s;
	}
	public int numActions() {
		return 4;
	}
	public boolean isGoal(EuState s) {
		if ((s.x[0]>=0.95)&&(s.x[1]>=0.95)) return true;
		return false;
	}
	private double getCost(EuState s,int a) {
		double r=1;
		double d;
		if(isGoal(s)) r=0;
		int i;
		for(i=0;i<puddles.length;i++) {
			d=java.awt.geom.Line2D.ptSegDist(puddles[i][0],puddles[i][1],
					puddles[i][2],puddles[i][3],
					s.x[0],s.x[1]);
			if(d<puddles[i][4]*(1.0+NOISE_LEVEL)) r=r+400*(puddles[i][4]*(1.0+NOISE_LEVEL)-d);
		}
		return r;
	}
	public Pair<Double,EuState> nextState(EuState s,int a) {
		if(isGoal(s)) return new Pair<>(0.0,s);
		EuState ns=new EuState(new double[2]);
		ns.x[0] = s.x[0];
		ns.x[1] = s.x[1];
		switch(a) {
		case 1:
			ns.x[0]+=randgen.nextGaussian()*0.01;
			ns.x[1]+=0.05+randgen.nextGaussian()*0.01;
			break;
		case 2:
			ns.x[0]+=randgen.nextGaussian()*0.01;
			ns.x[1]-=0.05+randgen.nextGaussian()*0.01;
			break;
		case 3:
			ns.x[1]+=randgen.nextGaussian()*0.01;
			ns.x[0]+=0.05+randgen.nextGaussian()*0.01;
			break;
		case 4:
			ns.x[1]+=randgen.nextGaussian()*0.01;
			ns.x[0]-=0.05+randgen.nextGaussian()*0.01;
			break;
		}
		ns.x[0] = Math.min(1,Math.max(0,ns.x[0]));
		ns.x[1]= Math.min(1,Math.max(0,ns.x[1]));
		return new Pair<>(-getCost(s,a),ns);
	}
    public EuState extract_feature(EuState s,int feat_type) {
    	return s;
    }
    public void set_noise(double noise) {
    	NOISE_LEVEL=noise;
    }
    
    public static void test1(int [] n_states_rng,double [] rd_rng,String RES_DIR,int n_par) throws Exception {
		double[][] puddle= { {0.1,0.75,0.5,0.75,0.1}, {0.45,0.8,0.45,0.4,0.1} };
		double [] init_s= {0.25,0.6};
		PuddleTask4b task=new PuddleTask4b(puddle,init_s,1,0);
		int n_dim=2;
		
		double [] tst_noise_rng={0,0.5,1,1.5,2};
		int rbf_K=20;
		double [] ln_sigma_rng={-5}; //{-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3};
		int phi_K=20;
		double [] ln_psig_rng={-4}; //{-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3};
		int n_policy = 4; //5
		int policy_offset=3;
		int n_outer=1;
		boolean tr_replace=true;
		int train_episodes=10;
		int train_ep_len=1000;
		int tst_iter=300;
		int tst_max_step=300;
		double exploration_p=0.25;
		double vi_epsilon=0.001;
		int vi_max_iter=100;
		HashMap<String,KMeans> km_cache=new HashMap<>();
		
		for(int n_i=0;n_i<n_states_rng.length;n_i++) {
			int n_states=n_states_rng[n_i];
			
			for(int rob_i=0;rob_i<rd_rng.length;rob_i++) {
				double robust_delta=rd_rng[rob_i];
			
				ArrayList<double[]> all_tr_log=new ArrayList<>();
				ArrayList<double[]> all_phi_log=new ArrayList<>();
				ArrayList<double[]> all_tst_log=new ArrayList<>();
				ArrayList<double[]> all_res_log=new ArrayList<>();

				for(int sigma_i=0;sigma_i<ln_sigma_rng.length;sigma_i++) {
					double rbf_sigma = Math.exp(ln_sigma_rng[sigma_i]);
					for(int psig_i=0;psig_i<ln_psig_rng.length;psig_i++) {
						double phi_sigma = Math.exp(ln_psig_rng[psig_i]);

						System.out.println();
						System.out.println(String.format(" ------------ %d %.02f %.02f %.02f >",n_states,robust_delta,ln_sigma_rng[sigma_i],ln_psig_rng[psig_i]));
						System.out.println();

						ArrayList<String> tr_log=new ArrayList<>();
						ArrayList<String> phi_log=new ArrayList<>();
						ArrayList<String> tst_log=new ArrayList<>();
						ArrayList<String> res_log=KernelRL.learn_and_test_policy(n_policy,policy_offset,
								n_outer,tr_replace,vi_epsilon,vi_max_iter,
								train_episodes,train_ep_len,exploration_p, n_states ,1,
								task,0,n_dim,rbf_K,rbf_sigma,phi_K,phi_sigma, 0.99, n_par,robust_delta,
								tst_noise_rng,tst_max_step,tst_iter, tr_log,phi_log,tst_log,km_cache);
						//for(String s : tr_log) all_tr_log.add(MatUtil.str2double(String.format("%d %.02f %.02f %.02f %s",n_states,robust_delta,ln_sigma_rng[sigma_i],ln_psig_rng[psig_i], s)));
						//for(String s : phi_log) all_phi_log.add(MatUtil.str2double(String.format("%d %.02f %.02f %.02f %s",n_states,robust_delta,ln_sigma_rng[sigma_i],ln_psig_rng[psig_i], s)));
						//for(String s : tst_log) all_tst_log.add(MatUtil.str2double(String.format("%d %.02f %.02f %.02f %s",n_states,robust_delta,ln_sigma_rng[sigma_i],ln_psig_rng[psig_i], s)));
						//for(String s : res_log) all_res_log.add(MatUtil.str2double(String.format("%d %.02f %.02f %.02f %s",n_states,robust_delta,ln_sigma_rng[sigma_i],ln_psig_rng[psig_i], s)));
					}
				}
				//MatUtil.write_to_mat(RES_DIR+String.format("out_%d_%.02f.mat",n_states,robust_delta), "r", all_tr_log);
				//MatUtil.write_to_mat(RES_DIR+String.format("out2_%d_%.02f.mat",n_states,robust_delta), "r", all_phi_log);
				//MatUtil.write_to_mat(RES_DIR+String.format("out3_%d_%.02f.mat",n_states,robust_delta), "r", all_tst_log);
				//MatUtil.write_to_mat(RES_DIR+String.format("out4_%d_%.02f.mat",n_states,robust_delta), "r", all_res_log);
			}
		}
    }

    //fixed grid version
    public static void test2(int [] tr_grid_rng,int [] phi_grid_rng,double [] rd_rng,String RES_DIR,int n_par) throws Exception {
		double[][] puddle= { {0.1,0.75,0.5,0.75,0.1}, {0.45,0.8,0.45,0.4,0.1} };
		double [] init_s= {0.25,0.6};
		PuddleTask4b task=new PuddleTask4b(puddle,init_s,1,0);
		int n_dim=2;
		
		double [] tst_noise_rng={0,0.5,1,1.5,2};
		int rbf_K=20;
		double [] ln_sigma_rng={-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3};
		int phi_K=20;
		double [] ln_psig_rng={-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3};
		int n_policy = 5;
		int tst_iter=30;
		int tst_max_step=300;
		double vi_epsilon=0.001;
		int vi_max_iter=100;
		double train_grid_noise=0;
		double phi_grid_noise=0;
		
		for(int tg_i=0;tg_i<tr_grid_rng.length;tg_i++) {
			int train_grid_n=tr_grid_rng[tg_i];

			for(int pg_i=0;pg_i<phi_grid_rng.length;pg_i++) {
				int phi_grid_n=phi_grid_rng[pg_i];
			
				for(int rob_i=0;rob_i<rd_rng.length;rob_i++) {
					double robust_delta=rd_rng[rob_i];

					ArrayList<double[]> all_tr_log=new ArrayList<>();
					ArrayList<double[]> all_phi_log=new ArrayList<>();
					ArrayList<double[]> all_tst_log=new ArrayList<>();
					ArrayList<double[]> all_res_log=new ArrayList<>();

					for(int sigma_i=0;sigma_i<ln_sigma_rng.length;sigma_i++) {
						double rbf_sigma = Math.exp(ln_sigma_rng[sigma_i]);
						for(int psig_i=0;psig_i<ln_psig_rng.length;psig_i++) {
							double phi_sigma = Math.exp(ln_psig_rng[psig_i]);

							System.out.println();
							System.out.println(String.format(" ------------ %d %d %.02f %.02f %.02f >",train_grid_n,phi_grid_n,robust_delta,ln_sigma_rng[sigma_i],ln_psig_rng[psig_i]));
							System.out.println();

							ArrayList<String> tr_log=new ArrayList<>();
							ArrayList<String> phi_log=new ArrayList<>();
							ArrayList<String> tst_log=new ArrayList<>();
							ArrayList<String> res_log=KernelRL.learn_and_test_policy_grid(n_policy,
									vi_epsilon,vi_max_iter,1,train_grid_n,train_grid_noise,phi_grid_n,phi_grid_noise,
									task,0,n_dim,rbf_K,rbf_sigma,phi_K,phi_sigma, 0.99, n_par,robust_delta,
									tst_noise_rng,tst_max_step,tst_iter, tr_log,phi_log,tst_log);
							//for(String s : tr_log) all_tr_log.add(MatUtil.str2double(String.format("%d %d %.02f %.02f %.02f %s",train_grid_n,phi_grid_n,robust_delta,ln_sigma_rng[sigma_i],ln_psig_rng[psig_i], s)));
							//for(String s : phi_log) all_phi_log.add(MatUtil.str2double(String.format("%d %d %.02f %.02f %.02f %s",train_grid_n,phi_grid_n,robust_delta,ln_sigma_rng[sigma_i],ln_psig_rng[psig_i], s)));
							//for(String s : tst_log) all_tst_log.add(MatUtil.str2double(String.format("%d %d %.02f %.02f %.02f %s",train_grid_n,phi_grid_n,robust_delta,ln_sigma_rng[sigma_i],ln_psig_rng[psig_i], s)));
							//for(String s : res_log) all_res_log.add(MatUtil.str2double(String.format("%d %d %.02f %.02f %.02f %s",train_grid_n,phi_grid_n,robust_delta,ln_sigma_rng[sigma_i],ln_psig_rng[psig_i], s)));
						}
					}
					//MatUtil.write_to_mat(RES_DIR+String.format("out_%d_%d_%.02f.mat",train_grid_n,phi_grid_n,robust_delta), "r", all_tr_log);
					//MatUtil.write_to_mat(RES_DIR+String.format("out2_%d_%d_%.02f.mat",train_grid_n,phi_grid_n,robust_delta), "r", all_phi_log);
					//MatUtil.write_to_mat(RES_DIR+String.format("out3_%d_%d_%.02f.mat",train_grid_n,phi_grid_n,robust_delta), "r", all_tst_log);
					//MatUtil.write_to_mat(RES_DIR+String.format("out4_%d_%d_%.02f.mat",train_grid_n,phi_grid_n,robust_delta), "r", all_res_log);
				}
			}
		}
    }
    
    public static void main(String [] args) throws Exception {
		String RES_DIR="result directory/";
		(new File(RES_DIR)).mkdirs();
		int n_par=6;
		
		//int [] n_states_rng = { 50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200};
		int [] n_states_rng = {200}; //{ 80,120,160,200 };
		int [] train_grid_rng = {20}; //,50};
		int [] phi_grid_rng = { 8};
		double [] rd_rng= {0,0.01,0.02,0.03,0.05,0.1,0.2,0.3,0.4,0.5,1,1.5,2}; //1-norm dist
		//rd_rng[0]=Double.parseDouble(args[0]);
		test1(n_states_rng,rd_rng,RES_DIR,n_par);
		//test2(train_grid_rng,phi_grid_rng,rd_rng,RES_DIR,n_par);

	}
}
