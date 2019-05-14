import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

/* based on AcroTask4, for Task3 interface */

class AcroTask4b implements Task3 {
	class AcroState {
		double theta1,theta2,thetadot1,thetadot2;
		public AcroState(double t1,double t2,double td1,double td2) {
			theta1 = t1;
			theta2 = t2;
			thetadot1 = td1;
			thetadot2 = td2;
		}
	}
	double [] initial_state;
	double [] bounds;
	AcroState y2,k1,k2,k3,k4;
	java.util.Random randgen;
	double NOISE_LEVEL;
	public AcroTask4b(double [] init_s,double [] bds,long seed,double noise) {
		NOISE_LEVEL = noise;
		randgen = new java.util.Random(seed);
		initial_state = init_s;
		bounds=bds;
		y2=new AcroState(0,0,0,0);
		k1=new AcroState(0,0,0,0);
		k2=new AcroState(0,0,0,0);
		k3=new AcroState(0,0,0,0);
		k4=new AcroState(0,0,0,0);
	}
	//shallow copy params
	public AcroTask4b copy(long seed) {
		return new AcroTask4b(initial_state,bounds,seed,NOISE_LEVEL);
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
		return 3;
	}
	public boolean isGoal(EuState s) {
		double x0 = (s.x[0]-0.5)*2*bounds[0];
		double x1 = (s.x[1]-0.5)*2*bounds[1];
		double y2=-Math.cos(x0)-Math.cos(x0+x1);
		return (y2>1);
	}
	static void acrofunc(AcroState s,double tau,AcroState ns,double h) {
		double d1 = 3.5 + Math.cos(s.theta2);
		double d2 = 1.25 + 0.5*Math.cos(s.theta2);
		double phi2 = 4.9*Math.cos(s.theta1 + s.theta2 - Math.PI/2);
		double phi1 = -0.5* s.thetadot2 * s.thetadot2 * Math.sin(s.theta2) - 
				s.thetadot2 * s.thetadot1 * Math.sin(s.theta2) + 
				14.7*Math.cos(s.theta1 - Math.PI/2)+phi2;

		ns.theta1=s.thetadot1;
		ns.theta2=s.thetadot2;
		ns.thetadot2=(tau + (d2*phi1)/d1 - 
				0.5*s.thetadot1*s.thetadot1*Math.sin(s.theta2) - phi2) / 
				(1.25 - d2*d2/d1);
		ns.thetadot1= -(d2*ns.thetadot2 + phi1)/d1;

		ns.theta1 *= h;
		ns.theta2 *= h;
		ns.thetadot1 *= h;
		ns.thetadot2 *= h;
	}

	public Pair<Double,EuState> nextState(EuState s,int a) {
		if(isGoal(s)) return new Pair<>(0.0,s);
		int i;
		double action;
		double h=0.05;
		if(a==1) action=-1+randgen.nextDouble()*NOISE_LEVEL;
		else if(a==2) action = 0;
		else action=1-randgen.nextDouble()*NOISE_LEVEL;
		AcroState ns=new AcroState((s.x[0]-0.5)*2*bounds[0],
				(s.x[1]-0.5)*2*bounds[1],
				(s.x[2]-0.5)*2*bounds[2],
				(s.x[3]-0.5)*2*bounds[3]);
		for(i=0;i<4;i++) {
			acrofunc(ns,action,k1,h);
			y2.theta1=ns.theta1 + 0.5*k1.theta1;
			y2.theta2=ns.theta2 + 0.5*k1.theta2;
			y2.thetadot1=ns.thetadot1 + 0.5*k1.thetadot1;
			y2.thetadot2=ns.thetadot2 + 0.5*k1.thetadot2;
			acrofunc(y2,action,k2,h);
			y2.theta1=ns.theta1 + 0.5*k2.theta1;
			y2.theta2=ns.theta2 + 0.5*k2.theta2;
			y2.thetadot1=ns.thetadot1 + 0.5*k2.thetadot1;
			y2.thetadot2=ns.thetadot2 + 0.5*k2.thetadot2;
			acrofunc(y2,action,k3,h);
			y2.theta1=ns.theta1 + k3.theta1;
			y2.theta2=ns.theta2 + k3.theta2;
			y2.thetadot1=ns.thetadot1 + k3.thetadot1;
			y2.thetadot2=ns.thetadot2 + k3.thetadot2;
			acrofunc(y2,action,k4,h);
			ns.theta1+=(k1.theta1 + 2*k2.theta1 + 2*k3.theta1 + k4.theta1)/6;
			ns.theta2+=(k1.theta2 + 2*k2.theta2 + 2*k3.theta2 + k4.theta2)/6;
			ns.thetadot1+=(k1.thetadot1 + 2*k2.thetadot1 + 2*k3.thetadot1 + k4.thetadot1)/6;
			ns.thetadot2+=(k1.thetadot2 + 2*k2.thetadot2 + 2*k3.thetadot2 + k4.thetadot2)/6;
		}
		if(ns.thetadot1<-4*Math.PI) ns.thetadot1=-4*Math.PI;
		else if(ns.thetadot1>4*Math.PI) ns.thetadot1=4*Math.PI;
		if(ns.thetadot2<-9*Math.PI) ns.thetadot2=-9*Math.PI;
		else if(ns.thetadot2>9*Math.PI) ns.thetadot2=9*Math.PI;
		
		while(ns.theta1<-bounds[0]) ns.theta1+=2*Math.PI;
		while(ns.theta1>bounds[0]) ns.theta1-=2*Math.PI;
		while(ns.theta2<-bounds[1]) ns.theta2+=2*Math.PI;
		while(ns.theta2>bounds[1]) ns.theta2-=2*Math.PI;

		s = new EuState(new double[4]);
		s.x[0] = ns.theta1/(2*bounds[0]) + 0.5;
		s.x[1] = ns.theta2/(2*bounds[1]) + 0.5;
		s.x[2] = ns.thetadot1/(2*bounds[2]) + 0.5;
		s.x[3] = ns.thetadot2/(2*bounds[3]) + 0.5;
		return new Pair<>(-1.0,s);
	}

	public EuState extract_feature(EuState s,int feat_type) {
    	return s;
    }
    public void set_noise(double noise) {
    	NOISE_LEVEL=noise;
    }
	
    public static void test1(int [] n_states_rng,double [] rd_rng,double train_noise,String RES_DIR,int n_par) throws Exception {
		double [] init_s= {0.5,0.5,0.5,0.5};
		double [] bds= {Math.PI,Math.PI,4*Math.PI,9*Math.PI};
		AcroTask4b task=new AcroTask4b(init_s,bds,1,train_noise);
		int n_dim=4;
				
		double [] tst_noise_rng={.1,.2,.3,.4,.5}; //{0.1,0.2,0.4,0.6,0.8}; //{0,.1,.2,.3,.4}; //{0.1,0.2,0.4,0.6,0.8};
		int rbf_K=20;
		double [] ln_sigma_rng={-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3};
		int phi_K=1; //20;
		double [] ln_psig_rng={-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3};
		//double [] rd_rng= {0.01}; //{0,0.05,0.1,0.2,0.4}; //1-norm dist
		int n_policy = 5;
		int policy_offset=0;
		int n_outer=1;
		boolean tr_replace=true;
		int train_episodes=30;
		int train_ep_len=1000;
		int tst_iter=30; //300
		int tst_max_step=1000;
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
						ArrayList<String> res_log = KernelRL.learn_and_test_policy(n_policy,policy_offset,
								n_outer,tr_replace,vi_epsilon,vi_max_iter,
								train_episodes,train_ep_len,exploration_p, n_states ,1,
								task,0,n_dim,rbf_K,rbf_sigma,phi_K,phi_sigma, 0.99, n_par,robust_delta,
								tst_noise_rng,tst_max_step,tst_iter, tr_log,phi_log,tst_log,
								km_cache);
						//for(String s : tr_log) all_tr_log.add(MatUtil.str2double(String.format("%d %.03f %.02f %.02f %s",n_states,robust_delta,ln_sigma_rng[sigma_i],ln_psig_rng[psig_i], s)));
						//for(String s : phi_log) all_phi_log.add(MatUtil.str2double(String.format("%d %.03f %.02f %.02f %s",n_states,robust_delta,ln_sigma_rng[sigma_i],ln_psig_rng[psig_i], s)));
						//for(String s : tst_log) all_tst_log.add(MatUtil.str2double(String.format("%d %.03f %.02f %.02f %s",n_states,robust_delta,ln_sigma_rng[sigma_i],ln_psig_rng[psig_i], s)));
						//for(String s : res_log) all_res_log.add(MatUtil.str2double(String.format("%d %.03f %.02f %.02f %s",n_states,robust_delta,ln_sigma_rng[sigma_i],ln_psig_rng[psig_i], s)));
					}
				}
				//MatUtil.write_to_mat(RES_DIR+String.format("out_%d_%.03f.mat",n_states,robust_delta), "r", all_tr_log);
				//MatUtil.write_to_mat(RES_DIR+String.format("out2_%d_%.03f.mat",n_states,robust_delta), "r", all_phi_log);
				//MatUtil.write_to_mat(RES_DIR+String.format("out3_%d_%.03f.mat",n_states,robust_delta), "r", all_tst_log);
				//MatUtil.write_to_mat(RES_DIR+String.format("out4_%d_%.03f.mat",n_states,robust_delta), "r", all_res_log);
			}
		}
    	
    }

    //fixed grid version
    public static void test2(int [] tr_grid_rng,int [] phi_grid_rng,double [] rd_rng,String RES_DIR,int n_par) throws Exception {
		double [] init_s= {0.5,0.5,0.5,0.5};
		double [] bds= {Math.PI,Math.PI,4*Math.PI,9*Math.PI};
		AcroTask4b task=new AcroTask4b(init_s,bds,1,0);
		int n_dim=4;
		
		double [] tst_noise_rng={0,0.1,.2,.3,.4};
		int rbf_K=20;
		double [] ln_sigma_rng={-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3};
		int phi_K=20;
		double [] ln_psig_rng={-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3};
		int n_policy = 5;
		int tst_iter=30;
		int tst_max_step=1000;
		double vi_epsilon=0.001;
		int vi_max_iter=100;
		double train_grid_noise=0.1;
		double phi_grid_noise=0.1;
		
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
		
		double train_noise=0.1; //0.1;
		int [] n_states_rng = {300,600,900,1200,1500,3000,6000};
		//int [] n_states_rng = {300};
		int [] train_grid_rng = {15}; //,50};
		int [] phi_grid_rng = { 6,8,10,12,14};
		//double [] rd_rng= {0,0.01,0.02,0.05,0.1,0.2,0.4}; //1-norm dist
		//double [] rd_rng= {0.001,0.002,0.005,0.5,1,1.5,2}; //1-norm dist
		//double [] rd_rng={0.16,.17,.18,.19,.21,.22,.23,.24,.25};// {0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,.1,.5,1,1.5,2}; //1-norm dist
		double [] rd_rng= {0};
		//double [] rd_rng=new double[20];
		//for(int i=0;i<rd_rng.length;i++) rd_rng[i]=i/100.0;
		//rd_rng[0]=Double.parseDouble(args[0]);
		test1(n_states_rng,rd_rng,train_noise,RES_DIR,n_par);
		//test2(train_grid_rng,phi_grid_rng,rd_rng,RES_DIR,n_par);
	}
}
