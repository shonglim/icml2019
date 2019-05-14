import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

/* based on AcroTask4b, for Task3 interface */

//double pole balancing

class CartpoleTask2 implements Task3 {
	class CartState {
		double x,xdot,theta1,theta2,thetadot1,thetadot2;
		public CartState(double ix,double ixd,double t1,double t2,double td1,double td2) {
			x=ix; xdot=ixd;
			theta1 = t1;
			theta2 = t2;
			thetadot1 = td1;
			thetadot2 = td2;
		}
	}
	double [][] initial_state;
	double [] bounds;
	CartState y2,k1,k2,k3,k4;
	java.util.Random randgen;
	double NOISE_LEVEL;
	public CartpoleTask2(double [] init_s,double [] bds,long seed,double noise) {
		NOISE_LEVEL = noise;
		randgen = new java.util.Random(seed);
		initial_state=new double[1][];
		initial_state[0] = init_s;
		bounds=bds;
		y2=new CartState(0,0,0,0,0,0);
		k1=new CartState(0,0,0,0,0,0);
		k2=new CartState(0,0,0,0,0,0);
		k3=new CartState(0,0,0,0,0,0);
		k4=new CartState(0,0,0,0,0,0);
	}
	public CartpoleTask2(double [][] init_s,double [] bds,long seed,double noise) {
		NOISE_LEVEL = noise;
		randgen = new java.util.Random(seed);
		initial_state = init_s;
		bounds=bds;
		y2=new CartState(0,0,0,0,0,0);
		k1=new CartState(0,0,0,0,0,0);
		k2=new CartState(0,0,0,0,0,0);
		k3=new CartState(0,0,0,0,0,0);
		k4=new CartState(0,0,0,0,0,0);
	}
	//shallow copy params
	public CartpoleTask2 copy(long seed) {
		return new CartpoleTask2(initial_state,bounds,seed,NOISE_LEVEL);
	}
	public EuState getInitialState() {
		int idx=randgen.nextInt(initial_state.length);
		EuState s=new EuState(new double[initial_state[idx].length]);
		int i;
		for(i=0;i<initial_state[idx].length;i++) {
			s.x[i] = initial_state[idx][i];
		}
		return s;
	}
	public int numActions() {
		return 2;
	}
	public boolean isGoal(EuState s) {
		double x = (s.x[0]-0.5)*2*bounds[0];
		double theta1 = (s.x[2]-0.5)*2*bounds[2];
		double theta2 = (s.x[3]-0.5)*2*bounds[3];
		double cutoff=36*Math.PI/180.0;
		if(x<-2.4 || x>2.4 || theta1>cutoff || theta1<-cutoff || theta2>cutoff || theta2<-cutoff) return true;
		return false;
	}
	static void cartfunc(CartState s,double F,CartState ns,double h) {
		double mu_pi=0.000002;
		double rat1=mu_pi*s.thetadot1/0.05;
		double rat2=mu_pi*s.thetadot2/0.0005;
		double tF1=0.05*s.thetadot1*s.thetadot1*Math.sin(s.theta1) + 
				0.075*Math.cos(s.theta1)*(rat1 + 9.8*Math.sin(s.theta1));
		double tF2=0.0005*s.thetadot2*s.thetadot2*Math.sin(s.theta2) + 
				0.0075*Math.cos(s.theta2)*(rat2 + 9.8*Math.sin(s.theta2));
		double tm1=0.1*(1-0.75*Math.cos(s.theta1)*Math.cos(s.theta1));
		double tm2=0.01*(1-0.75*Math.cos(s.theta2)*Math.cos(s.theta2));
		double xdotdot=( F-0.0005*Math.signum(s.xdot)+tF1+tF2 )/(1+tm1+tm2);
		
		ns.x=s.xdot;
		ns.xdot= xdotdot;
		ns.theta1=s.thetadot1;
		ns.theta2=s.thetadot2;
		ns.thetadot1= -1.5*(xdotdot*Math.cos(s.theta1) + 9.8*Math.sin(s.theta1) + rat1);
		ns.thetadot2= -15*(xdotdot*Math.cos(s.theta2) + 9.8*Math.sin(s.theta2) + rat2);

		ns.x *= h;
		ns.xdot *= h;
		ns.theta1 *= h;
		ns.theta2 *= h;
		ns.thetadot1 *= h;
		ns.thetadot2 *= h;
	}

	public Pair<Double,EuState> nextState(EuState s,int a) {
		if(isGoal(s)) return new Pair<>(0.0,s);
		int i;
		double action;
		double h=0.01;
		if(a==1) action=-10+randgen.nextDouble()*NOISE_LEVEL;
		else action=10-randgen.nextDouble()*NOISE_LEVEL;
		CartState ns=new CartState((s.x[0]-0.5)*2*bounds[0],
				(s.x[1]-0.5)*2*bounds[1],
				(s.x[2]-0.5)*2*bounds[2],
				(s.x[3]-0.5)*2*bounds[3],
				(s.x[4]-0.5)*2*bounds[4],
				(s.x[5]-0.5)*2*bounds[5] );
		for(i=0;i<2;i++) {
			cartfunc(ns,action,k1,h);
			y2.x=ns.x + 0.5*k1.x;
			y2.xdot=ns.xdot + 0.5*k1.xdot;
			y2.theta1=ns.theta1 + 0.5*k1.theta1;
			y2.theta2=ns.theta2 + 0.5*k1.theta2;
			y2.thetadot1=ns.thetadot1 + 0.5*k1.thetadot1;
			y2.thetadot2=ns.thetadot2 + 0.5*k1.thetadot2;
			cartfunc(y2,action,k2,h);
			y2.x=ns.x + 0.5*k2.x;
			y2.xdot=ns.xdot + 0.5*k2.xdot;
			y2.theta1=ns.theta1 + 0.5*k2.theta1;
			y2.theta2=ns.theta2 + 0.5*k2.theta2;
			y2.thetadot1=ns.thetadot1 + 0.5*k2.thetadot1;
			y2.thetadot2=ns.thetadot2 + 0.5*k2.thetadot2;
			cartfunc(y2,action,k3,h);
			y2.x=ns.x + k3.x;
			y2.xdot=ns.xdot + k3.xdot;
			y2.theta1=ns.theta1 + k3.theta1;
			y2.theta2=ns.theta2 + k3.theta2;
			y2.thetadot1=ns.thetadot1 + k3.thetadot1;
			y2.thetadot2=ns.thetadot2 + k3.thetadot2;
			cartfunc(y2,action,k4,h);
			ns.x+=(k1.x + 2*k2.x + 2*k3.x + k4.x)/6;
			ns.xdot+=(k1.xdot + 2*k2.xdot + 2*k3.xdot + k4.xdot)/6;
			ns.theta1+=(k1.theta1 + 2*k2.theta1 + 2*k3.theta1 + k4.theta1)/6;
			ns.theta2+=(k1.theta2 + 2*k2.theta2 + 2*k3.theta2 + k4.theta2)/6;
			ns.thetadot1+=(k1.thetadot1 + 2*k2.thetadot1 + 2*k3.thetadot1 + k4.thetadot1)/6;
			ns.thetadot2+=(k1.thetadot2 + 2*k2.thetadot2 + 2*k3.thetadot2 + k4.thetadot2)/6;
		}
//		if(ns.thetadot1<-4*Math.PI) ns.thetadot1=-4*Math.PI;
//		else if(ns.thetadot1>4*Math.PI) ns.thetadot1=4*Math.PI;
//		if(ns.thetadot2<-9*Math.PI) ns.thetadot2=-9*Math.PI;
//		else if(ns.thetadot2>9*Math.PI) ns.thetadot2=9*Math.PI;
		
//		while(ns.theta1<-bounds[0]) ns.theta1+=2*Math.PI;
//		while(ns.theta1>bounds[0]) ns.theta1-=2*Math.PI;
//		while(ns.theta2<-bounds[1]) ns.theta2+=2*Math.PI;
//		while(ns.theta2>bounds[1]) ns.theta2-=2*Math.PI;

		s = new EuState(new double[6]);
		s.x[0] = ns.x/(2*bounds[0]) + 0.5;
		s.x[1] = ns.xdot/(2*bounds[1]) + 0.5;
		s.x[2] = ns.theta1/(2*bounds[2]) + 0.5;
		s.x[3] = ns.theta2/(2*bounds[3]) + 0.5;
		s.x[4] = ns.thetadot1/(2*bounds[4]) + 0.5;
		s.x[5] = ns.thetadot2/(2*bounds[5]) + 0.5;
		return new Pair<>(1.0,s);
	}

	public EuState extract_feature(EuState s,int feat_type) {
    	return s;
    }
    public void set_noise(double noise) {
    	NOISE_LEVEL=noise;
    }
	
    public static void test1(int [] n_states_rng,double [] rd_rng,double train_noise,String RES_DIR,int n_par) throws Exception {
    	int init_n_dim=6;
    	int init_grid_n=2;
    	double width=1.0/init_grid_n;
		int N=(int)Math.round(Math.pow(init_grid_n, init_n_dim));
		double [][] init_s=new double[N][init_n_dim];
		for(int d=0;d<init_n_dim;d++) {
			int dups=(int)Math.round(Math.pow(init_grid_n, init_n_dim-1-d));
			int reps=N/(init_grid_n*dups);
			for(int k=0;k<reps;k++) {
				int offset=k*init_grid_n*dups;
				for(int i=0;i<init_grid_n;i++) {
					double x=i*width+0.5*width;
					//x=(x-0.5)*0.75+0.5;
					for(int j=0;j<dups;j++) {
						init_s[offset+i*dups+j][d]=x;
					}
				}
			}
		}
//		for(int i=0;i<init_s.length;i++) {
//			for(int j=0;j<init_s[i].length;j++) System.out.print(String.format("%.04f ",init_s[i][j] ));
//			System.out.println();
//		}
//		System.exit(0);;
//		double [][] init_s= { {0.5,0.5,0.5,0.5,0.5,0.5},
//				{0.75,0.75,0.75,0.25,0.75,0.25},{0.25,0.25,0.75,0.25,0.75,0.25} };
		double [] bds= {2.4,2.4/5,36*Math.PI/180.0,36*Math.PI/180.0,150*Math.PI/180,150*Math.PI/180};
		CartpoleTask2 task=new CartpoleTask2(init_s,bds,1,train_noise);
		int n_dim=6;
				
		//double [] tst_noise_rng={train_noise}; //{0,2,4,6,8};
		double [] tst_noise_rng={1,2,3,4,5}; //{0,2,4,6,8};
		int rbf_K=20;
		double [] ln_sigma_rng={-2}; //{-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3};
		int phi_K=20;
		double [] ln_psig_rng={0}; //{-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3};
		//double [] rd_rng= {0.01}; //{0,0.05,0.1,0.2,0.4}; //1-norm dist
		int n_policy = 5;
		int policy_offset=4;
		int n_outer=1;
		boolean tr_replace=true;
		int train_episodes=300;
		int train_ep_len=100;
		int tst_iter=300;
		int tst_max_step=3000;
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

	public static void main(String [] args) throws Exception {
		String RES_DIR="result directory/";
		(new File(RES_DIR)).mkdirs();
		int n_par=6;
		
		double train_noise=1; //1;
		//int [] n_states_rng = { 25,50,100,150,200,250,300 };
		//int [] n_states_rng= {20,40,60,80,100,120,140,160,180,200};
		int [] n_states_rng = {50};
		//n_states_rng[0]=Integer.parseInt(args[0]);
		//double [] rd_rng= {0,0.01,0.02,0.05,0.1,0.2,0.4}; //1-norm dist
		//rd_rng[0]=Double.parseDouble(args[0]);
		//rd_rng[1]=2*rd_rng[0];
		double [] rd_rng=new double[20];
		for(int i=0;i<rd_rng.length;i++) rd_rng[i]=i/100.0;
		test1(n_states_rng,rd_rng,train_noise,RES_DIR,n_par);
	}
}
