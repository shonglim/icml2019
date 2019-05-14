class EuState {
	static java.util.Random randgen;
	static int LSH_k, LSH_L, LSH_P, LSH_M;
	static double [][][] LSH_ab;
	static int [][] LSH_r;
	public double [] x;
	public EuState(double [] ix) {
		x = ix;
	}
	public double getDist(EuState s) {
		double d = 0;
		int i;
		for(i=0;i<x.length;i++) {
			d+=(x[i]-s.x[i])*(x[i]-s.x[i]);
		}
		return Math.sqrt(d);
	}
	public String toString() {
		int i;
		String str="";
		for(i=0;i<x.length;i++) {
			if(i>0) str+=" ";
			str+=x[i];
		}
		return str;
	}
	public int getLSH_M() {
		return LSH_M;
	}
	public int [] hashKey(int [][]h) {
		int j,i;
		int [] key=new int[LSH_L];
		for(j=0;j<LSH_L;j++) {
			key[j]=0;
			for(i=0;i<LSH_k;i++) {
				key[j]+=LSH_r[j][i]*h[j][i];
			}
			key[j]%=LSH_P;
			if(key[j]<0) key[j]+=LSH_P;
			key[j]%=LSH_M;
		}
		return key;
	}
	public int [][] hashLSH(double R) {
		int [][] H=new int[LSH_L][LSH_k];
		int j,i,d,dim=x.length;
		double h;
		for(j=0;j<LSH_L;j++) {
			for(i=0;i<LSH_k;i++) {
				h=0;
				for(d=0;d<dim;d++) {
					h+=x[d]*LSH_ab[j][d][i]/R;
				}
				H[j][i] = (int)Math.floor(h + LSH_ab[j][dim][i]);
			}
		}
		return H;
	}
	public static void initLSH(int dim,int k,int L,int P,int M) {
		randgen=new java.util.Random();
		LSH_k = k;
		LSH_L = L;
		LSH_P = P;
		LSH_M = M;
		int j,d,i;
		LSH_ab=new double[L][dim+1][k];
		LSH_r=new int[L][k];
		for(j=0;j<L;j++) {
			for(d=0;d<dim;d++) {
				for(i=0;i<k;i++) {
					LSH_ab[j][d][i]=randgen.nextGaussian()/4;
				}
			}
			for(i=0;i<k;i++) {
				LSH_ab[j][dim][i]=randgen.nextDouble();
				LSH_r[j][i] = randgen.nextInt(P);
			}
		}
	}
}
