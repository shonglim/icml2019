

import java.io.Serializable;

public class Pair<T1,T2> implements Serializable {
	T1 x;
	T2 y;
	Pair(T1 ix,T2 iy) {
		x=ix; y=iy;
	}
	public int hashCode() {
		return x.hashCode()+y.hashCode();
	}
	public boolean equals(Object obj) {
		Pair<T1,T2> m=(Pair<T1,T2>) obj;
		if(m.x.equals(x) && m.y.equals(y)) return true;
		return false;
	}
}
