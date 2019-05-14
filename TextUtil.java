

import java.util.ArrayList;
import java.util.Base64;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;

public class TextUtil {
	// to base-36
	public static String double2str(double x) {
		return Long.toString(Double.doubleToLongBits(x), 36);
	}
	
	// Comma separated
	public static String double2str(double [] x) {
		String [] str=new String[x.length];
		for(int i=0;i<x.length;i++) str[i]=double2str(x[i]);
		return String.join(",", str);
	}
	
	// tilde separated rows
	public static String double2str(double [][] x) {
		String [] str=new String[x.length];
		for(int i=0;i<x.length;i++) str[i]=double2str(x[i]);
		return String.join("~",str);
	}
	public static double str2double(String str) {
		return Double.longBitsToDouble(Long.parseLong(str,36));
	}
	public static double [] str2vec(String str) {
		String [] v=str.split("\\,",-1);
		double [] x=new double[v.length];
		for(int i=0;i<v.length;i++) x[i]=str2double(v[i]);
		return x;
	}
	public static double [][] str2mat(String str) {
		String [] rows=str.split("~",-1);
		double [][] x=new double[rows.length][];
		for(int i=0;i<rows.length;i++) x[i]=str2vec(rows[i]);
		return x;
	}
	// Comma separated
	public static String bool2str(boolean [] x) {
		String [] str=new String[x.length];
		for(int i=0;i<x.length;i++) str[i]=(x[i]?"1":"0");
		return String.join(",", str);
	}
	// tilde separated rows
	public static String bool2str(boolean [][] x) {
		String [] str=new String[x.length];
		for(int i=0;i<x.length;i++) str[i]=bool2str(x[i]);
		return String.join("~",str);
	}
	public static boolean [] str2boolvec(String str) {
		String [] v=str.split("\\,",-1);
		boolean [] x=new boolean[v.length];
		for(int i=0;i<v.length;i++) x[i]=(v[i].equals("1")?true:false);
		return x;
	}
	public static boolean [][] str2boolmat(String str) {
		String [] rows=str.split("~",-1);
		boolean [][] x=new boolean[rows.length][];
		for(int i=0;i<rows.length;i++) x[i]=str2boolvec(rows[i]);
		return x;
	}
	public static String toSafeString(String str) {
		try {
			return Base64.getUrlEncoder().encodeToString(str.getBytes("Cp1252"));
		} catch(Exception ex) {
			return Base64.getUrlEncoder().encodeToString(str.getBytes());
		}
	}
	public static String fromSafeString(String str) {
		try {
			return new String(Base64.getUrlDecoder().decode(str),"Cp1252");
		} catch(Exception ex) {
			return new String(Base64.getUrlDecoder().decode(str));
		}
	}
	
	static <T extends Object> String col2string(CharSequence delimiter,Collection<T> arr) {
		ArrayList<String> ret=new ArrayList<>();
		for(T x : arr) ret.add(x.toString());
		return String.join(delimiter,ret);
	}
	
}
