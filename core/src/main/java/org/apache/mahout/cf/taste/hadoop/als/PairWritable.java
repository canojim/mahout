package org.apache.mahout.cf.taste.hadoop.als;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;
import com.google.common.collect.ComparisonChain;

public abstract class PairWritable<T1 extends WritableComparable, T2 extends WritableComparable> implements WritableComparable<PairWritable> {

	private T1 first;
	private T2 second;

	public PairWritable(T1 value1, T2 value2) {
		this.first = value1;
		this.second = value2;
	}	
	
	@Override
	public void readFields(DataInput in) throws IOException {
		first.readFields(in);
		second.readFields(in);		
	}

	@Override
	public void write(DataOutput out) throws IOException {
		first.write(out);
		second.write(out);
	}

	@Override
	public int compareTo(PairWritable o) {		
		int firstResult = first.compareTo(o.first);
		int secondResult = second.compareTo(o.second);
		if (firstResult != 0)
			return firstResult;
		else {
			return secondResult;
		}
	}

	@Override
	public int hashCode() {

		// http://stackoverflow.com/questions/10034328/hashcode-for-objects-with-only-integers
		
		final int prime = 31;
	    int result = 1;
	    	    
	    result = prime * result + first.hashCode();
	    result = prime * result + second.hashCode();
	    return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (obj instanceof PairWritable) {
			PairWritable w = (PairWritable) obj;
			return first.equals(w.first) && second.equals(w.second); 
		}
		return false;
	}

	public T1 getFirst() {
		return first;
	}

	public void setFirst(T1 first) {
		this.first = first;
	}

	public T2 getSecond() {
		return second;
	}

	public void setSecond(T2 second) {
		this.second = second;
	}

	@Override
	public String toString() {
		return first.toString() + "," + second.toString();
	}	
}
