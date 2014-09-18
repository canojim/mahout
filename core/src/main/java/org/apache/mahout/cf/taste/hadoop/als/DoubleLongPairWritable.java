package org.apache.mahout.cf.taste.hadoop.als;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;

import com.google.common.collect.ComparisonChain;

public class DoubleLongPairWritable implements WritableComparable<DoubleLongPairWritable> {

	private double first;
	private long second;
	
	public DoubleLongPairWritable() {
		this.first = 0;
		this.second = 0;
	}
	
	public DoubleLongPairWritable(double value1, long value2) {
		this.first = value1;
		this.second = value2;
	}	
	
	@Override
	public void readFields(DataInput in) throws IOException {
		first = in.readDouble();
		second = in.readLong();		
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeDouble(first);
		out.writeLong(second);
	}

	@Override
	public int compareTo(DoubleLongPairWritable o) {
		return ComparisonChain.start().compare(first, o.first)
		        .compare(second, o.second).result();		
	}

	@Override
	public int hashCode() {

		// http://stackoverflow.com/questions/10034328/hashcode-for-objects-with-only-integers
		
		final int prime = 31;
	    int result = 1;
	    	    
	    result = prime * result + (int)(Math.round(first));
	    result = prime * result + (int) second;
	    return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (obj instanceof DoubleLongPairWritable) {
			DoubleLongPairWritable w = (DoubleLongPairWritable) obj;
			return first == w.first && second == w.second; 
		}
		return false;
	}

	public double getFirst() {
		return first;
	}

	public void setFirst(double first) {
		this.first = first;
	}

	public long getSecond() {
		return second;
	}

	public void setSecond(long second) {
		this.second = second;
	}

	@Override
	public String toString() {
		return Double.toString(first) + "," + Long.toString(second);
	}


	
}
