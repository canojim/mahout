package org.apache.mahout.cf.taste.hadoop.als;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;

import com.google.common.collect.ComparisonChain;

public class DoubleIntPairWritable implements WritableComparable<DoubleIntPairWritable> {

	private double first;
	private int second;
	
	public DoubleIntPairWritable() {
		this.first = 0;
		this.second = 0;
	}
	
	public DoubleIntPairWritable(double value1, int value2) {
		this.first = value1;
		this.second = value2;
	}	
	
	@Override
	public void readFields(DataInput in) throws IOException {
		first = in.readDouble();
		second = in.readInt();		
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeDouble(first);
		out.writeInt(second);
	}

	@Override
	public int compareTo(DoubleIntPairWritable o) {
		return ComparisonChain.start().compare(first, o.first)
		        .compare(second, o.second).result();		
	}

	@Override
	public int hashCode() {

		// http://stackoverflow.com/questions/10034328/hashcode-for-objects-with-only-integers
		
		final int prime = 31;
	    int result = 1;
	    	    
	    result = prime * result + (int)(Math.round(first));
	    result = prime * result + second;
	    return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (obj instanceof DoubleIntPairWritable) {
			DoubleIntPairWritable w = (DoubleIntPairWritable) obj;
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

	public int getSecond() {
		return second;
	}

	public void setSecond(int second) {
		this.second = second;
	}

	@Override
	public String toString() {
		return Double.toString(first) + "," + Integer.toString(second);
	}


	
}
