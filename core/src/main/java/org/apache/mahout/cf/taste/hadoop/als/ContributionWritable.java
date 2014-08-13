package org.apache.mahout.cf.taste.hadoop.als;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.MatrixWritable;

public class ContributionWritable implements Writable {

	private MatrixWritable first, second;
	
	public ContributionWritable() {
		this.first = new MatrixWritable();
		this.second = new MatrixWritable();
	}
	
	public ContributionWritable(MatrixWritable m1, MatrixWritable m2) {
		this.first = m1;
		this.second = m2;		
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

	public MatrixWritable getFirst() {
		return first;
	}

	public void setFirst(MatrixWritable first) {
		this.first = first;
	}

	public MatrixWritable getSecond() {
		return second;
	}

	public void setSecond(MatrixWritable second) {
		this.second = second;
	}

	
}
