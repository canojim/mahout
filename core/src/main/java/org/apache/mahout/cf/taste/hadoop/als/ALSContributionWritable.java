package org.apache.mahout.cf.taste.hadoop.als;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.MatrixWritable;

public class ALSContributionWritable implements Writable {

	private MatrixWritable A, b;
	
	public ALSContributionWritable() {
		this.A = new MatrixWritable();
		this.b = new MatrixWritable();
	}
	
	public ALSContributionWritable(MatrixWritable A, MatrixWritable b) {
		this.A =A;
		this.b = b;		
	}

	public ALSContributionWritable(Matrix A, Matrix b) {
		this.A =new MatrixWritable(A);
		this.b = new MatrixWritable(b);		
	}

	
	@Override
	public void readFields(DataInput in) throws IOException {
		A.readFields(in);
		b.readFields(in);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		A.write(out);
		b.write(out);		
	}

	public MatrixWritable getA() {
		return this.A;
	}

	public void setA(MatrixWritable A) {
		this.A = A;
	}

	public void setA(Matrix A) {
		this.A = new MatrixWritable(A);
	}


	public MatrixWritable getb() {
		return this.b;
	}

	public void setb(MatrixWritable b) {
		this.b = b;
	}

	public void setb(Matrix b) {
		this.b = new MatrixWritable(b);
	}	
}
