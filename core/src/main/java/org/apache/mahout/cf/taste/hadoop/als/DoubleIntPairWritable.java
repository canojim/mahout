package org.apache.mahout.cf.taste.hadoop.als;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;

public class DoubleIntPairWritable extends PairWritable<DoubleWritable, IntWritable> {
	public DoubleIntPairWritable() {
		super(new DoubleWritable(), new IntWritable());
	}

	public void setFirst(double value) {
		super.setFirst(new DoubleWritable(value));
	}

	public void setSecond(int value) {
		super.setSecond(new IntWritable(value));
	}

}
