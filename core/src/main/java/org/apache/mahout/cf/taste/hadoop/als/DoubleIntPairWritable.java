package org.apache.mahout.cf.taste.hadoop.als;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;

public class DoubleIntPairWritable extends PairWritable<DoubleWritable, IntWritable> {
	public void DoubleIntPairWritable() {
		super(new DoubleWritable(), new IntWritable());
	}

	public void setFirst(double value) {
		super.setFirst(new DoubleWritable(value));
	}

	public void setSecond(int value) {
		super.setSecond(new IntWritable(value));
	}

}
