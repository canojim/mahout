package org.apache.mahout.cf.taste.hadoop.als;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.DoubleWritable;

public class LongDoublePairWritable extends PairWritable<LongWritable, DoubleWritable> {
	public void LongDoublePairWritable() {
		super(new LongWritable(), new DoubleWritable());
	}	
}
