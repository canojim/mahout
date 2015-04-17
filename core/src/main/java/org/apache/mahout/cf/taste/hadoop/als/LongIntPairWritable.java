package org.apache.mahout.cf.taste.hadoop.als;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.IntWritable;

public class LongIntPairWritable extends PairWritable<LongWritable, IntWritable> {
	public LongIntPairWritable() {
		super(new LongWritable(), new IntWritable());
	}	
}
