package org.apache.mahout.cf.taste.hadoop.als;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;

public class IndexFormatMapper extends Mapper<VarIntWritable, VarLongWritable, IntWritable, LongWritable> {

	@Override
	protected void map(VarIntWritable key, VarLongWritable value, Context context)
			throws IOException, InterruptedException {
		context.write(new IntWritable(key.get()), new LongWritable(value.get()));
	}

}
