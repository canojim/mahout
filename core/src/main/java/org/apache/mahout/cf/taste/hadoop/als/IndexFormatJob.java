package org.apache.mahout.cf.taste.hadoop.als;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;

public class IndexFormatJob extends AbstractJob {

	@Override
	public int run(String[] args) throws Exception {
		
		if (args.length < 2) {
			System.out.println("Usage: IndexFormatJob inputPath outputPath");
		}
		
		Path inputPath = new Path(args[0]);
		Path outputPath = new Path(args[1]);
		
		Job formatJob = prepareJob(inputPath, outputPath, SequenceFileInputFormat.class,
				IndexFormatMapper.class, IntWritable.class, LongWritable.class, TextOutputFormat.class);
		
		boolean status = formatJob.waitForCompletion(true);
		
		if (status)
			return 0;
		else
			return -1;
	}

	public static void main(String[] args) throws Exception {		
		ToolRunner.run(new IndexFormatJob(), args);
	}
}
