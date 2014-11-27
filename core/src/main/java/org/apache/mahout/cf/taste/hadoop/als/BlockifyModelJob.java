package org.apache.mahout.cf.taste.hadoop.als;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.LazyOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.VectorWritable;

public class BlockifyModelJob extends AbstractJob {

	static final String NUM_BLOCKS = "numberOfBlocks";

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new BlockifyModelJob(), args);
	}

	@Override
	public int run(String[] args) throws Exception {

		addInputOption();
		addOutputOption();

		addOption("queueName", null,
				"mapreduce queueName. (optional)", "default");		
		addOption("numBlocks", null, "number of blocks");

		Map<String, List<String>> parsedArgs = parseArguments(args);
		if (parsedArgs == null) {
			return -1;
		}

		int numBlocks = Integer.parseInt(getOption("numBlocks"));

		boolean succeeded = false;

		Job blockifyModel = prepareJob(getInputPath(), getOutputPath(),
				SequenceFileInputFormat.class, BlockifyMapper.class,
				IntWritable.class, VectorWritable.class,
				SequenceFileOutputFormat.class, "BlockifyModel");

		Configuration blockifyModelConf = blockifyModel.getConfiguration();
		
		blockifyModelConf.set(JobManager.QUEUE_NAME, getOption("queueName"));		
		blockifyModelConf.setInt(NUM_BLOCKS, numBlocks);

		LazyOutputFormat.setOutputFormatClass(blockifyModel,
				SequenceFileOutputFormat.class);
		for (int blockId = 0; blockId < numBlocks; blockId++) {
			MultipleOutputs.addNamedOutput(blockifyModel,
					Integer.toString(blockId), SequenceFileOutputFormat.class,
					IntWritable.class, VectorWritable.class);
		}

		succeeded = blockifyModel.waitForCompletion(true);
		if (!succeeded) {
			throw new IllegalStateException("blockifyModel job failed!");
		}

		return 0;
	}

	static class BlockifyMapper extends
			Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

		private MultipleOutputs<IntWritable, VectorWritable> out;
		private int numBlocks; 
				
		@Override
		protected void cleanup(Context context) throws IOException,
				InterruptedException {
			out.close();
		}
		
		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			out = new MultipleOutputs<IntWritable, VectorWritable>(context);
			numBlocks = context.getConfiguration().getInt(NUM_BLOCKS, 0);
		}		
		
		@Override
		protected void map(IntWritable key, VectorWritable value,
				Context context) throws IOException, InterruptedException {
			
			String blockId = Integer.toString(BlockPartitionUtil.getBlockID(key.get(), numBlocks));			
			out.write(blockId, key, value);
			
		}

	}
}
