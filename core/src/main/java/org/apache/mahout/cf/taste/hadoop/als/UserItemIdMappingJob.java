/*
 * This code was written by Gordon Linoff of Data Miners, Inc. (http://www.data-miners.com on 25 Nov 2009.
 * 
 * The purpose of the code is to assign row numbers to data stored in HDFS files.
 * 
 * The method uses two passes of the map/reduce framework.  The first pass accomplishes two things:
 *      (1) It stores the data in a sequence file using the partition number and row number within
 *          the partition as keys.
 *      (2) It calculates the number of rows in each partition.
 *      
 * This information is then used to obtain an offset for each partition.
 * 
 * The final pass adds the offset to the row number to obtain the final row number for each row.
 */

package org.apache.mahout.cf.taste.hadoop.als;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.LazyOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class UserItemIdMappingJob extends AbstractJob {
	
	private static final Logger log = LoggerFactory
			.getLogger(BlockParallelALSFactorizationJob.class);
	
	static final String NUM_BLOCKS = UserItemIdMappingJob.class.getName() + "numberOfBlocks";
	static final String ID_FILE_DIR = "id_file_dir";
	static final String ID_MAPPING_CUMSUM_NUMVALS = "cumsum.numvals";	// number of partitions in the first pass
	static final String ID_MAPPING_CUMSUM_NTHVALUE = "cumsum.nthvalue";			// offset for each partition
	
	private int numThreadsPerSolver;
	private int numBlocks;
	
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new UserItemIdMappingJob(), args);
	}

	@Override
	public int run(String[] args) throws Exception {
		addInputOption();
		addOutputOption();
		addOption("numThreadsPerSolver", null, "threads per solver mapper",
				String.valueOf(1));
		addOption("queueName", null,
				"mapreduce queueName. (optional)", "default");				

		addOption("numBlocks", null, "number of User Block");

		Map<String, List<String>> parsedArgs = parseArguments(args);
		if (parsedArgs == null) {
			return -1;
		}

		numThreadsPerSolver = Integer
				.parseInt(getOption("numThreadsPerSolver"));
		numBlocks = Integer.parseInt(getOption("numBlocks"));

		Configuration defaultConf = new Configuration();
		FileSystem fs = FileSystem.get(defaultConf);
		
		boolean succeeded = false;
		
		if (!fs.exists(new Path(getOutputPath("idSummary").toString() + "/_SUCCESS"))) {
			Job userIdSummary = prepareJob(getInputPath(),
					getOutputPath("idSummary"), TextInputFormat.class,
					NewKeyOutputMap.class, IntWritable.class,
					LongWritable.class, KeySummaryReduce.class,
					IntWritable.class, LongWritable.class,
					SequenceFileOutputFormat.class);
			
			userIdSummary.setCombinerClass(KeySummaryReduce.class);		
			userIdSummary.getConfiguration().set(ID_FILE_DIR, getOutputPath("idFile").toString());
			
			log.info("Starting Id Summary Job");
			succeeded = userIdSummary.waitForCompletion(true);
			if (!succeeded) {
				throw new IllegalStateException("Id Summary Job failed!");
			}
		}
		
		if (!fs.exists(new Path(getOutputPath("idIndex").toString() + "/_SUCCESS"))) {			
			Job idMapping = prepareJob(getOutputPath("idFile"),
					getOutputPath("idIndex"), SequenceFileInputFormat.class,
					CalcRowNumberMap.class, IntWritable.class, LongWritable.class,
					SequenceFileOutputFormat.class);

			FileStatus[] files = fs.globStatus(new Path(getOutputPath("idSummary").toString() + "/p*"));
			int numvals = 0;
			long cumsum = 0;
			for (FileStatus fstat : files) {
				
				SequenceFile.Reader reader = new SequenceFile.Reader(fs, fstat.getPath(), defaultConf); 
				//SequenceFile.Reader reader = new SequenceFile.Reader(defaultConf, Reader.file(fstat.getPath()));

				IntWritable key = new IntWritable();
			    LongWritable val = new LongWritable();

			    while (reader.next(key, val)) {
			    	//System.out.println("DEBUG: key=" + key.get() + ", value=" + val.get());
					idMapping.getConfiguration().set(ID_MAPPING_CUMSUM_NTHVALUE + numvals++, key.get() + "\t" + val.get() + "\t" + cumsum);
					cumsum += val.get();
			    }

			    reader.close();
			}
			
			idMapping.getConfiguration().setInt(ID_MAPPING_CUMSUM_NUMVALS, numvals);
			
			idMapping.getConfiguration().setInt(NUM_BLOCKS, numBlocks);
			idMapping.getConfiguration().set(JobManager.QUEUE_NAME, getOption("queueName"));
			
			// use multiple output to suport block
			LazyOutputFormat.setOutputFormatClass(idMapping, SequenceFileOutputFormat.class);
			for (int blockId = 0; blockId < numBlocks; blockId++) {
				MultipleOutputs.addNamedOutput(idMapping, Integer.toString(blockId), SequenceFileOutputFormat.class, 
						IntWritable.class, LongWritable.class);
			}
			
			idMapping.setCombinerClass(KeySummaryReduce.class);
		
			log.info("Starting Map LongID for user job");
			succeeded = idMapping.waitForCompletion(true);
			if (!succeeded) {
				throw new IllegalStateException("MapLoingID-User job failed!");
			}
		}
				
		return 0;

	}
	
	// This Map function does two things.  First it outputs the data in a new
	// sequence file, creating a key from the partition id and row numbers within
	// the partition.
	static class NewKeyOutputMap extends Mapper<LongWritable, Text, IntWritable, LongWritable> {

		private SequenceFile.Writer sfw;
		private IntWritable partitionid = new IntWritable(0);
		private Text outkey = new Text("");
		private long localrownum = 0;
		private LongWritable localrownumvalue = new LongWritable(0);

		@Override
		protected void setup(Context ctx) throws IOException,
			InterruptedException {
			String saverecordsdir; 
			partitionid.set(ctx.getConfiguration().getInt("mapred.task.partition", 0));
			System.out.println("DEBUG: (Partition Id = " + partitionid.get() + ")");
			saverecordsdir = new String(ctx.getConfiguration().get(ID_FILE_DIR));
			if (saverecordsdir.endsWith("/")) {
				saverecordsdir.substring(0, saverecordsdir.length() - 1);
			}
//			try {
				FileSystem fs = FileSystem.get(ctx.getConfiguration());
				sfw = SequenceFile.createWriter(fs, ctx.getConfiguration(),
						new Path(saverecordsdir+"/"+String.format("records%05d", partitionid.get())),
						Text.class,	Text.class);
//			} catch (Exception e) {
//				e.printStackTrace();
//			}
		} // setup

		@Override
		protected void map(LongWritable key, Text value, Context ctx)
				throws IOException, InterruptedException {
			localrownumvalue.set(++localrownum);
			ctx.write(partitionid, localrownumvalue);
			outkey.set(partitionid.toString()+";" + localrownum);
			sfw.append(outkey, value);
		} // map()
	} // class NewKeyOutputMap
	
	
	// This reduce counts the number of records in each partition by
	// taking the maximum of the row numbers.  This reduce function is
	// used both as a combiner and reducer.
	static class KeySummaryReduce extends Reducer<IntWritable, LongWritable, IntWritable, LongWritable> {
		
		@Override
		protected void reduce(IntWritable key, Iterable<LongWritable> values, Context ctx)
			throws IOException, InterruptedException {
			
			LongWritable maxval = new LongWritable(Long.MIN_VALUE);
			Iterator<LongWritable> iter = values.iterator();
			while (iter.hasNext()) {
				long val = iter.next().get();
				if (maxval.get() < val) {
					maxval.set(val);
				}
			}
			
			ctx.write(key, maxval);
		}
	} // KeySummaryReduce()
	
	
	// This map function adds the appropriate offset to the row number and
	// outputs the results in the results directory.
	static class CalcRowNumberMap extends Mapper<Text, Text, IntWritable, LongWritable> {

		HashMap<String, Integer> offsets = new HashMap<String, Integer>();
		
		private int numBlocks;
		private MultipleOutputs<IntWritable,LongWritable> out;

		@Override
		protected void cleanup(Context context) throws IOException,
				InterruptedException {
			out.close();
		}
		
		@Override
		protected void setup(Context ctx) throws IOException,
				InterruptedException {
			int numvals = 0;
			numvals = ctx.getConfiguration().getInt(ID_MAPPING_CUMSUM_NUMVALS, 0);
			numBlocks = ctx.getConfiguration().getInt(NUM_BLOCKS, 10);
			
			offsets.clear();
			for (int i = 0; i < numvals; i++) {
				String val = ctx.getConfiguration().get(ID_MAPPING_CUMSUM_NTHVALUE + i);
				String[] parts = val.split("\t");
				offsets.put(parts[0], Integer.parseInt(parts[2]));
			}
			
			out = new MultipleOutputs<IntWritable,LongWritable>(ctx);
		} // configure

		@Override
	 	protected void map(Text key, Text value, Context ctx) 
	 			throws IOException, InterruptedException {
			String[] parts = key.toString().split(";");
			int rownum = Integer.parseInt(parts[1]) + offsets.get(parts[0]);
			
			int blockId = BlockPartitionUtil.getBlockID(rownum, numBlocks);
			
			//System.out.println("key: " + key + " value: " + value + " rownum: " + rownum + " blockId: " + blockId);
			
			out.write(Integer.toString(blockId), new IntWritable(rownum), 
					new LongWritable(Long.parseLong(value.toString())));
		} // map()
	} // class CalcRowNumberMap

}