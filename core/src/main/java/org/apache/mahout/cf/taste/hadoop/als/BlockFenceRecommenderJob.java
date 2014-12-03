/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.cf.taste.hadoop.als;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.map.MultithreadedMapper;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>
 * Computes the top-N recommendations per user from a decomposition of the
 * rating matrix
 * </p>
 * 
 * <p>
 * Command line arguments specific to this class are:
 * </p>
 * 
 * <ol>
 * <li>--input (path): Directory containing the vectorized user ratings</li>
 * <li>--output (path): path where output should go</li>
 * <li>--numRecommendations (int): maximum number of recommendations per user
 * (default: 10)</li>
 * <li>--maxRating (double): maximum rating of an item</li>
 * <li>--numThreads (int): threads to use per mapper, (default: 1)</li>
 * </ol>
 */
@Deprecated
public class BlockFenceRecommenderJob extends AbstractJob {
	
	private static final Logger log = LoggerFactory
			.getLogger(BlockParallelALSFactorizationJob.class);
	
	static final int DEFAULT_NUM_RECOMMENDATIONS = 10;	
	static final String FORMAT_CSV = "csv";
	
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new BlockFenceRecommenderJob(), args);
	}

	@Override
	public int run(String[] args) throws Exception {

		addInputOption();
		addOption("itemFeatures", null, "path to the item feature matrix", true);
		addOption("numRecommendations", null,
				"number of recommendations per user",
				String.valueOf(DEFAULT_NUM_RECOMMENDATIONS));
		addOption("maxRating", null, "maximum rating available", true);
		addOption("numThreads", null, "threads per mapper", String.valueOf(1));
		addOption("usesLongIDs", null,
				"input contains long IDs that need to be translated");
		addOption("userIDIndex", null,
				"index for user long IDs (necessary if usesLongIDs is true)");
		addOption("itemIDIndex", null,
				"index for user long IDs (necessary if usesLongIDs is true)");
		addOption("recommendFilterPath", null,
				"filter recommended user id. (optional)");
		addOption("queueName", null,
				"mapreduce queueName. (optional)", "default");		
		addOption("numUserBlock", null, "number of user blocks",
				String.valueOf(10));
		addOption("numItemBlock", null, "number of item blocks",
				String.valueOf(10));
		addOption("outputFormat", null, "outputformat: csv or raw", FORMAT_CSV);		
		
		addOutputOption();

		Map<String, List<String>> parsedArgs = parseArguments(args);
		if (parsedArgs == null) {
			return -1;
		}

		Configuration defaultConf = new Configuration();
		FileSystem fs = FileSystem.get(defaultConf);
		
		boolean succeeded = false;
		//String rcmPath = getOption("recommendFilterPath");
		boolean usesLongIDs = Boolean
				.parseBoolean(getOption("usesLongIDs"));
		
		// int numUserBlock = Integer.parseInt(getOption("numUserBlock"));
		int numItemBlock = Integer.parseInt(getOption("numItemBlock"));
		String outputFormat = getOption("outputFormat");
		
		Path allUserFeaturesPath = getInputPath(); 		
		Path sumAllUserFeaturePath = new Path(getTempPath().toString() + "/sumAllUserFeature");
		int blockId = 0;
		
		if (!fs.exists(new Path(sumAllUserFeaturePath.toString() + "/_SUCCESS"))) {
			Job sumAllUserFeature = prepareJob(allUserFeaturesPath,
					sumAllUserFeaturePath, VectorSumMapper.class,
					IntWritable.class, VectorWritable.class,
					VectorSumReducer.class, IntWritable.class,
					VectorWritable.class);
			
			log.info("Starting sumAllUserFeature job.");
			succeeded = sumAllUserFeature.waitForCompletion(true);
			if (!succeeded) {
				throw new IllegalStateException("sumAllUserFeature job failed");
			}				
		}		
		
		JobManager jobMgr = new JobManager();
		jobMgr.setQueueName(getOption("queueName"));
		
		for (int itemBlockId = 0; itemBlockId < numItemBlock; itemBlockId++) {
	
			Path blockItemFeaturesPath = new Path(getOption("itemFeatures") + "/"
					+ Integer.toString(itemBlockId) + "-*-*");
			Path blockItemIDIndexPath = new Path(getOption("itemIDIndex") + "/"
					+ Integer.toString(itemBlockId) + "-r-*");
			Path blockOutputPath = new Path(getTempPath().toString() + "/result/"
					+ Integer.toString(blockId) + "x" + Integer.toString(itemBlockId));

			if (!fs.exists(new Path(blockOutputPath.toString() + "/_SUCCESS"))) {
				Job blockPrediction = prepareJob(sumAllUserFeaturePath,
						blockOutputPath, SequenceFileInputFormat.class,
						MultithreadedSharingMapper.class, LongWritable.class,
						LongDoublePairWritable.class, SequenceFileOutputFormat.class);
				
				Configuration blockPredictionConf = blockPrediction
						.getConfiguration();
				int numThreads = Integer.parseInt(getOption("numThreads"));
				blockPredictionConf.set(BlockRecommenderJob.ITEM_FEATURES_PATH,
						blockItemFeaturesPath.toString());

				blockPredictionConf.setInt(BlockRecommenderJob.NUM_RECOMMENDATIONS,
						Integer.parseInt(getOption("numRecommendations")));
				blockPredictionConf.set(BlockRecommenderJob.MAX_RATING, getOption("maxRating"));				
				
				if (usesLongIDs) {
					blockPredictionConf.set(
							ParallelALSFactorizationJob.USES_LONG_IDS,
							String.valueOf(true));
					blockPredictionConf.set(BlockRecommenderJob.ITEM_INDEX_PATH,
							blockItemIDIndexPath.toString());
				}
	
				MultithreadedMapper.setMapperClass(blockPrediction,
						BlockFencePredictionMapper.class);

				MultithreadedMapper.setNumberOfThreads(blockPrediction, numThreads);
	
				jobMgr.addJob(blockPrediction);					
			}
		}
		
		boolean allFinished = jobMgr.waitForCompletion();
		
		if (!allFinished) {
			throw new IllegalStateException("Some BlockPredictionMapper jobs failed.");
		}
		
		Path blocksReduceInputPath = new Path(getTempPath().toString() + "/result/*/");
		Path blocksReduceOutputPath = new Path(getOutputPath().toString() + "/recomd/");
		
		if (!fs.exists(new Path(blocksReduceOutputPath.toString() + "/_SUCCESS"))) {
			Job blockRecommendation = null;
			Configuration blockRecommendationConf = null;
			if (FORMAT_CSV.equals(outputFormat)) {
				blockRecommendation = prepareJob(blocksReduceInputPath,
						blocksReduceOutputPath, SequenceFileInputFormat.class,
						Mapper.class, LongWritable.class,
						LongDoublePairWritable.class, 
						RecommendCSVReducer.class,
						LongWritable.class, Text.class, 
						TextOutputFormat.class); 	
				
				blockRecommendationConf = blockRecommendation.getConfiguration();
				blockRecommendationConf.set("mapred.textoutputformat.separator", ",");
			} else {
				blockRecommendation = prepareJob(blocksReduceInputPath,
					blocksReduceOutputPath, SequenceFileInputFormat.class,
					Mapper.class, LongWritable.class,
					LongDoublePairWritable.class, 
					RecommendReducer.class,
					LongWritable.class, RecommendedItemsWritable.class, 
					TextOutputFormat.class);
				
				blockRecommendationConf = blockRecommendation.getConfiguration();
			}
			
			blockRecommendationConf.set(JobManager.QUEUE_NAME, getOption("queueName"));
			blockRecommendationConf.setInt(BlockRecommenderJob.NUM_RECOMMENDATIONS,
					Integer.parseInt(getOption("numRecommendations")));
			blockRecommendationConf.setInt(BlockRecommenderJob.MAX_RATING, Integer.parseInt(getOption("maxRating")));

			
			log.info("Starting blockRecommendation (reduce) job.");
			succeeded = blockRecommendation.waitForCompletion(true);
			if (!succeeded) {
				throw new IllegalStateException("blockRecommendation (reduce) job failed");
			}
		}			
		
		return 0;
	}

	static class VectorSumMapper extends
			Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

		final IntWritable outKey = new IntWritable(1);
		
		@Override
		protected void map(IntWritable key, VectorWritable value,
				Context context)
				throws IOException, InterruptedException {
			context.write(outKey, value);
		}
				
	}
			
	static class VectorSumReducer extends
			Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {

		@Override
		protected void reduce(IntWritable key, Iterable<VectorWritable> features,
				Context ctx)
				throws IOException, InterruptedException {
			
			VectorWritable vw = VectorWritable.mergeSum(features.iterator());			
			
			ctx.write(key, vw);
		}
				
	}
			
}
