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
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.jobcontrol.ControlledJob;
import org.apache.hadoop.mapreduce.lib.jobcontrol.JobControl;
import org.apache.hadoop.mapreduce.lib.map.MultithreadedMapper;
import org.apache.hadoop.mapreduce.lib.output.LazyOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
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
public class BlockRecommenderJob extends AbstractJob {

	private static final Logger log = LoggerFactory
			.getLogger(BlockParallelALSFactorizationJob.class);

	static final String NUM_RECOMMENDATIONS = BlockRecommenderJob.class
			.getName() + ".numRecommendations";
	static final String USER_FEATURES_PATH = BlockRecommenderJob.class
			.getName() + ".userFeatures";
	static final String ITEM_FEATURES_PATH = BlockRecommenderJob.class
			.getName() + ".itemFeatures";
	static final String MAX_RATING = BlockRecommenderJob.class.getName()
			+ ".maxRating";
	static final String USER_INDEX_PATH = BlockRecommenderJob.class.getName()
			+ ".userIndex";
	static final String ITEM_INDEX_PATH = BlockRecommenderJob.class.getName()
			+ ".itemIndex";
	static final String RECOMMEND_FILTER_PATH = BlockRecommenderJob.class
			.getName() + ".recommendFilterPath";
	static final String NUM_USER_BLOCK = BlockRecommenderJob.class.getName()
			+ ".numUserBlock";

	static final String NUM_ITEM_BLOCK = BlockRecommenderJob.class.getName()
			+ ".numItemBlock";
	
	static final int DEFAULT_NUM_RECOMMENDATIONS = 10;

	public static void main(String[] args) throws Exception {
		ToolRunner.run(new BlockRecommenderJob(), args);
	}

	@Override
	public int run(String[] args) throws Exception {

		addInputOption();
		addOption("userFeatures", null, "path to the user feature matrix", true);
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
		addOption("numUserBlock", null, "number of user blocks",
				String.valueOf(10));
		addOption("numItemBlock", null, "number of item blocks",
				String.valueOf(10));
		addOutputOption();

		Map<String, List<String>> parsedArgs = parseArguments(args);
		if (parsedArgs == null) {
			return -1;
		}

		//
		int numUserBlock = Integer.parseInt(getOption("numUserBlock"));
		int numItemBlock = Integer.parseInt(getOption("numItemBlock"));
		
		/* create block-wise user rating */
		Job userRatings = prepareJob(getInputPath(),
				pathToUserRatingsByUserBlock(), Mapper.class,
				IntWritable.class, VectorWritable.class,
				MergeUserVectorsReducer.class, IntWritable.class,
				VectorWritable.class);

		// use multiple output to support block
		LazyOutputFormat.setOutputFormatClass(userRatings,
				SequenceFileOutputFormat.class);
		for (int blockId = 0; blockId < numUserBlock; blockId++) {
			MultipleOutputs.addNamedOutput(userRatings,
					Integer.toString(blockId), SequenceFileOutputFormat.class,
					IntWritable.class, VectorWritable.class);
		}

		//userRatings.setCombinerClass(MergeVectorsCombiner.class);
		userRatings.getConfiguration().setInt(NUM_USER_BLOCK, numUserBlock);

		log.info("Starting userRatings job");
		boolean succeeded = userRatings.waitForCompletion(true);
		if (!succeeded) {
			throw new IllegalStateException("userRatings job failed");
		}

		String userFeaturesPath = getOption("userFeatures");

		//for (int blockId = 0; blockId < numUserBlock; blockId++) {
		int blockId = 23;
		//TODO: Revert
			JobControl control = new JobControl("BlockRecommenderJob");
			// process each user block
			for (int itemBlockId = 0; itemBlockId < numItemBlock; itemBlockId++) {
				
		
				Path blockUserRatingsPath = new Path(pathToUserRatingsByUserBlock()
						.toString() + "/" + Integer.toString(blockId) + "/" + Integer.toString(itemBlockId) + "-r-*");
				Path blockUserFeaturesPath = new Path(userFeaturesPath + "/"
						+ Integer.toString(blockId) + "-r-*");
				Path blockItemFeaturesPath = new Path(getOption("itemFeatures") + "/"
						+ Integer.toString(itemBlockId) + "-r-*");
				Path blockUserIDIndexPath = new Path(getOption("userIDIndex") + "/"
						+ Integer.toString(blockId) + "-r-*");
				Path blockItemIDIndexPath = new Path(getOption("itemIDIndex") + "/"
						+ Integer.toString(itemBlockId) + "-r-*");
				Path blockOutputPath = new Path(getTempPath().toString() + "/result/"
						+ Integer.toString(blockId));

				Job blockPrediction = prepareJob(blockUserRatingsPath,
						blockOutputPath, SequenceFileInputFormat.class,
						MultithreadedSharingMapper.class, IntWritable.class,
						RecommendedItemsWritable.class, SequenceFileOutputFormat.class);
	
				Configuration blockPredictionConf = blockPrediction
						.getConfiguration();
				int numThreads = Integer.parseInt(getOption("numThreads"));
				blockPredictionConf.setInt(NUM_RECOMMENDATIONS,
						Integer.parseInt(getOption("numRecommendations")));
				blockPredictionConf.set(USER_FEATURES_PATH,
						blockUserFeaturesPath.toString());
				blockPredictionConf.set(ITEM_FEATURES_PATH,
						blockItemFeaturesPath.toString());
				blockPredictionConf.set(MAX_RATING, getOption("maxRating"));
				blockPredictionConf.setInt(NUM_USER_BLOCK, numUserBlock);
				blockPredictionConf.setInt(NUM_ITEM_BLOCK, numItemBlock);
				boolean usesLongIDs = Boolean
						.parseBoolean(getOption("usesLongIDs"));
				if (usesLongIDs) {
					blockPredictionConf.set(
							ParallelALSFactorizationJob.USES_LONG_IDS,
							String.valueOf(true));
					blockPredictionConf.set(USER_INDEX_PATH,
							blockUserIDIndexPath.toString());
					blockPredictionConf.set(ITEM_INDEX_PATH,
							getOption("itemIDIndex"));
				}
	
				String rcmPath = getOption("recommendFilterPath");
				if (rcmPath != null)
					blockPredictionConf.set(RECOMMEND_FILTER_PATH, rcmPath);
	
				MultithreadedMapper.setMapperClass(blockPrediction,
						BlockPredictionMapper.class);
				MultithreadedMapper.setNumberOfThreads(blockPrediction, numThreads);
	
				control.addJob(new ControlledJob(blockPredictionConf));
			}
			
			Thread t = new Thread(control);
			log.info("Starting " + numUserBlock + " block prediction jobs.");
			t.start();

			while (!control.allFinished()) {
				Thread.sleep(1000);
			}

			List<ControlledJob> failedJob = control.getFailedJobList();

			if (failedJob != null && failedJob.size() > 0) {
				control.stop();
				throw new IllegalStateException("control job failed: " + failedJob);
			} else {
				log.info("control job finished");
			}

			control.stop();
			
			Path blocksOutputPath = new Path(getTempPath().toString() + "/result/");
			Job blockRecommendation = prepareJob(blocksOutputPath,
					getOutputPath(), SequenceFileInputFormat.class,
					MultithreadedSharingMapper.class, IntWritable.class,
					RecommendedItemsWritable.class, TextOutputFormat.class); 

		//}


		return 0;
	}

	static class MergeUserVectorsReducer extends
			Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {

		private MultipleOutputs<IntWritable, VectorWritable> out;
		private int numUserBlocks;
		private final VectorWritable resultValue = new VectorWritable();

		@Override
		protected void setup(Context ctx) throws IOException,
				InterruptedException {
			out = new MultipleOutputs<IntWritable, VectorWritable>(ctx);
			numUserBlocks = ctx.getConfiguration().getInt(NUM_USER_BLOCK, 10);
		}

		@Override
		public void reduce(IntWritable key, Iterable<VectorWritable> values,
				Context ctx) throws IOException, InterruptedException {
						
			Vector merged = VectorWritable.merge(values.iterator()).get();
			//System.out.println("key: " + key + "merged vector: " + merged.size());
			
			resultValue.set(new SequentialAccessSparseVector(merged));
			int blockId = BlockPartitionUtil.getBlockID(key.get(),
					numUserBlocks);
			out.write(Integer.toString(blockId), key, resultValue);
		}

		@Override
		protected void cleanup(Context context) throws IOException,
				InterruptedException {
			out.close();
		}
	}

	private Path pathToUserRatingsByUserBlock() {
		return getOutputPath("userRatingsByUserBlock");
	}
}
