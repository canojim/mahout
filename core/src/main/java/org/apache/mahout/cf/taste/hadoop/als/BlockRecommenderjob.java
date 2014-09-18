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
import java.io.InputStream;
import java.io.StringWriter;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.CompressionCodecFactory;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.map.MultithreadedMapper;
import org.apache.hadoop.mapreduce.lib.output.LazyOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

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
	
	static final String QUEUE_NAME = "mapred.job.queue.name";
	
	private String defaultQueue = "pp_risk_dst"; //TODO: Pass in from arguments 
	
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
		
		/* create block-wise user ratings */
		Job userRatingsByUserBlock = prepareJob(getInputPath(),
				pathToUserRatingsByUserBlock(), UserRatingsByUserBlockMapper.class,
				IntWritable.class, VectorWritable.class,
				Reducer.class, IntWritable.class,
				VectorWritable.class);

		// use multiple output to support block
		LazyOutputFormat.setOutputFormatClass(userRatingsByUserBlock,
				SequenceFileOutputFormat.class);
		for (int userBlockId = 0; userBlockId < numUserBlock; userBlockId++) {
			for (int itemBlockId = 0; itemBlockId < numItemBlock; itemBlockId++) {

					String outputName = Integer.toString(userBlockId) + "x" + 
							Integer.toString(itemBlockId);
					MultipleOutputs.addNamedOutput(userRatingsByUserBlock,
							outputName, SequenceFileOutputFormat.class,
							IntWritable.class, VectorWritable.class);
			}
		}

		//userRatings.setCombinerClass(MergeVectorsCombiner.class);
		Configuration userRatingsConf = userRatingsByUserBlock.getConfiguration();
		
		userRatingsConf.setInt(NUM_USER_BLOCK, numUserBlock);
		userRatingsConf.setInt(NUM_ITEM_BLOCK, numItemBlock);
		
		String rcmPath = getOption("recommendFilterPath");
		if (rcmPath != null)
			userRatingsConf.set(RECOMMEND_FILTER_PATH, rcmPath);
		
		boolean usesLongIDs = Boolean
				.parseBoolean(getOption("usesLongIDs"));
		if (usesLongIDs) {
			userRatingsConf.set(
					ParallelALSFactorizationJob.USES_LONG_IDS,
					String.valueOf(true));
		}
		
		//TODO: Revert
		
		boolean succeeded = false;
		
		if (false) {
		log.info("Starting userRatingsByUserBlock job");
		succeeded = userRatingsByUserBlock.waitForCompletion(true);
		if (!succeeded) {
			throw new IllegalStateException("userRatingsByUserBlock job failed");
		}
		} // false

		String userFeaturesPath = getOption("userFeatures");
		
		//for (int blockId = 0; blockId < numUserBlock; blockId++) {
		int blockId = 23;
		//TODO: Revert
 
			// process each user block
			//TODO: Revert
			Job[] predictJobArray = new Job[numItemBlock];
			for (int itemBlockId = 0; itemBlockId < numItemBlock; itemBlockId++) {
		
				Path blockUserRatingsPath = new Path(pathToUserRatingsByUserBlock()
						.toString() + "/" + Integer.toString(blockId) + "x" + Integer.toString(itemBlockId) + "-m-*");				
				//userRatingsByUserBlock/23x91-m-03308
				
				Path blockUserFeaturesPath = new Path(userFeaturesPath + "/"
						+ Integer.toString(blockId) + "-r-*");
				Path blockItemFeaturesPath = new Path(getOption("itemFeatures") + "/"
						+ Integer.toString(itemBlockId) + "-r-*");
				Path blockUserIDIndexPath = new Path(getOption("userIDIndex") + "/"
						+ Integer.toString(blockId) + "-r-*");
				Path blockItemIDIndexPath = new Path(getOption("itemIDIndex") + "/"
						+ Integer.toString(itemBlockId) + "-r-*");
				Path blockOutputPath = new Path(getTempPath().toString() + "/result/"
						+ Integer.toString(blockId) + "x" + Integer.toString(itemBlockId));

				Job blockPrediction = prepareJob(blockUserRatingsPath,
						blockOutputPath, SequenceFileInputFormat.class,
						MultithreadedSharingMapper.class, LongWritable.class,
						DoubleLongPairWritable.class, SequenceFileOutputFormat.class);
				
				Configuration blockPredictionConf = blockPrediction
						.getConfiguration();
				int numThreads = Integer.parseInt(getOption("numThreads"));
				blockPredictionConf.set(USER_FEATURES_PATH,
						blockUserFeaturesPath.toString());
				blockPredictionConf.set(ITEM_FEATURES_PATH,
						blockItemFeaturesPath.toString());

				blockPredictionConf.setInt(NUM_RECOMMENDATIONS,
						Integer.parseInt(getOption("numRecommendations")));
				blockPredictionConf.set(MAX_RATING, getOption("maxRating"));
				
				blockPredictionConf.setInt(NUM_USER_BLOCK, numUserBlock);
				blockPredictionConf.setInt(NUM_ITEM_BLOCK, numItemBlock);

				blockPredictionConf.set(QUEUE_NAME, defaultQueue);
				
				if (usesLongIDs) {
					blockPredictionConf.set(
							ParallelALSFactorizationJob.USES_LONG_IDS,
							String.valueOf(true));
					blockPredictionConf.set(USER_INDEX_PATH,
							blockUserIDIndexPath.toString());
					blockPredictionConf.set(ITEM_INDEX_PATH,
							blockItemIDIndexPath.toString());
				}
	
				if (rcmPath != null)
					blockPredictionConf.set(RECOMMEND_FILTER_PATH, rcmPath);
	
				MultithreadedMapper.setMapperClass(blockPrediction,
						BlockPredictionMapper.class);

				MultithreadedMapper.setNumberOfThreads(blockPrediction, numThreads);
	
				predictJobArray[itemBlockId] = blockPrediction;
				
				log.info("Submitting block prediction map job");
				blockPrediction.submit();
				
			}
			
			boolean allFinished = false;
			
			while (!allFinished) {								
				Thread.sleep(10000);
				
				for (int i=0; i < numItemBlock; i++) {
					if (predictJobArray[i] != null) {
						if (!predictJobArray[i].isComplete()) {							
							break;
						} else {
							if (!predictJobArray[i].isSuccessful()) {
								throw new IllegalStateException("BlockPrediction job blockId: " + blockId + " itemBlockId: " + i + " failed");
							}
						}
					}
					
					if (i == numItemBlock-1) allFinished = true; 
				}
			}
			
			Path blocksOutputPath = new Path(getTempPath().toString() + "/result/*/");
			Job blockRecommendation = prepareJob(blocksOutputPath,
					new Path(getOutputPath().toString() + "/recomd/"), SequenceFileInputFormat.class,
					Mapper.class, LongWritable.class,
					DoubleLongPairWritable.class, 
					RecommendReducer.class,
					LongWritable.class, RecommendedItemsWritable.class, 
					TextOutputFormat.class); 
			Configuration blockRecommendationConf = blockRecommendation.getConfiguration();
			
			blockRecommendationConf.set(QUEUE_NAME, defaultQueue);
			blockRecommendationConf.setInt(NUM_RECOMMENDATIONS,
					Integer.parseInt(getOption("numRecommendations")));
			blockRecommendationConf.setInt(MAX_RATING, Integer.parseInt(getOption("maxRating")));

			
			log.info("Starting blockRecommendation (reduce) job");
			succeeded = blockRecommendation.waitForCompletion(true);
			if (!succeeded) {
				throw new IllegalStateException("blockRecommendation job failed");
			}
		//}


		return 0;
	}

	static class UserRatingsByUserBlockMapper extends
			Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

		private MultipleOutputs<IntWritable, VectorWritable> out;
		private int numUserBlocks;
		private int numItemBlocks;
		private Path rcmFilterPath;
		private HashSet<Integer> rcmFilterSet = null;
		private boolean usesLongIDs;
		
		@Override
		protected void setup(Context ctx) throws IOException,
				InterruptedException {
			
			Configuration conf = ctx.getConfiguration();
			
			out = new MultipleOutputs<IntWritable, VectorWritable>(ctx);
			numUserBlocks = ctx.getConfiguration().getInt(NUM_USER_BLOCK, 10);
			numItemBlocks = ctx.getConfiguration().getInt(NUM_ITEM_BLOCK, 10);
			
			usesLongIDs = conf.getBoolean(
					ParallelALSFactorizationJob.USES_LONG_IDS, false);
			
			String p = conf.get(BlockRecommenderJob.RECOMMEND_FILTER_PATH);
			if (p != null) {
				rcmFilterPath = new Path(p);
				rcmFilterSet = loadFilterList(conf);
				Preconditions.checkState(rcmFilterSet.size() > 0, "Empty filter list. Check " + BlockRecommenderJob.RECOMMEND_FILTER_PATH);
			}
			
		}

		@Override
		protected void map(IntWritable userIdWritable,
				VectorWritable ratingsWritable, Context ctx) throws IOException,
				InterruptedException {
			
			int userId = userIdWritable.get();

		    if (rcmFilterSet != null && !rcmFilterSet.contains(userId)) {
		    	return; // Generate recommendation for selected long id only
		    }	        
		    
			int userBlockId = BlockPartitionUtil.getBlockID(userIdWritable.get(),
					numUserBlocks);
			Iterator<Vector.Element> it = ratingsWritable.get().nonZeroes().iterator();
			int itemBlockId = BlockPartitionUtil.getBlockID(it.next().index(),
					numItemBlocks);			
			
			String outputName = Integer.toString(userBlockId) + "x" + Integer.toString(itemBlockId);
			
			
			out.write(outputName, userIdWritable, ratingsWritable);
		}

		@Override
		protected void cleanup(Context context) throws IOException,
				InterruptedException {
			out.close();
		}
		
		// load recommendation filter list
		private HashSet<Integer> loadFilterList(Configuration conf) throws IOException {
			return loadFilterList(rcmFilterPath, conf);
		}

		// load recommendation filter list
		private HashSet<Integer> loadFilterList(Path location, Configuration conf)
				throws IOException {

			HashSet<Integer> s = new HashSet<Integer>();

			FileSystem fileSystem = FileSystem.get(location.toUri(), conf);
			CompressionCodecFactory factory = new CompressionCodecFactory(conf);
			FileStatus[] items = fileSystem.listStatus(location);

			if (items == null) {
				System.out.println("No filter found.");
				return s;
			}

			for (FileStatus item : items) {

				System.out.println("loadFilterList file name: " + item.getPath().getName());
				// ignoring files like _SUCCESS
				if (item.getPath().getName().startsWith("_")) {
					continue;
				}

				CompressionCodec codec = factory.getCodec(item.getPath());
				InputStream stream = null;

				// check if we have a compression codec we need to use
				if (codec != null) {
					stream = codec
							.createInputStream(fileSystem.open(item.getPath()));
				} else {
					stream = fileSystem.open(item.getPath());
				}

				StringWriter writer = new StringWriter();
				IOUtils.copy(stream, writer, "UTF-8");
				String raw = writer.toString();

				for (String str : raw.split("\n")) {
					int id; 
					if (usesLongIDs) {
						long longId = Long.parseLong(str.trim());
						id = TasteHadoopUtils.idToIndex(longId);
						System.out.println("Long ID: " + longId + " Short ID :" + id);
					} else {
						id = Integer.parseInt(str.trim());
					}
					s.add(new Integer(id));
				}
			}

			System.out.println("filter size: " + s.size());
			
			return s;
		}
	}

	private Path pathToUserRatingsByUserBlock() {
		return getOutputPath("userRatingsByUserBlock");
	}
}
