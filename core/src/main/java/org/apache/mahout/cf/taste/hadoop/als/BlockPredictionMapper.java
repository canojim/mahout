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
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.CompressionCodecFactory;
import org.apache.mahout.cf.taste.hadoop.MutableRecommendedItem;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.hadoop.TopItemsQueue;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.IntObjectProcedure;
import org.apache.mahout.math.map.OpenIntLongHashMap;
import org.apache.mahout.math.map.OpenIntObjectHashMap;
import org.apache.mahout.math.set.OpenIntHashSet;

import com.google.common.base.Preconditions;

/**
 * a multithreaded mapper that loads the feature matrices U and M into memory.
 * Afterwards it computes recommendations from these. Can be executed by a
 * {@link MultithreadedSharingMapper}.
 */
public class BlockPredictionMapper
		extends
		SharingMapper<IntWritable, VectorWritable, LongWritable, RecommendedItemsWritable, OpenIntObjectHashMap<Vector>> {

	private int recommendationsPerUser;
	private float maxRating;

	private boolean usesLongIDs;
	private OpenIntLongHashMap userIDIndex;
	// private OpenIntLongHashMap itemIDIndex;

	private final LongWritable userIDWritable = new LongWritable();
	private final RecommendedItemsWritable recommendations = new RecommendedItemsWritable();

	// private Path userIndexPath;
	//private Path itemIndexPath;
	String itemIndexPathStr;
	int numItemBlocks;
	//private Path pathToBlockU;
	private Path pathToM;
	private Path rcmFilterPath;

	private HashSet<Long> rcmFilterSet = null;

	@Override
	OpenIntObjectHashMap<Vector> createSharedInstance(Context ctx) {

		Configuration conf = ctx.getConfiguration();

		String p = conf.get(BlockRecommenderJob.USER_FEATURES_PATH);
		OpenIntObjectHashMap<Vector> u = ALS.readMatrixByRowsGlob(new Path(p), conf); 
		System.out.println("p: " + p + " U.size: " + u.size());
		
		return u;
	}

	@Override
	protected void setup(Context ctx) throws IOException, InterruptedException {
		Configuration conf = ctx.getConfiguration();
		recommendationsPerUser = conf.getInt(
				BlockRecommenderJob.NUM_RECOMMENDATIONS,
				BlockRecommenderJob.DEFAULT_NUM_RECOMMENDATIONS);
		maxRating = Float.parseFloat(conf.get(BlockRecommenderJob.MAX_RATING));

		usesLongIDs = conf.getBoolean(
				ParallelALSFactorizationJob.USES_LONG_IDS, false);

		//pathToBlockU = new Path(conf.get(BlockRecommenderJob.USER_FEATURES_PATH));
		pathToM = new Path(conf.get(BlockRecommenderJob.ITEM_FEATURES_PATH));

		String p = conf.get(BlockRecommenderJob.RECOMMEND_FILTER_PATH);
		if (p != null) {
			rcmFilterPath = new Path(p);
			rcmFilterSet = loadFilterList(conf);
			Preconditions.checkState(rcmFilterSet.size() > 0, "Empty filter list. Check " + BlockRecommenderJob.RECOMMEND_FILTER_PATH);
		}

		if (usesLongIDs) {
			userIDIndex = TasteHadoopUtils.readIDIndexMapGlob(
					conf.get(BlockRecommenderJob.USER_INDEX_PATH), conf);
			// itemIDIndex =
			// TasteHadoopUtils.readIDIndexMap(conf.get(RecommenderJob.ITEM_INDEX_PATH),
			// conf);

			itemIndexPathStr = conf.get(BlockRecommenderJob.ITEM_INDEX_PATH);
		}
		
		numItemBlocks = conf.getInt(BlockRecommenderJob.NUM_ITEM_BLOCK, 10);

	}

	@Override
	protected void map(IntWritable userIndexWritable,
			VectorWritable ratingsWritable, Context ctx) throws IOException,
			InterruptedException {

		Configuration conf = ctx.getConfiguration();
		OpenIntObjectHashMap<Vector> U = getSharedInstance();
		
		int userIndex = userIndexWritable.get();
		long userIndexLongID = -1;

		if (usesLongIDs) {

			userIndexLongID = userIDIndex.get(userIndex);
			
			Preconditions.checkState(userIndexLongID > 0, "userIndexLongID must be greater than 0. userIndex: " + userIndex);
			
			if (rcmFilterSet != null && !rcmFilterSet.contains(userIndexLongID)) {
				return; // Generate recommendation for selected few
						// id only.
			}

		} else {
			if (rcmFilterSet != null
					&& !rcmFilterSet.contains(new Long(userIndex)))
				return;
		}
		
		System.out.println("after filter userIndex: " + userIndex + " longID: " + userIndexLongID);
		
		Vector ratings = ratingsWritable.get();

		final OpenIntHashSet alreadyRatedItems = new OpenIntHashSet(
				ratings.getNumNondefaultElements());

		for (Vector.Element e : ratings.nonZeroes()) {
			alreadyRatedItems.add(e.index());
		}

		final TopItemsQueue topItemsQueue = new TopItemsQueue(
				recommendationsPerUser);
		final Vector userFeature = U.get(userIndex);

		forEachItemPair(conf, new IntObjectProcedure<Vector>() {
			@Override
			public boolean apply(int itemID, Vector itemFeatures) {
				if (!alreadyRatedItems.contains(itemID)) {
					double predictedRating = userFeature.dot(itemFeatures);

					MutableRecommendedItem top = topItemsQueue.top();
					if (predictedRating > top.getValue()) {
						top.set(itemID, (float) predictedRating);
						topItemsQueue.updateTop();
					}
				}
				return true;
			}
		});
		
		
		int [] blockToLoad = new int[numItemBlocks];
				
		List<RecommendedItem> recommendedItems = topItemsQueue.getTopItems();

		if (!recommendedItems.isEmpty()) {

			HashMap<Long, RecommendedItem> recommendedMap = new HashMap<Long, RecommendedItem>();

			// cap predictions to maxRating
			for (RecommendedItem topItem : recommendedItems) {
				((MutableRecommendedItem) topItem).capToMaxValue(maxRating);

				recommendedMap.put(topItem.getItemID(), topItem);
				int blockId = BlockPartitionUtil.getBlockID((int) topItem.getItemID(), numItemBlocks);
				blockToLoad[blockId] += 1;
			}

			if (usesLongIDs) {

				userIDWritable.set(userIndexLongID);
				
				for (int i=0; i < blockToLoad.length; i++) {
					if (blockToLoad[i] > 0) {
						long count = 0;
						Path itemIndexPath = new Path(itemIndexPathStr + "/" + i + "-r-*");
						for (Pair<VarIntWritable, VarLongWritable> record : new SequenceFileDirIterable<VarIntWritable, VarLongWritable>(
								itemIndexPath, PathType.GLOB, null, null, false, conf)) {
							RecommendedItem item = recommendedMap.get(new Long(record
									.getFirst().get()));

							if (item != null) {
								count++;
								((MutableRecommendedItem) item).setItemID(record
										.getSecond().get());
							}

							if (blockToLoad[i] == count)
								break;
						} //for
					}
				}
				
				/*
				 * long userID = userIDIndex.get(userIndex);
				 * userIDWritable.set(userID);
				 * 
				 * for (RecommendedItem topItem : recommendedItems) { // remap
				 * item IDs long itemID = itemIDIndex.get((int)
				 * topItem.getItemID()); ((MutableRecommendedItem)
				 * topItem).setItemID(itemID); }
				 */
			} else {
				userIDWritable.set(userIndex);
			}

			recommendations.set(recommendedItems);
			ctx.write(userIDWritable, recommendations);
		} else {
			System.out.println("recommendedItems is empty");
		}
	}

	// Refactor M.forEachItemPair
	// M
	private boolean forEachItemPair(Configuration conf,
			IntObjectProcedure<Vector> procedure) {
		for (Pair<IntWritable, VectorWritable> pair : new SequenceFileDirIterable<IntWritable, VectorWritable>(
				pathToM, PathType.GLOB, null, conf)) {
			int rowIndex = pair.getFirst().get();
			Vector row = pair.getSecond().get();

			if (!procedure.apply(rowIndex, row)) {
				return false;
			}
		} // for

		return true;
	}

	// load recommendation filter list
	private HashSet<Long> loadFilterList(Configuration conf) throws IOException {
		return loadFilterList(rcmFilterPath, conf);
	}

	// load recommendation filter list
	private HashSet<Long> loadFilterList(Path location, Configuration conf)
			throws IOException {

		HashSet<Long> s = new HashSet<Long>();

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
				s.add(new Long(str.trim()));
			}
		}

		System.out.println("filter size: " + s.size());
		
		return s;
	}

}
