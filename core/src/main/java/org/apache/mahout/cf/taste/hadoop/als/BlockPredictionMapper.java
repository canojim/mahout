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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.StringWriter;
import java.util.ArrayList;
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
import org.apache.mahout.cf.taste.hadoop.TopItemsQueue;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.IntObjectProcedure;
import org.apache.mahout.math.map.OpenIntObjectHashMap;
import org.apache.mahout.math.set.OpenIntHashSet;

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
	// private OpenIntLongHashMap userIDIndex;
	// private OpenIntLongHashMap itemIDIndex;

	private final LongWritable userIDWritable = new LongWritable();
	private final RecommendedItemsWritable recommendations = new RecommendedItemsWritable();

	private Path userIndexPath;
	private Path itemIndexPath;
	private Path pathToBlockU;
	private Path pathToM;
	private Path rcmFilterPath;

	private HashSet<Long> rcmFilterSet;

	@Override
	OpenIntObjectHashMap<Vector> createSharedInstance(Context ctx) {

		Configuration conf = ctx.getConfiguration();
		 
		return ALS.readMatrixByRows(pathToBlockU, conf);
	}

	@Override
	protected void setup(Context ctx) throws IOException, InterruptedException {
		Configuration conf = ctx.getConfiguration();
		recommendationsPerUser = conf.getInt(
				RecommenderJob.NUM_RECOMMENDATIONS,
				RecommenderJob.DEFAULT_NUM_RECOMMENDATIONS);
		maxRating = Float.parseFloat(conf.get(RecommenderJob.MAX_RATING));

		usesLongIDs = conf.getBoolean(
				ParallelALSFactorizationJob.USES_LONG_IDS, false);

		pathToBlockU = new Path(conf.get(RecommenderJob.USER_FEATURES_PATH));
		pathToM = new Path(conf.get(RecommenderJob.ITEM_FEATURES_PATH));

		String p = conf.get(RecommenderJob.RECOMMEND_FILTER_PATH);
		if (p != null) {
			rcmFilterPath = new Path(p);
			rcmFilterSet = loadFilterList(conf);
		}

		if (usesLongIDs) {
			// userIDIndex =
			// TasteHadoopUtils.readIDIndexMap(conf.get(RecommenderJob.USER_INDEX_PATH),
			// conf);
			// itemIDIndex =
			// TasteHadoopUtils.readIDIndexMap(conf.get(RecommenderJob.ITEM_INDEX_PATH),
			// conf);

			userIndexPath = new Path(conf.get(RecommenderJob.USER_INDEX_PATH));
			itemIndexPath = new Path(conf.get(RecommenderJob.ITEM_INDEX_PATH));

		}
	}

	@Override
	protected void map(IntWritable userIndexWritable,
			VectorWritable ratingsWritable, Context ctx) throws IOException,
			InterruptedException {

		Configuration conf = ctx.getConfiguration();
		// Pair<OpenIntObjectHashMap<Vector>, OpenIntObjectHashMap<Vector>>
		// uAndM = getSharedInstance();
		// OpenIntObjectHashMap<Vector> U = uAndM.getFirst();
		// OpenIntObjectHashMap<Vector> M = uAndM.getSecond();

		int userIndex = userIndexWritable.get();
		long userIndexLongID = -1;

		if (usesLongIDs) {

			int count = 0;
			try {
				for (Pair<VarIntWritable, VarLongWritable> record : new SequenceFileDirIterable<VarIntWritable, VarLongWritable>(
						userIndexPath, PathType.LIST, PathFilters.partFilter(),
						null, false, conf)) {

					if (userIndex == record.getFirst().get()) {
						userIndexLongID = record.getSecond().get();

						if (rcmFilterSet != null
								&& !rcmFilterSet.contains(userIndexLongID)) {
							return; // Generate recommendation for selected few
									// id only.
						}
					}

					count++;
				} // for
			} catch (RuntimeException e) {
				System.out.println("usesLongIDs userIndex: " + userIndex);
				System.out.println("count: " + count);

				e.printStackTrace();
				throw e;
			}
		} else {
			if (rcmFilterSet != null
					&& !rcmFilterSet.contains(new Long(userIndex)))
				return;
		}

		Vector ratings = ratingsWritable.get();

		final OpenIntHashSet alreadyRatedItems = new OpenIntHashSet(
				ratings.getNumNondefaultElements());

		for (Vector.Element e : ratings.nonZeroes()) {
			alreadyRatedItems.add(e.index());
		}

		final TopItemsQueue topItemsQueue = new TopItemsQueue(
				recommendationsPerUser);
		final Vector userFeatures = getUserFeatures(conf, userIndex);

		forEachItemPair(conf, new IntObjectProcedure<Vector>() {
			@Override
			public boolean apply(int itemID, Vector itemFeatures) {
				if (!alreadyRatedItems.contains(itemID)) {
					double predictedRating = userFeatures.dot(itemFeatures);

					MutableRecommendedItem top = topItemsQueue.top();
					if (predictedRating > top.getValue()) {
						top.set(itemID, (float) predictedRating);
						topItemsQueue.updateTop();
					}
				}
				return true;
			}
		});

		List<RecommendedItem> recommendedItems = topItemsQueue.getTopItems();

		if (!recommendedItems.isEmpty()) {

			HashMap<Long, RecommendedItem> recommendedMap = new HashMap<Long, RecommendedItem>();

			// cap predictions to maxRating
			for (RecommendedItem topItem : recommendedItems) {
				((MutableRecommendedItem) topItem).capToMaxValue(maxRating);

				recommendedMap.put(topItem.getItemID(), topItem);
			}

			if (usesLongIDs) {

				userIDWritable.set(userIndexLongID);

				// Refactor long userID = userIDIndex.get(userIndex);
				long count = 0;
				for (Pair<VarIntWritable, VarLongWritable> record : new SequenceFileDirIterable<VarIntWritable, VarLongWritable>(
						itemIndexPath, PathType.LIST, PathFilters.partFilter(),
						null, false, conf)) {

					RecommendedItem item = recommendedMap.get(new Long(record
							.getFirst().get()));

					if (item != null) {
						count++;
						((MutableRecommendedItem) item).setItemID(record
								.getSecond().get());
					}

					if (recommendedMap.size() == count)
						break;
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
		}
	}

	// Refactor final Vector userFeatures = U.get(userIndex);
	// U
	private Vector getUserFeatures(Configuration conf, int userIndex) {
		for (Pair<IntWritable, VectorWritable> pair : new SequenceFileDirIterable<IntWritable, VectorWritable>(
				pathToU, PathType.LIST, PathFilters.partFilter(), conf)) {
			int rowIndex = pair.getFirst().get();
			if (rowIndex == userIndex) {
				Vector row = pair.getSecond().get();
				return row;
			}
		}
		return null;
	}

	// Refactor M.forEachItemPair
	// M
	private boolean forEachItemPair(Configuration conf,
			IntObjectProcedure<Vector> procedure) {
		for (Pair<IntWritable, VectorWritable> pair : new SequenceFileDirIterable<IntWritable, VectorWritable>(
				pathToM, PathType.LIST, PathFilters.partFilter(), conf)) {
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
		
/*		HashSet<Long> s = new HashSet<Long>();
		FileSystem fs = FileSystem.get(conf);

		BufferedReader br = new BufferedReader(new InputStreamReader(
				fs.open(rcmFilterPath)));

		String line;
		line = br.readLine();

		while (line != null && !"".equals(line)) {
			s.add(new Long(line));
			line = br.readLine();
		}

		return s;
*/
	}

	// load recommendation filter list
	private HashSet<Long> loadFilterList(Path location, Configuration conf)
			throws IOException {
		
		HashSet<Long> s = new HashSet<Long>();
		
		FileSystem fileSystem = FileSystem.get(location.toUri(), conf);
		CompressionCodecFactory factory = new CompressionCodecFactory(conf);
		FileStatus[] items = fileSystem.listStatus(location);
		
		if (items == null) {
			System.out.println("items is null.");
			return s;
		}
		
		for (FileStatus item : items) {

			System.out.println("file name: " + item.getPath().getName());
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
				s.add(new Long(str));
			}
		}
		
		System.out.println("s.size: " + s.size());
		return s;
	}

}
