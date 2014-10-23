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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.mahout.cf.taste.hadoop.MutableRecommendedItem;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.hadoop.TopItemsQueue;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.IntObjectProcedure;
import org.apache.mahout.math.map.OpenIntLongHashMap;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

/**
 * a multithreaded mapper that loads the feature matrices U and M into memory.
 * Afterwards it computes recommendations from these. Can be executed by a
 * {@link MultithreadedSharingMapper}.
 */
public class BlockFencePredictionMapper
		extends
		SharingMapper<IntWritable, VectorWritable, LongWritable, DoubleLongPairWritable, OpenIntObjectHashMap<Vector>> {

	private int recommendationsPerUser;

	private boolean usesLongIDs;
	private OpenIntLongHashMap itemIDIndex;

	private final LongWritable userIDWritable = new LongWritable();
	private final DoubleLongPairWritable scoreAndItemWritable = new DoubleLongPairWritable();
	private final LongWritable itemidWritable = new LongWritable();
	
	private Path pathToBlockM;


	@Override
	OpenIntObjectHashMap<Vector> createSharedInstance(Context ctx) {

		Configuration conf = ctx.getConfiguration();

		pathToBlockM = new Path(conf.get(BlockFenceRecommenderJob.ITEM_FEATURES_PATH));
		
		OpenIntObjectHashMap<Vector> M = ALS.readMatrixByRowsGlob(pathToBlockM, conf);
		
		System.out.println("pathToBlockM: " + pathToBlockM.toString());
		
		System.out.println("M.size: " + M.size());
		return M;
		
	}

	@Override
	protected void setup(Context ctx) throws IOException, InterruptedException {
		Configuration conf = ctx.getConfiguration();
		recommendationsPerUser = conf.getInt(
				BlockFenceRecommenderJob.NUM_RECOMMENDATIONS,
				BlockFenceRecommenderJob.DEFAULT_NUM_RECOMMENDATIONS);

		usesLongIDs = conf.getBoolean(
				ParallelALSFactorizationJob.USES_LONG_IDS, false);

		if (usesLongIDs) {
			String itemIndexPath = conf.get(BlockFenceRecommenderJob.ITEM_INDEX_PATH);
			
			System.out.println("itemIndexPath: " + itemIndexPath);
			
			itemIDIndex = TasteHadoopUtils.readIDIndexMapGlob(
					conf.get(BlockFenceRecommenderJob.ITEM_INDEX_PATH), conf);
			System.out.println("itemIDIndex.size(): " + itemIDIndex.size());
		}
		
	}

	@Override
	protected void map(IntWritable userIndexWritable,
			VectorWritable userFeaturesWritable, Context ctx) throws IOException,
			InterruptedException {
		
	    int userIndex = userIndexWritable.get();
	    	    
		OpenIntObjectHashMap<Vector> M = getSharedInstance();
		
	    final TopItemsQueue topItemsQueue = new TopItemsQueue(recommendationsPerUser);
	    final Vector userFeatures = userFeaturesWritable.get();	    

	    M.forEachPair(new IntObjectProcedure<Vector>() {
	      @Override
	      public boolean apply(int itemID, Vector itemFeatures) {
	          double predictedRating = userFeatures.dot(itemFeatures);
	
	          MutableRecommendedItem top = topItemsQueue.top();
	          if (predictedRating > top.getValue()) {
	            top.set(itemID, (float) predictedRating);
	            topItemsQueue.updateTop();
	          }

	          return true;
	      }
	    });

	    userIDWritable.set(userIndex);
	    
	    List<RecommendedItem> recommendedItems = topItemsQueue.getTopItems();
	    
	    if (recommendedItems.size() == 0) {
	    	System.out.println("WARN: recommendedItems.size() equals to zero.");
	    }
	    System.out.println("recommendedItems.size: " + recommendedItems.size());
	    
	    for (RecommendedItem topItem : recommendedItems) {
	    	//scoreWritable.set(topItem.getValue());
	    	scoreAndItemWritable.setFirst(topItem.getValue());
	    	
	    	if (usesLongIDs) {
	    		long itemID = itemIDIndex.get((int) topItem.getItemID());
	    		itemidWritable.set(itemID);
	    	} else {
	    		itemidWritable.set(topItem.getItemID());
	    	}
	    	scoreAndItemWritable.setSecond(itemidWritable.get());
	    		    	
	    	ctx.write(userIDWritable, scoreAndItemWritable);
	    }
	    
	}

}
