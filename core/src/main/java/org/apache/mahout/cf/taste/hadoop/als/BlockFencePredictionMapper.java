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
import org.apache.hadoop.io.DoubleWritable;
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
		SharingMapper<IntWritable, VectorWritable, LongWritable, LongDoublePairWritable, OpenIntObjectHashMap<Vector>> {

	private int recommendationsPerUser;

	private boolean usesLongIDs;
	private OpenIntLongHashMap userIDIndex;
	private OpenIntLongHashMap itemIDIndex;

	private final LongWritable userIDWritable = new LongWritable();
	private final LongDoublePairWritable itemAndScoreWritable = new LongDoublePairWritable();
	private final LongWritable itemidWritable = new LongWritable();
	private final DoubleWritable scoreWritable = new DoubleWritable();
	
	private Path pathToBlockM;


	@Override
	OpenIntObjectHashMap<Vector> createSharedInstance(Context ctx) {

		Configuration conf = ctx.getConfiguration();

		pathToBlockM = new Path(conf.get(BlockRecommenderJob.ITEM_FEATURES_PATH));
		
		OpenIntObjectHashMap<Vector> M = ALS.readMatrixByRowsGlob(pathToBlockM, conf);
		
		System.out.println("pathToBlockM: " + pathToBlockM.toString());
		
		System.out.println("M.size: " + M.size());
		return M;
		
	}

	@Override
	protected void setup(Context ctx) throws IOException, InterruptedException {
		Configuration conf = ctx.getConfiguration();
		recommendationsPerUser = conf.getInt(
				BlockRecommenderJob.NUM_RECOMMENDATIONS,
				BlockRecommenderJob.DEFAULT_NUM_RECOMMENDATIONS);

		usesLongIDs = conf.getBoolean(
				ParallelALSFactorizationJob.USES_LONG_IDS, false);

		if (usesLongIDs) {
			
			String userIndexPath = conf.get(BlockRecommenderJob.USER_INDEX_PATH);
			String itemIndexPath = conf.get(BlockRecommenderJob.ITEM_INDEX_PATH);
			
			System.out.println("userIndexPath: " + userIndexPath);
			System.out.println("itemIndexPath: " + itemIndexPath);
			
			userIDIndex = TasteHadoopUtils.readIDIndexMapGlob(
					conf.get(BlockRecommenderJob.USER_INDEX_PATH), conf);
			itemIDIndex = TasteHadoopUtils.readIDIndexMapGlob(
					conf.get(BlockRecommenderJob.ITEM_INDEX_PATH), conf);
			System.out.println("userIDIndex.size: " + userIDIndex.size() + " itemIDIndex.size(): " + itemIDIndex.size());
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

	    //System.out.println("M.size(): " + M.size());
	    M.forEachPair(new IntObjectProcedure<Vector>() {
	      @Override
	      public boolean apply(int itemID, Vector itemFeatures) {
	          double predictedRating = userFeatures.dot(itemFeatures);
	
	          MutableRecommendedItem top = topItemsQueue.top();
	          
	          float topValue = top.getValue();
	          if (predictedRating > topValue) {
	            top.set(itemID, (float) predictedRating);
	            topItemsQueue.updateTop();
	          } else {
	        	//System.out.println("itemID: " + itemID + " predictedRating: " + predictedRating + " < topValue: " + topValue);
	        	//Value lower than Float.NEGATIVE_INFINITY will not be considered.
	          }

	          return true;
	      }
	    });
	    
	    if (usesLongIDs && userIndex > 0) {
	        long userID = userIDIndex.get(userIndex);
	        userIDWritable.set(userID);
	    } else {
	    	userIDWritable.set(userIndex);
	    }
	    
	    List<RecommendedItem> recommendedItems = topItemsQueue.getTopItems();
	    
	    if (recommendedItems.size() == 0) {
	    	System.out.println("WARN: recommendedItems.size() equals to zero.");
	    }
	    System.out.println("recommendedItems.size: " + recommendedItems.size());
	    
	    for (RecommendedItem topItem : recommendedItems) {
	    	//scoreWritable.set(topItem.getValue());
	    	itemAndScoreWritable.setFirst(itemidWritable);
	    	
	    	int itemIndex = (int) topItem.getItemID();
	    	if (usesLongIDs && itemIndex > 0) {
	    		long itemID = itemIDIndex.get(itemIndex);
	    		itemidWritable.set(itemID);
	    	} else {
	    		itemidWritable.set(topItem.getItemID());
	    	}
	    	scoreWritable.set(topItem.getValue());
	    	itemAndScoreWritable.setSecond(scoreWritable);
	    		    	
	    	ctx.write(userIDWritable, itemAndScoreWritable);
	    }
	    
	}

}
