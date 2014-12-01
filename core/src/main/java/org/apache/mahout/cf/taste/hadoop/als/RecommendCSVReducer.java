package org.apache.mahout.cf.taste.hadoop.als;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.cf.taste.hadoop.MutableRecommendedItem;
import org.apache.mahout.cf.taste.hadoop.TopItemsQueue;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;

public class RecommendCSVReducer extends Reducer<LongWritable, LongDoublePairWritable, 
		LongWritable, Text> {

	private int maxRating;
	private int recommendationsPerUser;
	private String delimeter = ",";
	
	//private final RecommendedItemsWritable recommendations = new RecommendedItemsWritable();
	private final Text itemAndScore = new Text();
	  
	@Override
	protected void setup(Context context) throws IOException,
			InterruptedException {
		
		Configuration conf = context.getConfiguration();
		
		String d = conf.get("mapred.textoutputformat.separator");
		if (d != null) {
			delimeter = d;
		}
		
		recommendationsPerUser = conf.getInt(BlockRecommenderJob.NUM_RECOMMENDATIONS, 10);
		maxRating = conf.getInt(BlockRecommenderJob.MAX_RATING, 100);		
	}

	
	@Override
	protected void reduce(LongWritable userIDWritable,
			Iterable<LongDoublePairWritable> values,
			Context ctx) throws IOException, InterruptedException {

		TopItemsQueue topItemsQueue = new TopItemsQueue(recommendationsPerUser);
		
		for (LongDoublePairWritable i: values) {
			double score = i.getSecond().get();
			MutableRecommendedItem top = topItemsQueue.top();
			if (score > top.getValue()) {
				top.set(i.getFirst().get(), (float) score);
	            topItemsQueue.updateTop();
	        }
		}

		List<RecommendedItem> recommendedItems = topItemsQueue.getTopItems();
	    if (recommendedItems.size() == 0) {
	    	System.out.println("WARN: recommendedItems.size() equals to zero.");
	    }
	    System.out.println("recommendedItems.size: " + recommendedItems.size());
		
	    if (!recommendedItems.isEmpty()) {

	      // cap predictions to maxRating
	      for (RecommendedItem topItem : recommendedItems) {
	        ((MutableRecommendedItem) topItem).capToMaxValue(maxRating);
	        itemAndScore.set(topItem.getItemID() + delimeter + topItem.getValue());
	        ctx.write(userIDWritable, itemAndScore);
	      }
	    } 			
	}

	
}