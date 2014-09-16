package org.apache.mahout.cf.taste.hadoop.als;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.mahout.cf.taste.hadoop.MutableRecommendedItem;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.hadoop.TopItemsQueue;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.als.AlternatingLeastSquaresSolver;

import com.google.common.base.Preconditions;

public class RecommendReducer extends Reducer<LongWritable, DoubleLongPairWritable, 
		LongWritable, RecommendedItemsWritable> {

	private TopItemsQueue topItemsQueue;
	private int maxRating;
	
	private final RecommendedItemsWritable recommendations = new RecommendedItemsWritable();
	  
	@Override
	protected void setup(Context context) throws IOException,
			InterruptedException {
		
		Configuration conf = context.getConfiguration();
		
		int recommendationsPerUser = conf.getInt(BlockRecommenderJob.NUM_RECOMMENDATIONS, 10);
		maxRating = conf.getInt(BlockRecommenderJob.MAX_RATING, 100);
		
		topItemsQueue = new TopItemsQueue(recommendationsPerUser);
	}

	
	@Override
	protected void reduce(LongWritable userIDWritable,
			Iterable<DoubleLongPairWritable> values,
			Context ctx) throws IOException, InterruptedException {

		for (DoubleLongPairWritable i: values) {
			double score = i.getFirst();
			MutableRecommendedItem top = topItemsQueue.top();
			if (score > top.getValue()) {
				top.set(i.getSecond(), (float) score);
	            topItemsQueue.updateTop();
	        }
		}

		List<RecommendedItem> recommendedItems = topItemsQueue.getTopItems();

	    if (!recommendedItems.isEmpty()) {

	      // cap predictions to maxRating
	      for (RecommendedItem topItem : recommendedItems) {
	        ((MutableRecommendedItem) topItem).capToMaxValue(maxRating);
	      }

	      recommendations.set(recommendedItems);
	      ctx.write(userIDWritable, recommendations);
	    }
			
	}

	
}