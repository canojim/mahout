package org.apache.mahout.cf.taste.hadoop.als;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.hadoop.DistributedRowMatrix.MatrixEntryWritable;

public class UpdateUorMReducer extends Reducer<IntWritable, ALSContributionWritable, 
		IntWritable, VectorWritable> {

	private MultipleOutputs out;
	private int numFeatures;
	private int numBlocks;
	private Matrix YtransposeY;

	@Override
	protected void setup(Context context) throws IOException,
			InterruptedException {
		Configuration conf = context.getConfiguration();
		numFeatures = conf.getInt(ParallelALSFactorizationJob.NUM_FEATURES, -1);
		numBlocks = ctx.getConfiguration().getInt(NUM_BLOCKS, 10);

		Preconditions.checkArgument(numFeatures > 0,
				"numFeatures must be greater then 0!");

		Path pathToYty = ????;
		this.YtransposeY = readYtransposeYFromHdfs(pathToYty);

		out = new MultipleOutputs(ctx);
	}

	
	@Override
	protected void reduce(IntWritable key,
			Iterable<ALSContributionWritable> values,
			Context ctx) throws IOException, InterruptedException {
		
			Matrix combinedA = new DenseMatrix(numFeatures, numFeatures);
			Matrix combinedb = new DenseMatrix(numFeatures, 1);
			
			for (ALSContributionWritable contribution: values) {
				combinedA = combinedA.plus(contribution.getA());
				combinedb = combinedb.plus(contribution.getb());
			}	
			
			Vector result = AlternatingLeastSquaresSolver.solve(
										this.YtransposeY.plus(combinedA), combinedb);
			int blockId = BlockPartitionUtil.getBlockID(key numBlocks);
			ctx.write(key, new ALSContributionWritable(combinedA, combinedb));

			out.write(Integer.toString(blockId), key, new VectorWritable(result));
	}

	
}