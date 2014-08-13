package org.apache.mahout.cf.taste.hadoop.als;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.als.AlternatingLeastSquaresSolver;
import org.apache.mahout.math.hadoop.DistributedRowMatrix.MatrixEntryWritable;

import com.google.common.base.Preconditions;

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
		numFeatures = conf.getInt(BlockParallelALSFactorizationJob.NUM_FEATURES, -1);
		numBlocks = conf.getInt(BlockParallelALSFactorizationJob.NUM_BLOCKS, 10);

		Preconditions.checkArgument(numFeatures > 0,
				"numFeatures must be greater then 0!");

		Path pathToYty = new Path(conf.get(BlockParallelALSFactorizationJob.PATH_TO_YTY));
		this.YtransposeY = ALS.readYtransposeYFromHdfs(pathToYty, numFeatures, conf);

		out = new MultipleOutputs(context);
	}

	
	@Override
	protected void reduce(IntWritable key,
			Iterable<ALSContributionWritable> values,
			Context ctx) throws IOException, InterruptedException {
		
			Matrix combinedA = new DenseMatrix(numFeatures, numFeatures);
			Matrix combinedb = new DenseMatrix(numFeatures, 1);
			
			for (ALSContributionWritable contribution: values) {
				combinedA = combinedA.plus(contribution.getA().get());
				combinedb = combinedb.plus(contribution.getb().get());
			}	
			
			Vector result = AlternatingLeastSquaresSolver.solve(
										this.YtransposeY.plus(combinedA), combinedb);
			int blockId = BlockPartitionUtil.getBlockID(key.get(), numBlocks);

			out.write(Integer.toString(blockId), key, new VectorWritable(result));
	}

	
}