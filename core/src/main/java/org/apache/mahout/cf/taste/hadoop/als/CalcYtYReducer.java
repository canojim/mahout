package org.apache.mahout.cf.taste.hadoop.als;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.hadoop.DistributedRowMatrix.MatrixEntryWritable;

/**
 * Refactor the following logic to reducer
 * 
 * public class ImplicitFeedbackAlternatingLeastSquaresSolver
 * private Matrix getYtransposeY(OpenIntObjectHashMap<Vector> Y)
 * dot += row.getQuick(i) * row.getQuick(j);
 *
 */
public class CalcYtYReducer extends Reducer<MatrixEntryWritable, DoubleWritable, NullWritable, MatrixEntryWritable> {
	
	@Override
	protected void reduce(MatrixEntryWritable yty,
			Iterable<DoubleWritable> values,
			Context ctx) throws IOException, InterruptedException {
		
			double sum = 0;
			
			for (DoubleWritable val: values) {
				sum += val.get();
			}	
			
			yty.setVal(sum);
			
			ctx.write(NullWritable.get(), yty);
	}

	
}
