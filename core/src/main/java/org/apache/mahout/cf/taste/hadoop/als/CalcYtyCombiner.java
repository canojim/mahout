package org.apache.mahout.cf.taste.hadoop.als;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.hadoop.DistributedRowMatrix.MatrixEntryWritable;

public class CalcYtyCombiner extends Reducer<MatrixEntryWritable, DoubleWritable, MatrixEntryWritable, DoubleWritable> {
	
	@Override
	protected void reduce(MatrixEntryWritable yty,
			Iterable<DoubleWritable> values,
			Context ctx) throws IOException, InterruptedException {
		
			double sum = 0;
			
			for (DoubleWritable val: values) {
				sum += val.get();
			}	
			
			ctx.write(yty, new DoubleWritable(sum));
	}

	
}