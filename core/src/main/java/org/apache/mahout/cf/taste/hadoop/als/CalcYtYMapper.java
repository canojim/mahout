package org.apache.mahout.cf.taste.hadoop.als;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix.MatrixEntryWritable;

import com.google.common.base.Preconditions;

/**
 * Refactor the following code to Mapper
 * 
 * IntArrayList indexes = Y.keys(); indexes.quickSort(); int numIndexes =
 * indexes.size();
 * 
 * double[][] YtY = new double[numFeatures][numFeatures];
 * 
 * // Compute Y'Y by dot products between the 'columns' of Y for (int i = 0; i <
 * numFeatures; i++) { for (int j = i; j < numFeatures; j++) { double dot = 0;
 * for (int k = 0; k < numIndexes; k++) { Vector row =
 * Y.get(indexes.getQuick(k)); dot += row.getQuick(i) * row.getQuick(j); }
 * YtY[i][j] = dot; if (i != j) { YtY[j][i] = dot; } } } return new
 * DenseMatrix(YtY, true);
 * 
 * 
 * 
 */
public class CalcYtYMapper
		extends
		Mapper<IntWritable, VectorWritable, MatrixEntryWritable, DoubleWritable> {

	static final String NUM_FEATURES = CalcYtYMapper.class
			.getName() + ".numFeatures";
	
	int numFeatures;

	@Override
	protected void setup(Context context) throws IOException,
			InterruptedException {
		Configuration conf = context.getConfiguration();
		numFeatures = conf.getInt(NUM_FEATURES, -1);

		Preconditions.checkArgument(numFeatures > 0,
				"numFeatures must be greater then 0!");
	}

	@Override
	protected void map(IntWritable key, VectorWritable value, Context context)
			throws IOException, InterruptedException {

		// Compute Y'Y by dot products between the 'columns' of Y
		for (int i = 0; i < numFeatures; i++) {
			for (int j = i; j < numFeatures; j++) {
				double dot = 0;
				Vector row = value.get();
				dot = row.getQuick(i) * row.getQuick(j);

				MatrixEntryWritable yty = new MatrixEntryWritable();
				yty.setRow(i);
				yty.setCol(j);

				DoubleWritable dotWritable = new DoubleWritable(dot);

				context.write(yty, dotWritable);

				if (i != j) {
					yty = new MatrixEntryWritable();
					yty.setRow(j);
					yty.setCol(i);
					context.write(yty, dotWritable);
				}

			}
		}

	}

}
