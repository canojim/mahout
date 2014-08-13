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
import java.util.Date;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.QRDecomposition;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.hadoop.DistributedRowMatrix.MatrixEntryWritable;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

import com.google.common.base.Preconditions;

/**
 * see <a href="http://research.yahoo.com/pub/2433">Collaborative Filtering for
 * Implicit Feedback Datasets</a>
 */
public class DistributedImplicitFeedbackAlternatingLeastSquaresSolver {

	private final int numFeatures;
	private final double alpha;
	private final double lambda;

	private OpenIntObjectHashMap<Vector> Y;
	private Matrix YtransposeY;

	//private int numEntities;
	private Path pathToYty;
	private Configuration conf;
	
	private UserRatingsHashMapCache partialYcache;

	
	/*
	 * public DistributedImplicitFeedbackAlternatingLeastSquaresSolver(int
	 * numFeatures, double lambda, double alpha, OpenIntObjectHashMap<Vector> Y)
	 * { this.numFeatures = numFeatures; this.lambda = lambda; this.alpha =
	 * alpha; this.Y = Y; YtransposeY = getYtransposeY(Y); }
	 */

	public DistributedImplicitFeedbackAlternatingLeastSquaresSolver(
			int numFeatures, int numEntities, double lambda, double alpha, 
			String pathYty, Configuration conf) throws IOException {
		this.numFeatures = numFeatures;
		this.lambda = lambda;
		this.alpha = alpha;

		//this.numEntities = numEntities;
		this.pathToYty = new Path(pathYty);
		this.conf = conf;

		long start = System.currentTimeMillis();			
		System.out.println("readYtransposeYFromHdfs Start time: " + new Date(start));
		
		this.YtransposeY = ALS.readYtransposeYFromHdfs(pathToYty, this.numFeatures, this.conf);
		
		long duration = System.currentTimeMillis() - start;
		System.out.println("readYtransposeYFromHdfs duration: " + duration);
		
		start = System.currentTimeMillis();
		partialYcache = new UserRatingsHashMapCache(conf, numEntities);
		duration = System.currentTimeMillis() - start;
		System.out.println("new UserRatingsHashMapCache duration: " + duration);
			
	}

	public Vector solve(Vector ratings) throws IOException {
		
		long start = System.currentTimeMillis();			
		System.out.println("readSmallYbasedonUserRating Start time: " + new Date(start));
		
		this.Y = readSmallYbasedonUserRating(ratings);
		
		long duration = System.currentTimeMillis() - start;		
		System.out.println("readSmallYbasedonUserRating duration: " + duration);
		
		return solve(
				YtransposeY.plus(getYtransponseCuMinusIYPlusLambdaI(ratings)),
				getYtransponseCuPu(ratings));
	}

	private static Vector solve(Matrix A, Matrix y) {
		return new QRDecomposition(A).solve(y).viewColumn(0);
	}

	double confidence(double rating) {
		return 1 + alpha * rating;
	}

	private OpenIntObjectHashMap<Vector> readSmallYbasedonUserRating(
			Vector userRatings) throws IOException {

		OpenIntObjectHashMap<Vector> featureMatrix = partialYcache.getHashMap(userRatings);
		
		System.out.println("miniY size: " + featureMatrix.size());
		
		Preconditions.checkState(!featureMatrix.isEmpty(),
				"Feature matrix is empty");
		return featureMatrix;
	}

/*	 Y' Y 
	 Read pre-calculated Y'Y (using CalcYtY MapReduce) 
	private Matrix readYtransposeYFromHdfs(Path pathToYty) {

		double[][] YtY = new double[numFeatures][numFeatures];
		
		boolean hasValue = false;
		
		for (Pair<NullWritable, MatrixEntryWritable> record : new SequenceFileDirIterable<NullWritable, MatrixEntryWritable>(
				pathToYty, PathType.LIST, PathFilters.partFilter(), null, true, conf)) {
			
			MatrixEntryWritable entry = record.getSecond();
			YtY[entry.getRow()][entry.getCol()] = entry.getVal();
			
			hasValue = true;
		}
		
		Preconditions.checkState(hasValue, "Dense matrix is empty");
		
		return new DenseMatrix(YtY, true);
	}*/

	/* Y' Y */
/*	private Matrix getYtransposeY(OpenIntObjectHashMap<Vector> Y) {

		IntArrayList indexes = Y.keys();
		indexes.quickSort();
		int numIndexes = indexes.size();

		double[][] YtY = new double[numFeatures][numFeatures];

		// Compute Y'Y by dot products between the 'columns' of Y
		for (int i = 0; i < numFeatures; i++) {
			for (int j = i; j < numFeatures; j++) {
				double dot = 0;
				for (int k = 0; k < numIndexes; k++) {
					Vector row = Y.get(indexes.getQuick(k));
					dot += row.getQuick(i) * row.getQuick(j);
				}
				YtY[i][j] = dot;
				if (i != j) {
					YtY[j][i] = dot;
				}
			}
		}
		return new DenseMatrix(YtY, true);
	}*/

	/** Y' (Cu - I) Y + λ I */
	private Matrix getYtransponseCuMinusIYPlusLambdaI(Vector userRatings) {
		Preconditions.checkArgument(userRatings.isSequentialAccess(),
				"need sequential access to ratings!");

		/* (Cu -I) Y */
		OpenIntObjectHashMap<Vector> CuMinusIY = new OpenIntObjectHashMap<Vector>(
				userRatings.getNumNondefaultElements());
		for (Element e : userRatings.nonZeroes()) {
			CuMinusIY.put(e.index(),
					Y.get(e.index()).times(confidence(e.get()) - 1));
		}

		Matrix YtransponseCuMinusIY = new DenseMatrix(numFeatures, numFeatures);

		/* Y' (Cu -I) Y by outer products */
		for (Element e : userRatings.nonZeroes()) {
			for (Vector.Element feature : Y.get(e.index()).all()) {
				Vector partial = CuMinusIY.get(e.index()).times(feature.get());
				YtransponseCuMinusIY.viewRow(feature.index()).assign(partial,
						Functions.PLUS);
			}
		}

		/* Y' (Cu - I) Y + λ I add lambda on the diagonal */
		for (int feature = 0; feature < numFeatures; feature++) {
			YtransponseCuMinusIY.setQuick(feature, feature,
					YtransponseCuMinusIY.getQuick(feature, feature) + lambda);
		}

		return YtransponseCuMinusIY;
	}

	/** Y' Cu p(u) */
	private Matrix getYtransponseCuPu(Vector userRatings) {
		Preconditions.checkArgument(userRatings.isSequentialAccess(),
				"need sequential access to ratings!");

		Vector YtransponseCuPu = new DenseVector(numFeatures);

		for (Element e : userRatings.nonZeroes()) {
			YtransponseCuPu.assign(Y.get(e.index()).times(confidence(e.get())),
					Functions.PLUS);
		}

		return columnVectorAsMatrix(YtransponseCuPu);
	}

	private Matrix columnVectorAsMatrix(Vector v) {
		double[][] matrix = new double[numFeatures][1];
		for (Vector.Element e : v.all()) {
			matrix[e.index()][0] = e.get();
		}
		return new DenseMatrix(matrix, true);
	}

}
