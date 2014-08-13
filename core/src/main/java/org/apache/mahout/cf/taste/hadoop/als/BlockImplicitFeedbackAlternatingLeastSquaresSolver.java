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
public class BlockImplicitFeedbackAlternatingLeastSquaresSolver {

	private final int numFeatures;
	private final double alpha;
	private final double lambda;

	private OpenIntObjectHashMap<Vector> Y;

	private static final Logger log = LoggerFactory.getLogger(
									BlockImplicitFeedbackAlternatingLeastSquaresSolver.class);	

	/*
	 * public DistributedImplicitFeedbackAlternatingLeastSquaresSolver(int
	 * numFeatures, double lambda, double alpha, OpenIntObjectHashMap<Vector> Y)
	 * { this.numFeatures = numFeatures; this.lambda = lambda; this.alpha =
	 * alpha; this.Y = Y; YtransposeY = getYtransposeY(Y); }
	 */

	public BlockImplicitFeedbackAlternatingLeastSquaresSolver(
			int numFeatures, int numEntities, double lambda, double alpha, 
			OpenIntObjectHashMap<Vector> Y) throws IOException {
		this.numFeatures = numFeatures;
		this.lambda = lambda;
		this.alpha = alpha;
		this.Y = Y;
	}

	public Matrix solveA(Vector ratings) throws IOException {		
		return getYtransponseCuMinusIYPlusLambdaI(ratings);
	}

	public Matrix solveb(Vector ratings) throws IOException {
		return getYtransponseCuPu(ratings);
	}

	double confidence(double rating) {
		return 1 + alpha * rating;
	}

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
