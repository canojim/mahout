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
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.jobcontrol.ControlledJob;
import org.apache.hadoop.mapreduce.lib.jobcontrol.JobControl;
import org.apache.hadoop.mapreduce.lib.map.MultithreadedMapper;
import org.apache.hadoop.mapreduce.lib.output.LazyOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.mapreduce.MergeVectorsCombiner;
import org.apache.mahout.common.mapreduce.MergeVectorsReducer;
import org.apache.mahout.common.mapreduce.VectorSumCombiner;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix.MatrixEntryWritable;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.Vectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;

/**
 * <p>
 * MapReduce implementation of the two factorization algorithms described in
 * 
 * <p>
 * "Large-scale Parallel Collaborative Filtering for the Netﬂix Prize" available
 * at http://www.hpl.hp.com/personal/Robert_Schreiber/papers/2008%20AAIM%20
 * Netflix/netflix_aaim08(submitted).pdf.
 * </p>
 * 
 * "
 * <p>
 * Collaborative Filtering for Implicit Feedback Datasets" available at
 * http://research.yahoo.com/pub/2433
 * </p>
 * 
 * </p>
 * <p>
 * Command line arguments specific to this class are:
 * </p>
 * 
 * <ol>
 * <li>--input (path): Directory containing one or more text files with the
 * dataset</li>
 * <li>--output (path): path where output should go</li>
 * <li>--lambda (double): regularization parameter to avoid overfitting</li>
 * <li>--userFeatures (path): path to the user feature matrix</li>
 * <li>--itemFeatures (path): path to the item feature matrix</li>
 * <li>--numThreadsPerSolver (int): threads to use per solver mapper, (default:
 * 1)</li>
 * </ol>
 */
public class BlockParallelALSFactorizationJob extends AbstractJob {

	private static final Logger log = LoggerFactory
			.getLogger(BlockParallelALSFactorizationJob.class);


	static final String LAMBDA = BlockParallelALSFactorizationJob.class.getName()
			+ ".lambda";
	static final String ALPHA = BlockParallelALSFactorizationJob.class.getName()
			+ ".alpha";
	static final String NUM_ENTITIES = BlockParallelALSFactorizationJob.class
			.getName() + ".numEntities";

	static final String USES_LONG_IDS = BlockParallelALSFactorizationJob.class
			.getName() + ".usesLongIDs";
	static final String TOKEN_POS = BlockParallelALSFactorizationJob.class.getName()
			+ ".tokenPos";

	static final String PATH_TO_YTY = BlockParallelALSFactorizationJob.class
			.getName() + ".pathToYty";

	static final String NUM_BLOCKS = "numberOfBlocks";

	private boolean implicitFeedback;
	private int numIterations;
	private int numFeatures;
	private double lambda;
	private double alpha;
	private int numThreadsPerSolver;
	private boolean usesLongIDs;

	private int numItems;
	private int numUsers;
	private int numUserBlocks;
	private int numItemBlocks;

	enum Stats {
		NUM_USERS
	}

	public static void main(String[] args) throws Exception {
		ToolRunner.run(new BlockParallelALSFactorizationJob(), args);
	}

	@Override
	public int run(String[] args) throws Exception {

		addInputOption();
		addOutputOption();
		addOption("lambda", null, "regularization parameter", true);
		addOption("implicitFeedback", null,
				"data consists of implicit feedback?", String.valueOf(false));
		addOption("alpha", null,
				"confidence parameter (only used on implicit feedback)",
				String.valueOf(40));
		addOption("numFeatures", null, "dimension of the feature space", true);
		addOption("numIterations", null, "number of iterations", true);
		addOption("numThreadsPerSolver", null, "threads per solver mapper",
				String.valueOf(1));
		addOption("usesLongIDs", null,
				"input contains long IDs that need to be translated");
		addOption("numUserBlocks", null,
				"number of User Block");
		addOption("numItemBlocks", null,
				"number of Item Block");
		

		Map<String, List<String>> parsedArgs = parseArguments(args);
		if (parsedArgs == null) {
			return -1;
		}

		numFeatures = Integer.parseInt(getOption("numFeatures"));
		numIterations = Integer.parseInt(getOption("numIterations"));
		lambda = Double.parseDouble(getOption("lambda"));
		alpha = Double.parseDouble(getOption("alpha"));
		implicitFeedback = Boolean.parseBoolean(getOption("implicitFeedback"));

		numThreadsPerSolver = Integer
				.parseInt(getOption("numThreadsPerSolver"));
		usesLongIDs = Boolean.parseBoolean(getOption("usesLongIDs",
				String.valueOf(false)));
		numUserBlocks = Integer.parseInt(getOption("numUserBlocks"));
		numItemBlocks = Integer.parseInt(getOption("numItemBlocks"));

		/*
		 * compute the factorization A = U M'
		 * 
		 * where A (users x items) is the matrix of known ratings U (users x
		 * features) is the representation of users in the feature space M
		 * (items x features) is the representation of items in the feature
		 * space
		 */
		boolean succeeded = false;
		
		if (usesLongIDs) {
			Job mapUsers = prepareJob(getInputPath(),
					getOutputPath("userIDIndex"), TextInputFormat.class,
					MapLongIDsMapper.class, IntPairWritable.class,
					VarLongWritable.class, IDMapReducer.class,
					VarIntWritable.class, VarLongWritable.class,
					SequenceFileOutputFormat.class);
			mapUsers.getConfiguration().set(TOKEN_POS,
					String.valueOf(TasteHadoopUtils.USER_ID_POS));
			
			LazyOutputFormat.setOutputFormatClass(mapUsers, SequenceFileOutputFormat.class);
			for (int blockId = 0; blockId < numItemBlocks; blockId++) {
				MultipleOutputs.addNamedOutput(mapUsers, Integer.toString(blockId), SequenceFileOutputFormat.class, 
						VarIntWritable.class, VarLongWritable.class);
			}
			
			mapUsers.waitForCompletion(true);

			Job mapItems = prepareJob(getInputPath(),
					getOutputPath("itemIDIndex"), TextInputFormat.class,
					MapLongIDsMapper.class, IntPairWritable.class,
					VarLongWritable.class, IDMapReducer.class,
					VarIntWritable.class, VarLongWritable.class,
					SequenceFileOutputFormat.class);
			mapItems.getConfiguration().set(TOKEN_POS,
					String.valueOf(TasteHadoopUtils.ITEM_ID_POS));

			LazyOutputFormat.setOutputFormatClass(mapItems, SequenceFileOutputFormat.class);
			for (int blockId = 0; blockId < numItemBlocks; blockId++) {
				MultipleOutputs.addNamedOutput(mapItems, Integer.toString(blockId), SequenceFileOutputFormat.class, 
						VarIntWritable.class, VarLongWritable.class);
			}			
			
			mapItems.waitForCompletion(true);
		}
		//input content: uID,mID,rating E.g. 21349098,444875844,2 21349098,1436281125,1 21349098,1856996949,1
		//output filename: als/tmp/itemRatings/blockID-r-nnnnn E.g. 0-r-00000 1-r-00000
		//output content: uID, Vector of mID:rating E.g. 21349098 {444875844:2.0,1436281125:1.0,1856996949:1.0}


		/* create A' */
		Job itemRatings = prepareJob(getInputPath(), pathToItemRatings(),
				TextInputFormat.class, ItemRatingVectorsMapper.class,
				IntPairWritable.class, VectorWritable.class,
				VectorSumReducer.class, IntWritable.class,
				VectorWritable.class, SequenceFileOutputFormat.class);

		// use multiple output to suport block
		LazyOutputFormat.setOutputFormatClass(itemRatings, SequenceFileOutputFormat.class);
	    for (int blockId = 0; blockId < numUserBlocks; blockId++) {
	      MultipleOutputs.addNamedOutput(itemRatings, Integer.toString(blockId), SequenceFileOutputFormat.class, 
	      																IntWritable.class, VectorWritable.class);
	    }

		itemRatings.setCombinerClass(VectorSumCombiner.class);
		itemRatings.getConfiguration().set(USES_LONG_IDS,
				String.valueOf(usesLongIDs));
		itemRatings.getConfiguration().setInt(NUM_BLOCKS,
				numUserBlocks);
		succeeded = itemRatings.waitForCompletion(true);
		if (!succeeded) {
			return -1;
		}
				
		//output file: /als/out/userRatings/0-r-00000
		//output file content:

		/* create A */
		Job userRatings = prepareJob(pathToItemRatings(), pathToUserRatings(),
				BlockTransposeMapper.class, IntPairWritable.class, VectorWritable.class,
				MergeUserVectorsReducer.class, IntWritable.class,
				VectorWritable.class);

		// use multiple output to support block
		LazyOutputFormat.setOutputFormatClass(userRatings, SequenceFileOutputFormat.class);
		for (int blockId = 0; blockId < numItemBlocks; blockId++) {
			MultipleOutputs.addNamedOutput(userRatings, Integer.toString(blockId), SequenceFileOutputFormat.class, 
      																IntWritable.class, VectorWritable.class);
		}

		userRatings.setCombinerClass(MergeVectorsCombiner.class);
		userRatings.getConfiguration().set(NUM_BLOCKS,
				String.valueOf(numItemBlocks));

		succeeded = userRatings.waitForCompletion(true);
		if (!succeeded) {
			return -1;
		}
		
		//als/tmp/averageRatings/part-r-00000
		
		// TODO this could be fiddled into one of the upper jobs
		Job averageItemRatings = prepareJob(pathToItemRatings(),
				getTempPath("averageRatings"), AverageRatingMapper.class,
				IntWritable.class, VectorWritable.class,
				MergeVectorsReducer.class, IntWritable.class,
				VectorWritable.class);
		averageItemRatings.setCombinerClass(MergeVectorsCombiner.class);
		succeeded = averageItemRatings.waitForCompletion(true);
		if (!succeeded) {
			return -1;
		}

		Vector averageRatings = ALS.readFirstRow(getTempPath("averageRatings"),
				getConf());

		numItems = averageRatings.getNumNondefaultElements();
		numUsers = (int) userRatings.getCounters().findCounter(Stats.NUM_USERS)
				.getValue();

		log.info("Found {} users and {} items", numUsers, numItems);
		
		/* create an initial M */
		initializeBlockM(averageRatings, numItemBlocks);		

		for (int currentIteration = 0; currentIteration < numIterations; currentIteration++) {
			/* broadcast M, read A row-wise, recompute U row-wise */
			log.info("Recomputing U (iteration {}/{})", currentIteration,
					numIterations);
			runSolver(pathToUserRatings(), pathToU(currentIteration),
					pathToM(currentIteration - 1),
					pathToPrefix("UYtY", currentIteration), currentIteration, "U",
					numItems, numUserBlocks, numItemBlocks);
			/* broadcast U, read A' row-wise, recompute M row-wise */
			log.info("Recomputing M (iteration {}/{})", currentIteration,
					numIterations);
			runSolver(pathToItemRatings(), pathToM(currentIteration),
					pathToU(currentIteration),
					pathToPrefix("MYtY", currentIteration), currentIteration, "M",
					numUsers, numUserBlocks, numItemBlocks);
		}

		return 0;
	}

	private void initializeM(Vector averageRatings, String partName)
			throws IOException {

		final int TOTAL_PART_NUM = 10;

		Random random = RandomUtils.getRandom();
		int partSize = (int) Math.ceil(averageRatings.getNumNondefaultElements() / TOTAL_PART_NUM);

		FileSystem fs = FileSystem.get(pathToM(-1).toUri(), getConf());
		SequenceFile.Writer writer = null;
		long count = 0;
		long prevPartId = -1;
		
		try {	
				IntWritable index = new IntWritable();
				VectorWritable featureVector = new VectorWritable();
	
				for (Vector.Element e : averageRatings.nonZeroes()) {
					
					long partId = (count++)/partSize;
					
					if (partId != prevPartId) {
						
						if (prevPartId != -1) {
							Closeables.close(writer, false);
						}
						
						String partPath = partName + "-r-" + String.format("%05d", partId);
						writer = new SequenceFile.Writer(fs, getConf(), new Path(
								pathToM(-1), partPath), IntWritable.class,
								VectorWritable.class);
						prevPartId = partId;
					}

					Vector row = new DenseVector(numFeatures);
					row.setQuick(0, e.get());
					for (int m = 1; m < numFeatures; m++) {
						row.setQuick(m, random.nextDouble());
					}
					index.set(e.index());
					featureVector.set(row);
					writer.append(index, featureVector);
				}
		} finally {
			Closeables.close(writer, false);
		}
	}	
		
	/**
	 * 
	 * Change Log: Split to 100 parts.
	 * @param averageRatings
	 * @param numEntity
	 * @throws IOException
	 */
	private void initializeBlockM(Vector averageRatings, int numItemBlocks)
			throws IOException {

		HashMap<String, Vector> partMap = new HashMap<String, Vector>();		
		for (Vector.Element e : averageRatings.nonZeroes()) {
					
			int partName = BlockPartitionUtil.getBlockID(e.index(), numItemBlocks);
					
			Vector blockRatings = partMap.get(Integer.toString(partName));
			if (blockRatings == null) {
				blockRatings = new RandomAccessSparseVector(averageRatings.getNumNondefaultElements(), 1);
				partMap.put(Integer.toString(partName), blockRatings);
			}
			
			blockRatings.setQuick(e.index(), e.get());
		}
				
		Iterator it = partMap.entrySet().iterator();
		while (it.hasNext()) {
			Map.Entry<String, Vector> pairs = (Map.Entry<String, Vector>)it.next();
			initializeM(pairs.getValue(), pairs.getKey());
			it.remove(); // avoids a ConcurrentModificationException
		}
	}	

	static class VectorSumReducer extends
		Reducer<IntPairWritable, VectorWritable, IntWritable, VectorWritable> { //WritableComparable<?>

		private MultipleOutputs<IntWritable, VectorWritable> out;
		private final IntWritable resultKey = new IntWritable();
		private final VectorWritable resultValue = new VectorWritable();

		@Override
		protected void setup(Context ctx) throws IOException, InterruptedException {
			out = new MultipleOutputs<IntWritable, VectorWritable>(ctx);
		}

		@Override
		protected void reduce(IntPairWritable key,
				Iterable<VectorWritable> values, Context ctx)
				throws IOException, InterruptedException {
			Vector sum = Vectors.sum(values.iterator());
			
			resultKey.set(key.getFirst());
			resultValue.set(new SequentialAccessSparseVector(sum));
			//System.out.println("x reduce: " + Integer.toString(key.getSecond()) + " key: " + resultKey + " value: " + resultValue);
			out.write(Integer.toString(key.getSecond()), resultKey, resultValue);
		}

		@Override
		protected void cleanup(Context context)
				throws IOException, InterruptedException {
			out.close();
		}
		
		
	}

	static class MergeUserVectorsReducer extends
			Reducer<IntPairWritable, VectorWritable, IntWritable, VectorWritable> {

		private MultipleOutputs<IntWritable,VectorWritable> out;
		private final IntWritable resultKey = new IntWritable();
		private final VectorWritable resultValue = new VectorWritable();

		@Override
		protected void setup(Context ctx) throws IOException, InterruptedException {
			out = new MultipleOutputs<IntWritable,VectorWritable>(ctx);
		}

		@Override
		public void reduce(IntPairWritable key,
				Iterable<VectorWritable> values, Context ctx)
				throws IOException, InterruptedException {
			Vector merged = VectorWritable.merge(values.iterator()).get();

			resultKey.set(key.getFirst());
			resultValue.set(new SequentialAccessSparseVector(merged));
			out.write(Integer.toString(key.getSecond()), resultKey, resultValue);

			//System.out.println("MergeUserVectorsReducer: " + Integer.toString(key.getSecond()) + " key: " + resultKey + " value: " + resultValue);
			ctx.getCounter(Stats.NUM_USERS).increment(1);
		}

		@Override
		protected void cleanup(Context context)
				throws IOException, InterruptedException {
			out.close();
		}
		
		
	}

	static class ItemRatingVectorsMapper extends
			Mapper<LongWritable, Text, IntPairWritable, VectorWritable> {

		private final IntPairWritable key = new IntPairWritable();
		private final VectorWritable value = new VectorWritable(true);
		private final Vector ratings = new RandomAccessSparseVector(
				Integer.MAX_VALUE, 1);

		private boolean usesLongIDs;
		private int numUserBlocks;

		@Override
		protected void setup(Context ctx) throws IOException,
				InterruptedException {
			usesLongIDs = ctx.getConfiguration().getBoolean(USES_LONG_IDS,
					false);
			numUserBlocks = ctx.getConfiguration().getInt(NUM_BLOCKS, 10);
		}

		@Override
		protected void map(LongWritable offset, Text line, Context ctx)
				throws IOException, InterruptedException {
			String[] tokens = TasteHadoopUtils.splitPrefTokens(line.toString());
			int userID = TasteHadoopUtils.readID(
					tokens[TasteHadoopUtils.USER_ID_POS], usesLongIDs);
			int itemID = TasteHadoopUtils.readID(
					tokens[TasteHadoopUtils.ITEM_ID_POS], usesLongIDs);
			float rating = Float.parseFloat(tokens[2]);

			ratings.setQuick(userID, rating);

			key.setFirst(itemID);
			key.setSecond(BlockPartitionUtil.getBlockID(userID, numUserBlocks));
			
			//System.out.println("key: " + key.toString());
			value.set(ratings);

			ctx.write(key, value);

			// prepare instance for reuse
			ratings.setQuick(userID, 0.0d);
		}
	}

	static class BlockTransposeMapper extends
		Mapper<IntWritable, VectorWritable, IntPairWritable, VectorWritable> {

		private final IntPairWritable key = new IntPairWritable();

		private int numItemBlocks;

		@Override
		protected void setup(Context ctx) throws IOException,
				InterruptedException {
			numItemBlocks = ctx.getConfiguration().getInt(NUM_BLOCKS, 10);
		}

	 	@Override
	 	protected void map(IntWritable r, VectorWritable v, Context ctx) throws IOException, InterruptedException {
	 		int row = r.get();
	 		
	 		Iterator<Vector.Element> it = v.get().nonZeroes().iterator();
	 		
	 		while (it.hasNext()) {
	 			Vector.Element e = it.next();
	 			RandomAccessSparseVector tmp = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);
	 			tmp.setQuick(row, e.get());
	 			key.setFirst(e.index());
	 			key.setSecond(BlockPartitionUtil.getBlockID(row, numItemBlocks));
	 			ctx.write(key, new VectorWritable(tmp));
	 		}
	 	}
	}

	private void runSolver(Path ratings, Path output, Path pathToUorM,
			Path pathToYty, int currentIteration, String matrixName,
			int numEntities, int numBlocks1, int numBlocks2) throws ClassNotFoundException, IOException,
			InterruptedException {

		// necessary for local execution in the same JVM only
		SharingMapper.reset();

		Class<? extends Mapper<IntWritable, VectorWritable, IntWritable, ALSContributionWritable>> solverMapperClassInternal;
		String name;

		if (implicitFeedback) {
			solverMapperClassInternal = BlockSolveImplicitFeedbackMapper.class;
			name = "Recompute " + matrixName + ", iteration ("
					+ currentIteration + '/' + numIterations + "), " + '('
					+ numThreadsPerSolver + " threads, " + numFeatures
					+ " features, implicit feedback)";
		} else {
			//TODO: support explicit feedback
			throw new RuntimeException("Explicit feedback currently not supported in block version.");
		}
		
		boolean succeeded = false;
		
		// prepareJob to calculate Y'Y
		Job calYtY = prepareJob(pathToUorM, pathToYty,
				SequenceFileInputFormat.class, CalcYtYMapper.class,
				MatrixEntryWritable.class, DoubleWritable.class,
				CalcYtYReducer.class, NullWritable.class,
				MatrixEntryWritable.class, SequenceFileOutputFormat.class);

		calYtY.setCombinerClass(CalcYtyCombiner.class);

		Configuration calYtYConf = calYtY.getConfiguration();
		//System.out.println("numFeatures: " + numFeatures);
		
		calYtYConf.setInt(CalcYtYMapper.NUM_FEATURES, numFeatures);

		log.info("Starting YtY job");
		succeeded = calYtY.waitForCompletion(true);
		if (!succeeded) {
			throw new IllegalStateException("calYtY Job failed!");
		}

		JobControl control = new JobControl("BlockParallelALS");
		
		String blockOutputName = "BlockRatingOutput-" + matrixName + "-" + Integer.toString(currentIteration-1);
		
		for (int blockId = 0; blockId < numBlocks2; blockId++) {
			// process each block
			Path blockRatings = new Path(ratings.toString() + "/" + Integer.toString(blockId) + "-r-*");
			
			Path blockRatingsOutput = new Path(getTempPath(blockOutputName).toString() + "/" + Integer.toString(blockId));
			Path blockFixUorM = new Path(pathToUorM.toString() + "/" + Integer.toString(blockId) + "-r-*");
				
			Job solveBlockUorI = prepareJob(blockRatings, blockRatingsOutput,
						SequenceFileInputFormat.class,
						MultithreadedSharingMapper.class, IntWritable.class,
						ALSContributionWritable.class, SequenceFileOutputFormat.class, name + " blockId: " + blockId);
			Configuration solverConf = solveBlockUorI.getConfiguration();
			solverConf.set(LAMBDA, String.valueOf(lambda));
			solverConf.set(ALPHA, String.valueOf(alpha));
			solverConf.setInt(CalcYtYMapper.NUM_FEATURES, numFeatures);
			solverConf.set(NUM_ENTITIES, String.valueOf(numEntities));

			FileSystem fs = FileSystem.get(blockFixUorM.toUri(), solverConf);
			
			FileStatus[] parts = fs.globStatus(blockFixUorM);
			
			log.info("Pushing " + parts.length + " files to distributed cache.");
			for (FileStatus part : parts) {
				//System.out.println("Adding {} to distributed cache: " + part.getPath().toString());
				if (log.isDebugEnabled()) {
					log.debug("Adding {} to distributed cache", part.getPath()
						.toString());
				}
				DistributedCache.addCacheFile(part.getPath().toUri(), solverConf);
			}
				
			MultithreadedMapper.setMapperClass(solveBlockUorI,
						solverMapperClassInternal);
			MultithreadedMapper.setNumberOfThreads(solveBlockUorI,
						numThreadsPerSolver);
				
			control.addJob(new ControlledJob(solverConf));
		}

		Thread t = new Thread(control);
		log.info("Starting " + numBlocks2 + " block rating jobs.");
		t.start();
				
		while (!control.allFinished()) {
			Thread.sleep(1000);
		}
						
		List<ControlledJob> failedJob = control.getFailedJobList();
		
		if (failedJob != null && failedJob.size() > 0) {
			throw new IllegalStateException("control job failed: " + failedJob);
		} else {
			log.info("control job finished");
		}
		
		control.stop();
		
		log.info("Aggregating block result");
		Path updateInputPath = new Path(getTempPath(blockOutputName).toString() + "/*/");
		Job updateUorM = prepareJob(updateInputPath, output,
				SequenceFileInputFormat.class, Mapper.class,
				IntWritable.class, ALSContributionWritable.class,
				UpdateUorMReducer.class, IntWritable.class,
				VectorWritable.class, SequenceFileOutputFormat.class);

		// use multiple output to suport block
		LazyOutputFormat.setOutputFormatClass(updateUorM, SequenceFileOutputFormat.class);
		for (int blockId = 0; blockId < numBlocks1; blockId++) {
			MultipleOutputs.addNamedOutput(updateUorM, Integer.toString(blockId), SequenceFileOutputFormat.class, 
      																IntWritable.class, VectorWritable.class);
		}

		updateUorM.setCombinerClass(UpdateUorMCombiner.class);
		updateUorM.getConfiguration().setInt(CalcYtYMapper.NUM_FEATURES, numFeatures);
		updateUorM.getConfiguration().set(NUM_BLOCKS,
				String.valueOf(numBlocks1));
		updateUorM.getConfiguration().set(PATH_TO_YTY, pathToYty.toString());

		succeeded = updateUorM.waitForCompletion(true);
		if (!succeeded) {
			throw new RuntimeException("updateUorM job failed!");
		}

	}

	static class AverageRatingMapper extends
			Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

		private final IntWritable firstIndex = new IntWritable(0);
		private final Vector featureVector = new RandomAccessSparseVector(
				Integer.MAX_VALUE, 1);
		private final VectorWritable featureVectorWritable = new VectorWritable();

		@Override
		protected void map(IntWritable r, VectorWritable v, Context ctx)
				throws IOException, InterruptedException {
			RunningAverage avg = new FullRunningAverage();
			for (Vector.Element e : v.get().nonZeroes()) {
				avg.addDatum(e.get());
			}

			featureVector.setQuick(r.get(), avg.getAverage());
			featureVectorWritable.set(featureVector);
			ctx.write(firstIndex, featureVectorWritable);

			// prepare instance for reuse
			featureVector.setQuick(r.get(), 0.0d);
		}
	}

	static class MapLongIDsMapper extends
			Mapper<LongWritable, Text, IntPairWritable, VarLongWritable> {

		private int tokenPos, numUserBlocks;
		private final IntPairWritable index = new IntPairWritable();
		private final VarLongWritable idWritable = new VarLongWritable();

		@Override
		protected void setup(Context ctx) throws IOException,
				InterruptedException {
			tokenPos = ctx.getConfiguration().getInt(TOKEN_POS, -1);
			numUserBlocks = ctx.getConfiguration().getInt(NUM_BLOCKS, 10);
			
			Preconditions.checkState(tokenPos >= 0);
			
		}

		@Override
		protected void map(LongWritable key, Text line, Context ctx)
				throws IOException, InterruptedException {
			String[] tokens = TasteHadoopUtils.splitPrefTokens(line.toString());

			long id = Long.parseLong(tokens[tokenPos]);
			
			int shortId = TasteHadoopUtils.idToIndex(id);
			index.setFirst(shortId);
			index.setSecond(BlockPartitionUtil.getBlockID(shortId, numUserBlocks));
			idWritable.set(id);
			ctx.write(index, idWritable);
		}
	}

	static class IDMapReducer
			extends
			Reducer<IntPairWritable, VarLongWritable, VarIntWritable, VarLongWritable> {
		
		private MultipleOutputs<VarIntWritable,VarLongWritable> out;
		
		@Override
		protected void cleanup(Context context) throws IOException,
				InterruptedException {
			out.close();
		}

		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			out = new MultipleOutputs<VarIntWritable,VarLongWritable>(context);
		}

		@Override
		protected void reduce(IntPairWritable index,
				Iterable<VarLongWritable> ids, Context ctx) throws IOException,
				InterruptedException {
			//ctx.write(index, ids.iterator().next());			
			out.write(Integer.toString(index.getSecond()), new VarIntWritable(index.getFirst()), ids.iterator().next());
		}
	}

	private Path pathToM(int iteration) {
		return iteration == numIterations - 1 ? getOutputPath("M")
				: getTempPath("M-" + iteration);
	}

	private Path pathToU(int iteration) {
		return iteration == numIterations - 1 ? getOutputPath("U")
				: getTempPath("U-" + iteration);
	}

	private Path pathToPrefix(String prefix, int iteration) {
		return iteration == numIterations - 1 ? getOutputPath(prefix)
				: getTempPath(prefix + iteration);
	}

	private Path pathToItemRatings() {
		return getTempPath("itemRatings");
	}

	private Path pathToUserRatings() {
		return getOutputPath("userRatings");
	}
}
