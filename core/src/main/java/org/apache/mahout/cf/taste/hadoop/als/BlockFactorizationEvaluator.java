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
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.jobcontrol.ControlledJob;
import org.apache.hadoop.mapreduce.lib.jobcontrol.JobControl;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.map.OpenIntObjectHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>Measures the root-mean-squared error of a rating matrix factorization against a test set.</p>
 *
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>--output (path): path where output should go</li>
 * <li>--pairs (path): path containing the test ratings, each line must be userID,itemID,rating</li>
 * <li>--userFeatures (path): path to the user feature matrix</li>
 * <li>--itemFeatures (path): path to the item feature matrix</li>
 * </ol>
 */
public class BlockFactorizationEvaluator extends AbstractJob {

	private static final Logger log = LoggerFactory
			.getLogger(BlockFactorizationEvaluator.class);
	
  private static final String USER_FEATURES_PATH = BlockRecommenderJob.class.getName() + ".userFeatures";
  private static final String ITEM_FEATURES_PATH = BlockRecommenderJob.class.getName() + ".itemFeatures";
  private static final String USER_BLOCKID = BlockFactorizationEvaluator.class.getName() + ".userBlockid";
  private static final String ITEM_BLOCKID = BlockFactorizationEvaluator.class.getName() + ".itemBlockid";
  
  private int numUserBlocks;
  private int numItemBlocks;
	
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new BlockFactorizationEvaluator(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOption("userFeatures", null, "path to the user feature matrix", true);
    addOption("itemFeatures", null, "path to the item feature matrix", true);
    addOption("usesLongIDs", null, "input contains long IDs that need to be translated");
	addOption("numUserBlocks", null,
			"number of User Block");
	addOption("numItemBlocks", null,
			"number of Item Block");
	
    addOutputOption();

    Map<String,List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

	numUserBlocks = Integer.parseInt(getOption("numUserBlocks"));
	numItemBlocks = Integer.parseInt(getOption("numItemBlocks"));

    JobControl control = new JobControl("BlockFactorizationEvaluator");
    for (int userBlockId = 0; userBlockId < numUserBlocks; userBlockId++) {
    	for (int itemBlockId = 0; itemBlockId < numItemBlocks; itemBlockId++) {
    		
    		String userItemBlockId = Integer.toString(userBlockId) + "-" + Integer.toString(itemBlockId); 
    		Path errors = new Path(getTempPath("errors"), userItemBlockId);
    		
    	    Job predictRatings = prepareJob(getInputPath(), errors , TextInputFormat.class, BlockPredictRatingsMapper.class,
    	            IntPairWritable.class, DoubleWritable.class, SequenceFileOutputFormat.class);

	        Configuration conf = predictRatings.getConfiguration();
	        conf.set(USER_FEATURES_PATH, getOption("userFeatures") + "/" + Integer.toString(userBlockId) + "-r-");
	        conf.set(ITEM_FEATURES_PATH, getOption("itemFeatures") + "/" + Integer.toString(itemBlockId) + "-r-");
	        conf.setInt(USER_BLOCKID, userBlockId);
	        conf.setInt(ITEM_BLOCKID, itemBlockId);
	        
	        boolean usesLongIDs = Boolean.parseBoolean(getOption("usesLongIDs"));
	        if (usesLongIDs) {
	          conf.set(ParallelALSFactorizationJob.USES_LONG_IDS, String.valueOf(true));
	        }
	        
	        control.addJob(new ControlledJob(conf));
    	}
    }
    
    Thread t = new Thread(control);
	log.info("Starting " + numUserBlocks * numItemBlocks + " block eval jobs.");
	t.start();
			
	while (!control.allFinished()) {
		Thread.sleep(1000);
	}
					
	List<ControlledJob> failedJob = control.getFailedJobList();
	
	if (failedJob != null && failedJob.size() > 0) {
		control.stop();
		throw new IllegalStateException("control job failed: " + failedJob);
	} else {
		log.info("control job finished");
	}
	
	control.stop();

	Job computeRmse = prepareJob(getTempPath("errors"),
			getOutputPath("rmse.txt"), ComputerRmseMapper.class,
			IntWritable.class ,DoubleIntPairWritable.class,
			ComputeRmseReducer.class, 
			DoubleWritable.class, NullWritable.class);
	
	computeRmse.setCombinerClass(ComputeRmseCombiner.class);
	
	log.info("Starting compute rmse job");
	boolean succeeded = computeRmse.waitForCompletion(true);
	if (!succeeded) {
		throw new IllegalStateException("compute rmse job failed!");
	}

    return 0;
  }

  double computeRmse(Path errors) {
    RunningAverage average = new FullRunningAverage();
    for (Pair<DoubleWritable,NullWritable> entry
        : new SequenceFileDirIterable<DoubleWritable, NullWritable>(errors, PathType.LIST, PathFilters.logsCRCFilter(),
          getConf())) {
      DoubleWritable error = entry.getFirst();
      average.addDatum(error.get() * error.get());
    }

    return Math.sqrt(average.getAverage());
  }

  public static class BlockPredictRatingsMapper extends Mapper<LongWritable,Text,IntPairWritable,DoubleWritable> {

    private OpenIntObjectHashMap<Vector> U;
    private OpenIntObjectHashMap<Vector> M;

    private boolean usesLongIDs;
    private int userBlockId;
    private int itemBlockId;
    
    private final DoubleWritable error = new DoubleWritable();
    private final IntPairWritable outkey = new IntPairWritable();
    
    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      Configuration conf = ctx.getConfiguration();

      Path pathToU = new Path(conf.get(USER_FEATURES_PATH));
      Path pathToM = new Path(conf.get(ITEM_FEATURES_PATH));

      U = ALS.readMatrixByRowsGlob(pathToU, conf);
      M = ALS.readMatrixByRowsGlob(pathToM, conf);

      usesLongIDs = conf.getBoolean(ParallelALSFactorizationJob.USES_LONG_IDS, false);
      
      userBlockId = conf.getInt(USER_BLOCKID, 0);
      itemBlockId = conf.getInt(ITEM_BLOCKID, 0);
    }

    @Override
    protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {

      String[] tokens = TasteHadoopUtils.splitPrefTokens(value.toString());

      int userID = TasteHadoopUtils.readID(tokens[TasteHadoopUtils.USER_ID_POS], usesLongIDs);
      int itemID = TasteHadoopUtils.readID(tokens[TasteHadoopUtils.ITEM_ID_POS], usesLongIDs);
      double rating = Double.parseDouble(tokens[2]);

      if (U.containsKey(userID) && M.containsKey(itemID)) {
        double estimate = U.get(userID).dot(M.get(itemID));
        error.set(rating - estimate);
        outkey.setFirst(userBlockId);
        outkey.setSecond(itemBlockId);
        
        ctx.write(outkey, error);
      }
    }
  }

  static class ComputerRmseMapper extends
  	Mapper<IntPairWritable, DoubleWritable, IntWritable, DoubleIntPairWritable> {
	  	IntWritable outkey = new IntWritable(0);
	  	DoubleIntPairWritable error = new DoubleIntPairWritable();
	  	
		@Override
		protected void map(IntPairWritable key, DoubleWritable value, Context ctx)
				throws IOException, InterruptedException {
			error.setFirst(value.get() * value.get()); // error * error
			error.setSecond(1);
			ctx.write(outkey, error);
		}
  }
  
  static class ComputeRmseCombiner extends
	Reducer<IntWritable, DoubleIntPairWritable, IntWritable, DoubleIntPairWritable> {

	private DoubleIntPairWritable value = new DoubleIntPairWritable();

	@Override
	public void reduce(IntWritable key, Iterable<DoubleIntPairWritable> vectors, Context ctx)
			throws IOException, InterruptedException {
  
		double sum = 0.0;
		int count = 0;
		Iterator<DoubleIntPairWritable> iter = vectors.iterator();
		while (iter.hasNext()) {
			DoubleIntPairWritable avgInfo = iter.next();
			sum += avgInfo.getFirst() * avgInfo.getSecond();
			count += avgInfo.getSecond();
		}
  
		value.setFirst(sum/count);
		value.setSecond(count);
		ctx.write(key, value);
	}
  }	
  
	
  static class ComputeRmseReducer extends
  	Reducer<IntWritable, DoubleIntPairWritable, DoubleWritable, NullWritable> {

	  	private DoubleWritable rmse = new DoubleWritable();

		@Override
		public void reduce(IntWritable key, Iterable<DoubleIntPairWritable> errors, Context ctx)
				throws IOException, InterruptedException {
		
			Iterator<DoubleIntPairWritable> iter = errors.iterator();
			
			double sum = 0.0;
			int count = 0;
			
			while (iter.hasNext()) {
				DoubleIntPairWritable avgInfo = iter.next();
				sum += avgInfo.getFirst() * avgInfo.getSecond();
				count += avgInfo.getSecond();
			}
		
			rmse.set(Math.sqrt(sum/count));
			ctx.write(rmse, NullWritable.get());
			
		}
  }	

}
