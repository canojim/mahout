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
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>
 * Sum up all features for ringfence type of recommendation
 * </p>
 * 
 * <p>
 * Command line arguments specific to this class are:
 * </p>
 * 
 * <ol>
 * <li>--input (path): Directory containing the U or M</li>
 * <li>--output (path): path where output should go</li>
 * </ol>
 */
public class SumAllFeaturesJob extends AbstractJob {
	
	public static final String OUTPUT_SHORT_ID = "outputShortId";
	
	private static final Logger log = LoggerFactory
			.getLogger(BlockParallelALSFactorizationJob.class);
	
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new SumAllFeaturesJob(), args);
	}

	@Override
	public int run(String[] args) throws Exception {

		addInputOption();
		addOption("queueName", null,
				"mapreduce queueName. (optional)", "default");		
		addOption(OUTPUT_SHORT_ID, null,
				"short id for ringfence. (optional)", "-1");
		
		addOutputOption();

		Map<String, List<String>> parsedArgs = parseArguments(args);
		if (parsedArgs == null) {
			return -1;
		}

		Configuration defaultConf = new Configuration();
		FileSystem fs = FileSystem.get(defaultConf);
		
		boolean succeeded = false;
		
		Path allUserFeaturesPath = getInputPath(); 		
		Path sumAllUserFeaturePath = getOutputPath();		
		
		if (!fs.exists(new Path(sumAllUserFeaturePath.toString() + "/_SUCCESS"))) {
			Job sumAllUserFeature = prepareJob(allUserFeaturesPath,
					sumAllUserFeaturePath, VectorSumMapper.class,
					IntWritable.class, VectorWritable.class,
					VectorSumReducer.class, IntWritable.class,
					VectorWritable.class);
			
			int shortId = Integer.parseInt(getOption(OUTPUT_SHORT_ID));
			if (shortId > 0)
				shortId = -shortId;
			
			sumAllUserFeature.getConfiguration().set(JobManager.QUEUE_NAME, getOption("queueName"));
			sumAllUserFeature.getConfiguration().setInt(OUTPUT_SHORT_ID, shortId);
			
			log.info("Starting sumAllUserFeature job.");
			succeeded = sumAllUserFeature.waitForCompletion(true);
			if (!succeeded) {
				throw new IllegalStateException("sumAllUserFeature job failed");
			}				
		}						
		
		return 0;
	}

	static class VectorSumMapper extends
			Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

		final IntWritable outKey = new IntWritable(-1);
		
		@Override
		protected void map(IntWritable key, VectorWritable value,
				Context context)
				throws IOException, InterruptedException {
			context.write(outKey, value);
		}
				
	}
			
	static class VectorSumReducer extends
			Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {

		@Override
		protected void reduce(IntWritable key, Iterable<VectorWritable> features,
				Context ctx)
				throws IOException, InterruptedException {
			
			VectorWritable vw = VectorWritable.mergeAverage(features.iterator());			
			
			int k = ctx.getConfiguration().getInt(OUTPUT_SHORT_ID, -1); 
			IntWritable reduceOutKey = new IntWritable(k);
			
			ctx.write(reduceOutKey, vw);
		}
				
	}
			
}
