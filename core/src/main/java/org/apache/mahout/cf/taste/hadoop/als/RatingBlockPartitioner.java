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

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.List;
import java.util.Map;

import com.google.common.base.Charsets;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.LazyOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
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
public class RatingBlockPartitioner extends AbstractJob {

  private static final String NUMBER_STRIPES = "options.numStripes";
  private static final String OUTPUT_ROOT = "options.outputRoot";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new RatingBlockPartitioner(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOption("numStripes", null, "number of the stripes to be divided for each dimension", true);
    addOutputOption();

    Map<String,List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Job segmentRatings = prepareJob(getInputPath(), getOutputPath(), TextInputFormat.class, RatingBlockMapper.class,
        NullWritable.class, Text.class, TextOutputFormat.class);

    LazyOutputFormat.setOutputFormatClass(segmentRatings, TextOutputFormat.class);

    Configuration conf = segmentRatings.getConfiguration();
    conf.set(NUMBER_STRIPES, getOption("numStripes"));

    int numBlocks = (int)(Math.pow(Integer.parseInt(getOption("numStripes")), 2.0));
    for (int blockId = 0; blockId < numBlocks; blockId++) {
      MultipleOutputs.addNamedOutput(segmentRatings, Integer.toString(blockId), TextOutputFormat.class, NullWritable.class, Text.class);
    }


    boolean succeeded = segmentRatings.waitForCompletion(true);

    if (!succeeded) {
      return -1;
    }
    else {
      return 1;
    }
  }


  public static class RatingBlockMapper extends Mapper<LongWritable,Text,NullWritable,Text> {
    //
    private MultipleOutputs out;
    private int numStripes;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      Configuration conf = ctx.getConfiguration();
      numStripes = Integer.parseInt(conf.get(NUMBER_STRIPES));

      out = new MultipleOutputs(ctx);
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      out.close();
    }

    protected int getBlockID(Text value, int numStripes) {
      String[] tokens = TasteHadoopUtils.splitPrefTokens(value.toString());

      int userID = TasteHadoopUtils.readID(tokens[TasteHadoopUtils.USER_ID_POS], true);
      int itemID = TasteHadoopUtils.readID(tokens[TasteHadoopUtils.ITEM_ID_POS], true);

      return ((TasteHadoopUtils.byteswap32(userID) % numStripes + numStripes) % numStripes) * numStripes +
             ((TasteHadoopUtils.byteswap32(itemID) % numStripes + numStripes) % numStripes);
    }

    @Override
    protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {

      String output = Integer.toString(getBlockID(value, numStripes));
      out.write(output, NullWritable.get(), value);
    }

  }
}
