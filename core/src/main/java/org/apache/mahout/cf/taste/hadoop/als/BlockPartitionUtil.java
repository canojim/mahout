package org.apache.mahout.cf.taste.hadoop.als;

import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;

public class BlockPartitionUtil {
	public static int getBlockID(int userOrItemID, int numBlocks) {
		return (TasteHadoopUtils.byteswap32(userOrItemID) % numBlocks + numBlocks) % numBlocks;
	}
}
