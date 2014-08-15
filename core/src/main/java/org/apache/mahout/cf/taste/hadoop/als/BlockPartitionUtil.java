package org.apache.mahout.cf.taste.hadoop.als;

import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;

public class BlockPartitionUtil {
	public static int getBlockID(int userOrItemID, int numBlocks) {
		return (TasteHadoopUtils.byteswap32(userOrItemID) % numBlocks + numBlocks) % numBlocks;
	}
	
	public static void main(String argv[]) {
		System.out.println(getBlockID(121328523, 10));
		System.out.println(getBlockID(552638733, 10));
		System.out.println(getBlockID(453961070, 10));
	}
}
