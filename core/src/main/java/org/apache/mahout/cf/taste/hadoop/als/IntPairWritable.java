package org.apache.mahout.cf.taste.hadoop.als;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;

import com.google.common.collect.ComparisonChain;

public class IntPairWritable implements WritableComparable<IntPairWritable> {

	private int userId;
	private int blockId;
	
	@Override
	public void readFields(DataInput arg0) throws IOException {
		userId = arg0.readInt();
		blockId = arg0.readInt();		
	}

	@Override
	public void write(DataOutput arg0) throws IOException {
		arg0.writeInt(userId);
		arg0.writeInt(blockId);
	}

	@Override
	public int compareTo(IntPairWritable o) {
		return ComparisonChain.start().compare(userId, o.userId)
		        .compare(blockId, o.blockId).result();		
	}

}
