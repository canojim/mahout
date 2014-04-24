package org.apache.mahout.cf.taste.hadoop.als;

import java.io.IOException;
import java.util.HashSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

import com.google.common.io.Closeables;

/**
 * This class is created to avoid OutOfMemoryError when loading all entries, while still maintain the advantage of HashMap
 * In our case, Vector is small, getHashMap(Vector keys)
 *
 */
public class UserRatingsHashMapCache {

	//TODO: Dynamic cache size
	//private final static int CACHE_SIZE = 3000000;

	private OpenIntObjectHashMap<Vector> cachedFeaturesForY;

	private Path[] cachedFiles;
	private LocalFileSystem localFs;
	private int startIndexForNotLoadedFiles = 0;
	private Configuration conf;
	
	public UserRatingsHashMapCache(Configuration conf, int numEntities) throws IOException {
		
		long freeMemory = Runtime.getRuntime().freeMemory();
		System.out.println("Available memory in bytes: " + freeMemory);
		
		int dynamicCacheSize = (int) (freeMemory * 0.8 / 20);
				
		this.conf = conf;		
		
		this.cachedFeaturesForY = numEntities < dynamicCacheSize
		        ? new OpenIntObjectHashMap<Vector>(numEntities) : new OpenIntObjectHashMap<Vector>(dynamicCacheSize); 
		        
				cachedFiles = HadoopUtil.getCachedFiles(conf);
				localFs = FileSystem.getLocal(conf);

				IntWritable rowIndex = new IntWritable();
				VectorWritable row = new VectorWritable();
				
				for (int i =0; i < cachedFiles.length; i++) {						
						
					Path cachedFile = cachedFiles[i];
					
					SequenceFile.Reader reader = null;
					try {
						reader = new SequenceFile.Reader(localFs, cachedFile, conf);
						while (reader.next(rowIndex, row)) {
							if (cachedFeaturesForY.size() >= dynamicCacheSize) {
								startIndexForNotLoadedFiles = i;
								System.out.println("Cache Full: numEntities: " + numEntities + " dynamicCacheSize: " + dynamicCacheSize);
								System.out.println("Available memory in bytes: " + Runtime.getRuntime().freeMemory());
								return;
							} else {
								cachedFeaturesForY.put(rowIndex.get(), row.get());
							}
						}
					} finally {
						Closeables.close(reader, true);
					}
					
				} // for				
				
	}
	
	public OpenIntObjectHashMap<Vector> getHashMap(Vector userRatings) throws IOException {
		OpenIntObjectHashMap<Vector> featureMatrix = new OpenIntObjectHashMap<Vector>();

		HashSet<Integer> userRatingsSet = new HashSet<Integer>();

		for (Element e : userRatings.nonZeroes()) {
			userRatingsSet.add(e.index());
			if (cachedFeaturesForY.containsKey(e.index())) {
				featureMatrix.put(e.index(), cachedFeaturesForY.get(e.index()));
			}
		}

		if (userRatingsSet.size() == featureMatrix.size()) {
			System.out.println("Feeling Lucky. All needed features are in memory");
			return featureMatrix;
		}
		
		IntWritable rowIndex = new IntWritable();
		VectorWritable row = new VectorWritable();


		long count = 0;
		
		for (int i=startIndexForNotLoadedFiles; i < cachedFiles.length; i++) {
		//for (Path cachedFile : cachedFiles) {
			Path cachedFile = cachedFiles[i];
					
			SequenceFile.Reader reader = null;
			try {
				reader = new SequenceFile.Reader(localFs, cachedFile, conf);
				while (reader.next(rowIndex, row)) {
					count++;
					if (userRatingsSet.contains(rowIndex.get())) {
						featureMatrix.put(rowIndex.get(), row.get());
					}
					
					if (userRatingsSet.size() == featureMatrix.size()) {
						System.out.println("Found all after searching " + count);
						return featureMatrix;
					}
				}
			} finally {
				Closeables.close(reader, true);
			}
		}

		return featureMatrix;
	}

}
