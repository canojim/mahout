package org.apache.mahout.cf.taste.hadoop.als;

import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import org.apache.hadoop.mapreduce.Job;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class JobManager {

	private static final Logger log = LoggerFactory
			.getLogger(JobManager.class);
	
	private static final int MAX_JOBS = 10;
	private static final int SLEEP_INTERVAL = 10000;
	public static final String QUEUE_NAME = "mapred.job.queue.name";
	
	private LinkedList<Job> queuedJobs = new LinkedList<Job>();
	private LinkedList<Job> runningJobs = new LinkedList<Job>();
	
	private String queueName = "default";
	
	public void addJob(Job job) {
		queuedJobs.add(job);
	}
	
	public void setQueueName(String q) {
		this.queueName = q;
	}
	
	public boolean waitForCompletion() {
		
		try {
			
			while ((queuedJobs.size() > 0) || runningJobs.size() > 0) {
				
				while ((queuedJobs.size() > 0) && (runningJobs.size() < MAX_JOBS)) {
					Job job = queuedJobs.pop();
					log.info("Submitting " + job.getJobName());
					
					job.getConfiguration().set(QUEUE_NAME, queueName);
					job.submit();
	
					runningJobs.add(job);
				}
	
				Thread.sleep(SLEEP_INTERVAL);
				
				for (Iterator<Job> iterator = runningJobs.iterator(); iterator.hasNext(); ) {
					Job job = iterator.next();
					
					if (job.isComplete()) {
						if (job.isSuccessful()) {
							log.info("Job success: " + job.getJobID().toString() + " " + job.getJobName());
							iterator.remove();
						} else {
							String msg = "Job fail: " + job.getJobID().toString() + " " + job.getJobName(); 
							log.info(msg);
							iterator.remove();							
							return false;
						}											
					}
				}
			}		
		} catch (Exception e) {
			log.error("waitForCompletion", e);
			throw new IllegalStateException(e);			
		} finally {
			cleanup();			
		}
		
		return false;
	}
	
	private void cleanup() {
		for (Iterator<Job> iterator = runningJobs.iterator(); iterator.hasNext(); ) {
			Job job = iterator.next();
			try {
				job.killJob();
				iterator.remove();
			} catch (IOException e) {
				log.info("Cleanup exception: " + job.getJobID().toString() + " " + job.getJobName() + " " + e.toString());
			}
		}
	}

}
