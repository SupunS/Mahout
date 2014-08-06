package org.wso2.mlearn;

import java.util.HashMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import javax.ws.rs.Consumes;
import javax.ws.rs.POST;
import javax.ws.rs.Produces;
import javax.ws.rs.core.Response;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

@javax.ws.rs.Path("/")
public class MachineLearningService {
	
	public static HashMap<String,String> currentJobs = new HashMap<String,String> ();
	
	@POST
	@Consumes("application/json")
	@Produces("application/json")
	public Response mahoutMachineLearningService(Parameters parameters) throws Exception {
		Configuration configuration = new Configuration();
		FileSystem hdfs = FileSystem.get(configuration);
		configuration.addResource(new Path(hdfs.getWorkingDirectory() +
				"/repository/conf/advanced/hive-site.xml"));
		hdfs.setConf(configuration);
		System.out.println(parameters);

		//String hdfsPath = FileSystem.getDefaultUri(configuration).toString();
		String input = parameters.getinPath();
		String output = parameters.getoutPath();
		String model = parameters.getalgorithm();
		Options[] options=parameters.getoptions();
		String message = null;

		ExecutorService executor = Executors.newCachedThreadPool();
		Future<String> future;
		
		if (model.equalsIgnoreCase("Naive Bayes")) {			
			Callable<String> service=new NaiveBayesService(input, output, options, configuration, hdfs);
			future=executor.submit(service);
			String jobId = future.get();
			message = "{\"id\" : \""+jobId+"\" , \"status\" : \""+currentJobs.get(jobId)+"\"}";
		} else if(model.equalsIgnoreCase("Logistic Regression")) {
			Callable<String> service=new LogisticRegressionService(input, output, options, hdfs);
			future=executor.submit(service);
			String jobId = future.get();
			message = "{\"id\" : \""+jobId+"\" , \"status\" : \""+currentJobs.get(jobId)+"\"}";
		}else if (model.equalsIgnoreCase("Multilayer Perceptron")) {
			Callable<String> service=new MultilayerPerceptronService(input, output, options, hdfs);
			future=executor.submit(service);
			String jobId = future.get();
			message = "{\"id\" : \""+jobId+"\" , \"status\" : \""+currentJobs.get(jobId)+"\"}";
		}else {
			System.out.println("Invalid model type. Please give a valid model.");
			message = "{\"status\" : \"failed\"}";
		}
		return Response.ok(message, "application/json").build();
	}
}