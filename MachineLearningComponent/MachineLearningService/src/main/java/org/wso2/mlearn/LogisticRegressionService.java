package org.wso2.mlearn;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.log4j.Logger;
import org.apache.mahout.classifier.sgd.L2;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

public class LogisticRegressionService implements Callable<String>{
	private String inputPath;
	private String outputPath;
	private FileSystem hdfs;

	private int numCategories;
	private int numFeatures;
	private List <Vector> trainingFeatureSet = new ArrayList <Vector>();
	private List <Integer> trainingResponseSet =new ArrayList <Integer> ();
	private List <Integer> trainingIndexSet = new ArrayList <Integer> ();
	private List <Vector> testFeatureSet = new ArrayList <Vector>();
	private List <Integer> testResponseSet =new ArrayList <Integer> ();
	private List <Integer> testIndexSet = new ArrayList <Integer> ();
	private Map <String,Double> optionsSet = new HashMap <String,Double> ();
	Logger logger;

	public LogisticRegressionService(String inputPath,String outputPath, Options [] options, FileSystem hdfs)
			throws Exception {
		this.inputPath=inputPath;
		this.outputPath=outputPath;
		this.hdfs=hdfs;		
		for (int i=0 ; i<options.length ; i++){
			this.optionsSet.put(options[i].getoption(),Double.parseDouble(options[i].getvalue()));
		}
		logger = Logger.getLogger(LogisticRegressionService.class);
	}

	@Override
	public String call() throws Exception {
		resetLists();
		String jobId = String.valueOf(Thread.currentThread().getId());
		MachineLearningService.currentJobs.put(jobId,"active");
		String modelPath = outputPath + "/LogisticRegressionModel";
		try {
	        vectorizeInputData(inputPath,hdfs);
        } catch (Exception e) {
        	logger.error("Failed to create data vectors from the input file: "+inputPath , e);
        }
		logger.info("Training set size: "+trainingIndexSet.size());
		logger.info("Test set size: "+testIndexSet.size());
		logger.info("Number of Features: "+ (numFeatures));
		logger.info("Number of Response Categories: " + numCategories);

		//Train
		Random random = RandomUtils.getRandom();
		OnlineLogisticRegression model = new  OnlineLogisticRegression(numCategories, numFeatures+1, new L2(1));
		model.learningRate(optionsSet.get("learningRate"));
		model.lambda(optionsSet.get("lambda"));
		model.alpha(1);
		model.stepOffset(1000);
		model.decayExponent(optionsSet.get("decay"));
		for(int pass=0 ; pass<100 ; pass++){
			Collections.shuffle(trainingIndexSet,random);
			for(int k : trainingIndexSet){
				model.train(trainingResponseSet.get(k), trainingFeatureSet.get(k));
			}
		}
		
		// write the trained model to a file
  	    double [][] weights = new double[model.getBeta().numRows()][model.getBeta().numCols()];
      	for(int i=0 ; i<model.getBeta().numRows() ; i++){
      		for(int j=0 ; j<model.getBeta().numCols() ; j++){	  
      			weights[i][j]=model.getBeta().get(i, j);
      		}
      	}
      	LogisticRegressionModel modelObject = new LogisticRegressionModel(model.numFeatures(),model.numCategories(),weights, model.currentLearningRate(), model.getLambda(), 1, model.getStep(), optionsSet.get("decay"));
      	FileOutputStream fileOutStream = new FileOutputStream(modelPath);
      	ObjectOutputStream objectOutStream = new ObjectOutputStream(fileOutStream);
      	objectOutStream.writeObject(modelObject); 
      	objectOutStream.close();  
      	fileOutStream.close();

		// Validate
		int total=0;
		int correct=0;
		for(int i:testIndexSet){
			Vector precidtedResponse = model.classifyFull(testFeatureSet.get(i));
			if(precidtedResponse.maxValueIndex()==testResponseSet.get(i)){
				correct++;
			}
			total++;
		}
		model.close();
		logger.info("Successfully trained a Logistic Regression model for the data file : \'"+inputPath+"\'");
		logger.info("Model Accuracy = " +correct*1.0/total*100 +" %");
		logger.info("Model is saved to : \'"+modelPath+"\'");		
		return jobId;
	}


	private void vectorizeInputData(String inputData, FileSystem fileSystem) throws Exception {
		// find total number of records
		String line;
		FSDataInputStream lineNumberStream = fileSystem.open(new Path(inputData));
		BufferedReader lineNumberReader = new BufferedReader(new InputStreamReader(lineNumberStream));
		int size = -1;
		while (lineNumberReader.readLine() != null) {
			size++;
		}
		lineNumberStream.close();
		lineNumberReader.close();

		//create the data vectors
		FSDataInputStream dataStream = fileSystem.open(new Path(inputData));
		BufferedReader dataReader = new BufferedReader(new InputStreamReader(dataStream));
		List <String> classes = new ArrayList<String>();

		/*ignore the first line (headers)
		 *  Set the number of features in the data
		 */
		numFeatures = dataReader.readLine().split(",").length-1;
		double[] dataRaw = new double[numFeatures+1];
		Vector rawVector;
		int index=0;
		String[] values;
		// create a training data vector
		for(int lineNo=0 ; lineNo<(int)(size*0.7) ; lineNo++){
			line = dataReader.readLine();
			values = line.split(",");
			rawVector = new RandomAccessSparseVector(numFeatures+1);
			// find distinguished categories
			if(!classes.contains(values[0])){
				classes.add(values[0]);
			}
			// set the intercept term to 1
			dataRaw[0]=1;

			for (int i = 1; i <=numFeatures; i++) {
				dataRaw[i] = Double.parseDouble(values[i]);
			}
			rawVector.assign(dataRaw);
			trainingResponseSet.add(Integer.parseInt(values[0]));
			trainingFeatureSet.add(rawVector);
			trainingIndexSet.add(index++);
		}
		numCategories=classes.size();

		//Create the validate data vector
		index=0;
		for(int lineNo=(int)(size*0.7) ; lineNo<size ; lineNo++){
			line = dataReader.readLine();
			values = line.split(",");
			rawVector = new RandomAccessSparseVector(numFeatures+1);
			// set the intercept term to 1
			dataRaw[0]=1;

			for (int i = 1; i <= numFeatures; i++) {
				dataRaw[i] = Double.parseDouble(values[i]);
			}
			rawVector.assign(dataRaw);
			testResponseSet.add(Integer.parseInt(values[0]));
			testFeatureSet.add(rawVector);
			testIndexSet.add(index++);
		}
		dataReader.close();
		dataStream.close();
	}
	
    private void resetLists(){
    	trainingFeatureSet.clear();
    	trainingResponseSet.clear();
        trainingIndexSet.clear();
        testFeatureSet.clear();
        testResponseSet.clear();
        testIndexSet.clear();
    }
}