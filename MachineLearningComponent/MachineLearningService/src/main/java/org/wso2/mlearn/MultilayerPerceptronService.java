package org.wso2.mlearn;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
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
import org.apache.mahout.classifier.mlp.MultilayerPerceptron;
import org.apache.mahout.classifier.mlp.NeuralNetwork.TrainingMethod;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

public class MultilayerPerceptronService implements Callable<String>{
	private String inputPath;
	private String modelPath;
	private FileSystem hdfs;
	private int numCategories;
	private int numFeatures;
	private List <Vector> trainingDataSet = new ArrayList <Vector>();
	private List <Integer> trainingIndexSet = new ArrayList <Integer> ();
	private List <Vector> testFeatureSet = new ArrayList <Vector>();
	private List <Integer> testResponseSet =new ArrayList <Integer> ();
	private List <Integer> testIndexSet = new ArrayList <Integer> ();
	private Map <String,Double> optionsSet = new HashMap <String,Double> ();
	Logger logger;

	public MultilayerPerceptronService(String inputPath,String outputPath, Options [] options, FileSystem hdfs)	throws Exception {
		this.inputPath=inputPath;
		this.modelPath=outputPath+ "/MLPModel";
		this.hdfs=hdfs;
		for (Options option : options) {
			this.optionsSet.put(option.getoption(),Double.parseDouble(option.getvalue()));
		}
		logger = Logger.getLogger(MultilayerPerceptronService.class);
	}

	@Override
	public String call() {
		resetLists();
		String jobId = String.valueOf(Thread.currentThread().getId());
		MachineLearningService.currentJobs.put(jobId,"active");
		try {
			vectorizeInputdata(inputPath,hdfs);
		} catch (Exception e) {
			logger.error("Failed to create data vectors from the input file: "+inputPath , e);
		}
		logger.info("Training set size: "+trainingIndexSet.size());
		logger.info("Test set size :"+testIndexSet.size());
		logger.info("Number of Features: "+ numFeatures);
		logger.info("Number of Response Categories: " + numCategories);

		//Train
		MultilayerPerceptron model = new MultilayerPerceptron();
		model.setLearningRate(optionsSet.get("learningRate"));
		model.addLayer(numFeatures, false, "Sigmoid");
		model.addLayer(numFeatures/3, false, "Sigmoid");
		if(numCategories==2){
			model.addLayer(1, true, "Sigmoid");
		}else{
			model.addLayer(numCategories, true, "Sigmoid");
		}
		model.setTrainingMethod(TrainingMethod.GRADIENT_DESCENT);
		Random random = RandomUtils.getRandom();
		for(int pass=0 ; pass<100 ; pass++){
			Collections.shuffle(trainingIndexSet,random);
			for(int index : trainingIndexSet){
				model.trainOnline(trainingDataSet.get(index));
			}
		}
		model.setModelPath(modelPath);
		try {
			model.writeModelToFile();
		} catch (IOException e) {
			logger.error("Failed to write the model to the file \'"+ modelPath+"\'" ,e);
		} finally{
			model.close();
		}

		//Validate
		int total=0;
		int correct=0;
		Vector precidtedResponse;
		for(int index : testIndexSet){
			precidtedResponse = model.getOutput(testFeatureSet.get(index));
			if(numCategories==2){
				if(Math.round(precidtedResponse.get(0))==testResponseSet.get(index)){
					correct++;
				}
			}else{
				if(precidtedResponse.maxValueIndex()==testResponseSet.get(index)){
					correct++;
				}
			}
			total++;
		}
		logger.info("Successfully trained a Multilayer Perceptron model for the data file : \'"+inputPath+"\'");
		logger.info("Model Accuracy: " + correct*1.0/total*100+" %");
		logger.info("Model is saved to : \'"+modelPath+"\'");
		return jobId;
	}

	private void vectorizeInputdata(String trainData, FileSystem fileSystem) throws Exception {
		FSDataInputStream dataStream = fileSystem.open(new Path(trainData));
		BufferedReader dataReader = new BufferedReader(new InputStreamReader(dataStream));
		List <String> classes = new ArrayList<String>();
		String line;
		int size=0;

		// find distinguish response classes and the number of records
		BufferedReader linesReader = new BufferedReader(new InputStreamReader(fileSystem.open(new Path(trainData))));
		linesReader.readLine();
		while ((line=linesReader.readLine()) != null){
			String response = line.split(",")[0];
			if(!classes.contains(response)){
				classes.add(response);
			}
			size++;
		}
		numCategories = classes.size();
		linesReader.close();

		// create the training data vector
		numFeatures = dataReader.readLine().split(",").length-1;
		double[] dataRaw;
		Vector dataRawVector;
		int index=0;
		String[] values;

		if(numCategories==2){
			dataRaw = new double[numFeatures+1];
			for(int lineNo=0 ; lineNo<(int)(size*0.7) ; lineNo++){
				line = dataReader.readLine();
				values = line.split(",");
				//set the features
				for (int i = 1; i <= numFeatures; i++) {
					dataRaw[i-1] = Double.parseDouble(values[i]);
				}
				dataRawVector = new RandomAccessSparseVector(numFeatures+1);
				//set the response
				dataRaw[numFeatures] = Integer.parseInt(values[0]);
				dataRawVector.assign(dataRaw);
				trainingDataSet.add(dataRawVector);
				trainingIndexSet.add(index++);
			}
		}else{
			//set the training set
			dataRaw = new double[numFeatures+numCategories];
			for(int lineNo=0 ; lineNo<(int)(size*0.7) ; lineNo++){
				line = dataReader.readLine();
				values = line.split(",");

				//set the features
				for (int i = 1; i <= numFeatures; i++) {
					dataRaw[i-1] = Double.parseDouble(values[i]);
				}
				//set the responses
				dataRawVector = new RandomAccessSparseVector(numFeatures+numCategories);
				for (int j = 0; j < numCategories; j++) {
					if(Integer.parseInt(values[0])==j){
						dataRaw[numFeatures+j] = 1;
					}
					else{
						dataRaw[numFeatures+j] = 0;
					}
				}
				dataRawVector.assign(dataRaw);
				trainingDataSet.add(dataRawVector);
				trainingIndexSet.add(index++);
			}
		}

		//Set the validate set
		index=0;
		dataRaw = new double[numFeatures];
		for(int lineNo=(int)(size*0.7) ; lineNo<size ; lineNo++){
			line = dataReader.readLine();
			values = line.split(",");
			dataRawVector = new RandomAccessSparseVector(numFeatures);
			for (int i = 1; i <= numFeatures; i++) {
				dataRaw[i-1] = Double.parseDouble(values[i]);
			}
			dataRawVector.assign(dataRaw);
			testResponseSet.add(Integer.parseInt(values[0]));
			testFeatureSet.add(dataRawVector);
			testIndexSet.add(index++);
		}
		dataReader.close();
		dataStream.close();
	}

	private void resetLists(){
		trainingDataSet.clear();
		trainingIndexSet.clear();
		testFeatureSet.clear();
		testResponseSet.clear();
		testIndexSet.clear();
	}
}