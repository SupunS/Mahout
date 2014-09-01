package org.wso2.mahout.algorithms.regression;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;
import org.apache.mahout.classifier.sgd.L2;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.classifier.sgd.PriorFunction;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.wso2.mahout.algorithms.Supervised;

public class LogisticRegression implements Supervised {
	private OnlineLogisticRegression  model;
	private double decay;
	private Logger logger;
	
	/*
	 * Initialize a Regression Model object with a given number of features, output categories, and a prior function.
	 */
	public LogisticRegression(int numCategories, int numFeatures, PriorFunction prior){
		logger = Logger.getLogger(LogisticRegression.class);
		model = new OnlineLogisticRegression(numCategories, numFeatures+1, prior);
	}
	
	
	/*
	 * Create a Regression Model object using an existing model file.
	 */
	public LogisticRegression(String path) throws IOException{
		logger = Logger.getLogger(LogisticRegression.class);
		FileInputStream fileInputStream = null;
		LogisticRegressionModel LRmodelObject = null;
		ObjectInputStream objectInputStream = null;
		double [][] modelWeights = null;
        try {
        	//read the model from file, and initialize an Online Logistic Regression model using its properties
	        fileInputStream = new FileInputStream(path);
	        objectInputStream = new ObjectInputStream(fileInputStream);
			LRmodelObject = (LogisticRegressionModel) objectInputStream.readObject();			
			model = new OnlineLogisticRegression(LRmodelObject.getNumCategories(), LRmodelObject.getNumFeatures(), new L2(1));
			//set the hyper-parameters
			model.learningRate(LRmodelObject.getLearningRate());
			model.lambda(LRmodelObject.getLambda());
			model.alpha(LRmodelObject.getAlpha());
			model.stepOffset(LRmodelObject.getStepOffset());
			model.decayExponent(LRmodelObject.getDecayExponent());
			modelWeights=LRmodelObject.getWeights();			
        } catch (FileNotFoundException e) {
        	logger.error("Invalid model file: "+path);
        	logger.error("Failed to create a model!");
        } catch (ClassNotFoundException e) {
        	logger.error("Invalid Object type. Failed to create a model.");
        }finally{
        	fileInputStream.close();
			objectInputStream.close();
        }
		for(int row=0 ; row<modelWeights.length ; row++){
			for(int column=0 ; column<modelWeights[0].length ; column++){
				model.setBeta(row,column, modelWeights[row][column]);
			}
		}
		logger.info("Logistic Regression model successfully loaded from the file: "+ path);
	}

	
	/*
	 * Assign the given values to the hyper-parameters of the model
	 */
	public void setParameters(double learningRate, double lambda, double alpha, double decay, int stepOffset){
		model.learningRate(learningRate);
		model.lambda(lambda);
		model.alpha(alpha);
		model.decayExponent(decay);
		model.stepOffset(stepOffset);
		this.decay=decay;
	}

	
	/*
	 * Takes lists of responses and features as inputs, add the intercept term to the feature set
	 * and trains the model. A particular row of responseSet should contain the response 
	 * corresponds to the same row of the featureSet.
	 */
	public void train(List <Integer> indexSet, List <Integer> responseSet , List <Vector> featureSet, int passes){
		Random random = RandomUtils.getRandom();
		double[] featureArray = new double[featureSet.get(0).size()+1];
		Vector featureVector = new DenseVector(featureSet.get(0).size()+1);
		//set intercept to 1
		featureArray[0]=1;
 	    for(int pass=0 ; pass<passes ; pass++){
 			//shuffle the input data
   			Collections.shuffle(indexSet,random);
       		for(int index : indexSet){
       			//create a data vector from the input data, including the intercept term
       			for(int feature=0 ; feature<featureSet.get(index).size() ; feature++){
       				featureArray[feature+1]=featureSet.get(index).get(feature);
       			}
       			featureVector.assign(featureArray);
       			//train the model using the above data vector
       			model.train(responseSet.get(index), featureVector);
       		}
       	}
 	   logger.info("Logistic Regression model successfully trained!");
	}

	
	/*
	 * writes the model to a file.
	 * Since OnlineLogisticModel object is not serializable, a custom serializable object that can store the details of the trained model is created,
	 * and then that custom object is written to a binary file. 
	 */
	public void export(String exportPath){
		//create a weights matrix
  	    double [][] weights = new double[model.getBeta().numRows()][model.getBeta().numCols()];
  	    //populate it with model's weights
      	for(int row=0 ; row<model.getBeta().numRows() ; row++){
      		for(int column=0 ; column<model.getBeta().numCols() ; column++){	  
      			weights[row][column]=model.getBeta().get(row, column);
      		}
      	}
      	//create a serializable LogisticRegressionModel object and assign the properties of the previously trained Online Logistic Regression model
      	LogisticRegressionModel modelObject = new LogisticRegressionModel(model.numFeatures(),model.numCategories(),weights, model.currentLearningRate(), model.getLambda(), 1, model.getStep(), decay);
      	FileOutputStream fileOutStream;
        try {
        	//write the model to the file
	        fileOutStream = new FileOutputStream(exportPath);
	      	ObjectOutputStream objectOutStream = new ObjectOutputStream(fileOutStream);
	      	objectOutStream.writeObject(modelObject); 
	      	objectOutStream.close();  
	      	fileOutStream.close();
        } catch (Exception e) {
        	logger.error("Failed to export the model to the path: "+exportPath);
        }
        logger.info("Logistic Regression model successfully exported to file: "+exportPath);
	}
	
	
	/*
	 * Evaluates the model with known data. Output is predicted using the data in each row of the 
	 * featureSet and is compared with the actual response, which is in responseSet
	 * A particular row of responseSet should contain the response corresponds to the features that
	 * are in the same row of the featureSet.
	 */
	public void test(List <Integer> responseSet , List <Vector> featureSet){
		int total=0;
		int correct=0;
		//A temporary vector and an array to hold a set of input features 
		double[] featureArray = new double[featureSet.get(0).size()+1];
		Vector featureVector = new DenseVector(featureSet.get(0).size()+1);
		//set intercept to 1
		featureArray[0]=1;
		for(int row=0 ; row<featureSet.size() ; row++){
			//populate the vector with a set of features, including the intercept term
			for(int column=0 ; column<featureSet.get(row).size() ; column++){
   				featureArray[column+1]=featureSet.get(row).get(column);
   			}
   			featureVector.assign(featureArray);
   			//predict the response using the above data vector
			Vector precidtedResponse = model.classifyFull(featureVector);
			//compare the predicted response with the actual response
			if(precidtedResponse.maxValueIndex()==responseSet.get(row)){
				correct++;
			}
			total++;
		}
		model.close();
		logger.info("Model Accuracy = " +correct*1.0/total*100 +" %");
	}
}
