package org.wso2.mahout.algorithms.neuralnet;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.wso2.mahout.algorithms.Supervised;

/*
 * Responses must be re-coded to integers starting with zero.
 */
public class MultilayerPerceptron implements Supervised {
	
	private org.apache.mahout.classifier.mlp.MultilayerPerceptron model;
	private Logger logger;
	private int numFeatures;
	private int numCategories;

	/*
	 * Create a Multilayer Perceptron model with an input layer and an output layer,
	 * and no hidden layers.
	 */
	public MultilayerPerceptron (int numCategories, int numFeatures){
		logger = Logger.getLogger(MultilayerPerceptron.class);
		this.numFeatures=numFeatures;
		this.numCategories=numCategories;
		model = new org.apache.mahout.classifier.mlp.MultilayerPerceptron();
		//add the input nodes
		model.addLayer(numFeatures, false, "Sigmoid");		
		//add the output node(s)
		if(numCategories==2){
			model.addLayer(1, true, "Sigmoid");
		}else{
			model.addLayer(numCategories, true, "Sigmoid");
		}
	}
	
	
	/*
	 * Create a Multilayer Perceptron Model object using an existing model file.
	 */
	public MultilayerPerceptron (String path){
		logger = Logger.getLogger(MultilayerPerceptron.class);
		model = new org.apache.mahout.classifier.mlp.MultilayerPerceptron(path);
		logger.info("Multilayer Perceptron model successfully loaded from the file "+ path);
	 }

	
	/*
	 * Set the values of hyper-parameters of the model.
	 */
	public void setParameters(double learningRate){
		model.setLearningRate(learningRate);
	}
	
	
	/*
	 * Takes lists of responses and features as inputs, add the intercept term to the feature set
	 * and trains a  Multilayer Perceptron model. A particular row of responseSet should contain the response 
	 * corresponds to the same row of the featureSet.
	 */
    public void train(List<Integer> indexSet, List<Integer> responseSet, List<Vector> featureSet, int passes) {
		Random random = RandomUtils.getRandom();
		double[] dataRow;
		Vector dataRowVector;
		if(numCategories==2){
			dataRow = new double[numFeatures+1];
			for(int pass=0 ; pass<passes ; pass++){
				//shuffle data
				Collections.shuffle(indexSet,random);
    			for(int index : indexSet){
    				//take a set of features and populate a temporary array
    				for (int i = 0; i < numFeatures; i++) {
    					dataRow[i] = featureSet.get(index).get(i);
    				}
    				//append the response to end of the array
    				dataRow[numFeatures] = responseSet.get(index);				
    				//create a training instance (i.e. a vector) using the features and the response
    				dataRowVector = new RandomAccessSparseVector(numFeatures+1);
    				dataRowVector.assign(dataRow);
    				//train the model using the training instance
    				model.trainOnline(dataRowVector);
    			}
			}
		}else{
			//set the training set
			dataRow = new double[numFeatures+numCategories];
			for(int pass=0 ; pass<passes ; pass++){
				Collections.shuffle(indexSet,random);
    			for(int index : indexSet){
    				//take a set of features and populate a temporary array
    				for (int i = 0; i < numFeatures; i++) {
    					dataRow[i] = featureSet.get(index).get(i);
    				}
    				//Binarize the response class and append it to the end of the array
    				for (int j = 0; j < numCategories; j++) {
    					if(responseSet.get(index)==j){
    						dataRow[numFeatures+j] = 1;
    					}
    					else{
    						dataRow[numFeatures+j] = 0;
    					}
    				}
    				//create a training instance (i.e. a vector) using the features and the response
    				dataRowVector = new RandomAccessSparseVector(numFeatures+numCategories);
    				dataRowVector.assign(dataRow);
    				//train the model using the training instance
    				model.trainOnline(dataRowVector);
    			}
			}
		}
	 	logger.info("Multilayer Perceptron model successfully trained!");
    }


	/*
	 * Evaluates the model with known data. Output is predicted using the data in each row of the 
	 * featureSet and is compared with the actual response, which is in responseSet
	 * A particular row of responseSet should contain the response corresponds to the features that
	 * are in the same row of the featureSet.
	 */
    public void test(List<Integer> responseSet, List<Vector> featureSet) {
		int total=0;
		int correct=0;
		Vector precidtedResponse;
		for(int row=0 ; row<featureSet.size() ; row++){
			//take a set of features, send it to the model and get the output
			precidtedResponse = model.getOutput(featureSet.get(row));
			//compare the model output with the actual response
			if(numCategories==2){
				//if there are only two categories, predicted output is the rounded-value of the model's output
				if(Math.round(precidtedResponse.get(0))==responseSet.get(row)){
					correct++;
				}
			}else{
				//if there are more than two categories, predicted output is the index  with the highest value, of the model's output vector
				if(precidtedResponse.maxValueIndex()==responseSet.get(row)){
					correct++;
				}
			}
			total++;
		}
		logger.info("Model Accuracy: " + correct*1.0/total*100+" %");
    }

	
	/*
	 * write the model to a file
	 */
    public void export(String exportPath) {
		model.setModelPath(exportPath);
		try {
			model.writeModelToFile();
		} catch (IOException e) {
			logger.error("Failed to export the model to the path "+exportPath);
		} finally{
			model.close();
		}
		logger.info("Multilayer Perceptron model successfully exported to "+exportPath);
    }
}
