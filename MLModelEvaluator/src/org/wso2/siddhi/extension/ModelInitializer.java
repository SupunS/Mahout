package org.wso2.siddhi.extension;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.log4j.Logger;
import org.apache.mahout.classifier.mlp.MultilayerPerceptron;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.classifier.sgd.L2;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.wso2.mlearn.LogisticRegressionModel;

public class ModelInitializer {
	static Logger logger = Logger.getLogger(MLModelEvaluator.class);

	public static OnlineLogisticRegression InitializeLogisticRegression(String modelPath){
		OnlineLogisticRegression LRmodel = null;
		FileInputStream fileInputStream = null;
		ObjectInputStream objectInputStream = null;
		double[][] modelWeights = null;
		LogisticRegressionModel LRmodelObject;
		try {
			fileInputStream = new FileInputStream(modelPath);
			objectInputStream = new ObjectInputStream(fileInputStream);
			LRmodelObject = (LogisticRegressionModel) objectInputStream.readObject();
			LRmodel = new OnlineLogisticRegression(LRmodelObject.getNumCategories(), LRmodelObject.getNumFeatures(), new L2(1));
			LRmodel.learningRate(LRmodelObject.getLearningRate());
			LRmodel.lambda(LRmodelObject.getLambda());
			LRmodel.alpha(LRmodelObject.getAlpha());
			LRmodel.stepOffset(LRmodelObject.getStepOffset());
			LRmodel.decayExponent(LRmodelObject.getDecayExponent());
			modelWeights = LRmodelObject.getWeights();
			fileInputStream.close();
			objectInputStream.close();
			for (int i = 0; i < modelWeights.length; i++) {
				for (int j = 0; j < modelWeights[0].length; j++) {
					LRmodel.setBeta(i, j, modelWeights[i][j]);
				}
			}
		} catch (Exception e) {
			logger.debug("Failed to create a Logistic Regression model from the file \"" + modelPath+"\"");
		} finally{
			try {
				fileInputStream.close();
				objectInputStream.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		logger.debug("Logistic Regression model execution plan successfully intialized for \""+ modelPath+"\" model file.");
		return LRmodel;
	}

	
	public static MultilayerPerceptron InitializeMultilayerPerceptron(String modelPath){
		MultilayerPerceptron MLPmodel = new MultilayerPerceptron(modelPath);
		logger.debug("Multilayer Perceptron model execution plan successfully intialized for \""+ modelPath+"\" model file.");
		return MLPmodel;
	}

	
	public static StandardNaiveBayesClassifier InitializeNaiveBayes(String modelPath){
		Configuration configuration = new Configuration();
		NaiveBayesModel naiveBayesModel = null;
		StandardNaiveBayesClassifier NBmodel = null;
		try {
			naiveBayesModel = NaiveBayesModel.materialize(new Path(modelPath), configuration);
			NBmodel = new StandardNaiveBayesClassifier(naiveBayesModel);
		} catch (Exception e) {
			logger.debug("Failed to create a Naive Bayes model from the file \"" + modelPath+"\"");
		}
		logger.debug("Naive Bayes model execution plan successfully intialized for  \""+ modelPath+"\" model file.");
		return NBmodel;
	}
}
