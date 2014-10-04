/*
 * Copyright (c) 2005-2014, WSO2 Inc. (http://www.wso2.org) All Rights Reserved.
 *
 * WSO2 Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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

	/*
	 * Initialize a LogisticRegression model object from a model file
	 */
	public static OnlineLogisticRegression InitializeLogisticRegression(String modelPath) {
		OnlineLogisticRegression LRmodel = null;
		FileInputStream fileInputStream = null;
		ObjectInputStream objectInputStream = null;
		double[][] modelWeights = null;
		LogisticRegressionModel LRmodelObject;
		try {
			// get the values for hyper-parameters from model file.
			fileInputStream = new FileInputStream(modelPath);
			objectInputStream = new ObjectInputStream(fileInputStream);
			LRmodelObject = (LogisticRegressionModel) objectInputStream.readObject();
			LRmodel =
					new OnlineLogisticRegression(LRmodelObject.getNumCategories(),
					                             LRmodelObject.getNumFeatures(), new L2(1));
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
			logger.error("Failed to create a Logistic Regression model from the file \"" +
					modelPath + "\"", e);
		} finally {
			try {
				fileInputStream.close();
				objectInputStream.close();
			} catch (IOException e) {
				logger.error("Failed to close the model input stream!", e);
			}
		}
		logger.info("Logistic Regression model execution plan successfully intialized for \"" +
				modelPath + "\" model file.");
		return LRmodel;
	}

	/*
	 * Initialize a MultilayerPerceptron model object from a model file
	 */
	public static MultilayerPerceptron InitializeMultilayerPerceptron(String modelPath) {
		MultilayerPerceptron MLPmodel = new MultilayerPerceptron(modelPath);
		logger.info("Multilayer Perceptron model execution plan successfully intialized for \"" +
				modelPath + "\" model file.");
		return MLPmodel;
	}

	/*
	 * Initialize a NaiveBayesClassifier model object from a model file
	 */
	public static StandardNaiveBayesClassifier InitializeNaiveBayes(String modelPath) {
		Configuration configuration = new Configuration();
		NaiveBayesModel naiveBayesModel = null;
		StandardNaiveBayesClassifier NBmodel = null;
		try {
			naiveBayesModel = NaiveBayesModel.materialize(new Path(modelPath), configuration);
			NBmodel = new StandardNaiveBayesClassifier(naiveBayesModel);
		} catch (Exception e) {
			logger.error("Failed to create a Naive Bayes model from the file \"" + modelPath + "\"");
		}
		logger.info("Naive Bayes model execution plan successfully intialized for  \"" + modelPath +
				"\" model file.");
		return NBmodel;
	}
}
