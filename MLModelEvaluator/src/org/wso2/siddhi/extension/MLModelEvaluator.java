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

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang.ArrayUtils;
import org.apache.log4j.Logger;
import org.apache.mahout.classifier.mlp.MultilayerPerceptron;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.wso2.siddhi.core.config.SiddhiContext;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.event.in.InEvent;
import org.wso2.siddhi.core.event.in.InListEvent;
import org.wso2.siddhi.core.event.in.InStream;
import org.wso2.siddhi.core.executor.expression.ExpressionExecutor;
import org.wso2.siddhi.core.query.processor.transform.TransformProcessor;
import org.wso2.siddhi.core.util.parser.ExecutorParser;
import org.wso2.siddhi.query.api.definition.Attribute;
import org.wso2.siddhi.query.api.definition.StreamDefinition;
import org.wso2.siddhi.query.api.expression.Expression;
import org.wso2.siddhi.query.api.expression.Variable;
import org.wso2.siddhi.query.api.expression.constant.StringConstant;
import org.wso2.siddhi.query.api.extension.annotation.SiddhiExtension;

@SiddhiExtension(namespace = "mlearn", function = "predict")
public class MLModelEvaluator extends TransformProcessor {
	private Map<String, Integer> parameterPositions = new HashMap<String, Integer>();
	private List<String> features = new ArrayList<String>();
	private String modelPath;
	private StringConstant algo;
	private Vector precidtedResponse;
	private double[] dataRow;
	private Vector dataRowVector;
	private double[] intercept = { 1 };
	private Double[] result = new Double[1];
	private Vector LRrowVector;
	private int outputNodes;

	private enum Algorithm {
		NaiveBayes, LogisticRegression, MultilayerPerceptron
	};

	private Algorithm algorithm;
	private MultilayerPerceptron MLPmodel;
	private OnlineLogisticRegression LRmodel;
	private StandardNaiveBayesClassifier NBmodel;

	public MLModelEvaluator() {
	}

	/*
	 * Initialize event streams and models.
	 * (non-Javadoc)
	 * @see
	 * org.wso2.siddhi.core.query.processor.transform.TransformProcessor#init
	 * (org.wso2.siddhi.query.api.expression.Expression[], java.util.List,
	 * org.wso2.siddhi.query.api.definition.StreamDefinition,
	 * org.wso2.siddhi.query.api.definition.StreamDefinition, java.lang.String,
	 * org.wso2.siddhi.core.config.SiddhiContext)
	 */
	@Override
	protected void init(Expression[] parameters, List<ExpressionExecutor> expressionExecutors,
	                    StreamDefinition inStreamDefinition, StreamDefinition outStreamDefinition,
	                    String elementId, SiddhiContext siddhiContext) {
		Logger logger = Logger.getLogger(MLModelEvaluator.class);
		for (Expression parameter : parameters) {
			if (parameter instanceof Variable) {
				Variable var = (Variable) parameter;
				String attributeName = var.getAttributeName();
				parameterPositions.put(attributeName,
				                       inStreamDefinition.getAttributePosition(attributeName));
			}
		}
		// check whether the first and second arguments in siddhi query are
		// strings
		if (parameters[0] instanceof StringConstant && parameters[0] instanceof StringConstant) {
			// read the model path from the first argument of the siddhi query
			Expression expression = parameters[0];
			ExpressionExecutor executor =
					ExecutorParser.parseExpression(expression, null,
					                               elementId, false,
					                               siddhiContext);
			modelPath = (String) executor.execute(null);

			// read the algorithm name from the second argument of the siddhi
			// query
			expression = parameters[1];
			executor =
					ExecutorParser.parseExpression(expression, null, elementId, false,
					                               siddhiContext);
			algo = (StringConstant) executor.execute(null);

			// if a valid model file path is given
			if (isFilePath(modelPath)) {
				// initialize a model of the relevant algorithm type
				if (Algorithm.MultilayerPerceptron.toString().equalsIgnoreCase(algo.toString())) {
					MLPmodel = ModelInitializer.InitializeMultilayerPerceptron(modelPath);
					algorithm = Algorithm.MultilayerPerceptron;
					Matrix[] weights = MLPmodel.getWeightMatrices();
					outputNodes = weights[weights.length - 1].numRows();
				} else if (Algorithm.LogisticRegression.toString()
						.equalsIgnoreCase(algo.toString())) {
					algorithm = Algorithm.LogisticRegression;
					LRmodel = ModelInitializer.InitializeLogisticRegression(modelPath);
					LRrowVector = new RandomAccessSparseVector(LRmodel.numFeatures());
				} else if (Algorithm.NaiveBayes.toString().equalsIgnoreCase(algo.toString())) {
					algorithm = Algorithm.NaiveBayes;
					NBmodel = ModelInitializer.InitializeNaiveBayes(modelPath);
				} else {
					logger.debug("Invalid algorithm. Please provide a valid alorithm type for  \"" +
							modelPath + "\" model file.");
				}
				// add all the feature names to the features list
				for (String field : parameterPositions.keySet()) {
					features.add(field);
				}

				// initialize the output stream
				this.outStreamDefinition = new StreamDefinition().name("MLPredictedStream");
				this.outStreamDefinition.attribute("response", Attribute.Type.DOUBLE);
			} else {
				logger.debug("Invalid model file :" + modelPath);
			}
		} else {
			logger.debug("Invalid query. Please check the parameters!");
		}
		dataRow = new double[features.size()];
		dataRowVector = new RandomAccessSparseVector(features.size());
	}

	/*
	 * Apply the model for each instance in input-stream
	 * (non-Javadoc)
	 * @see org.wso2.siddhi.core.query.processor.transform.TransformProcessor#
	 * processEvent(org.wso2.siddhi.core.event.in.InEvent)
	 */
	@Override
	protected InStream processEvent(InEvent inEvent) {
		// create a vector from the input stream
		for (String feature : features) {
			dataRow[parameterPositions.get(feature)] =
					(double) inEvent.getData(parameterPositions.get(feature));
		}
		dataRowVector.assign(dataRow);

		// Apply the relevant model to the data vector and get the predicted
		// output
		switch (algorithm) {
			case MultilayerPerceptron:
				precidtedResponse = MLPmodel.getOutput(dataRowVector);
				// if there is only one output node
				if (outputNodes == 1) {
					// output is the rounded value of the resulting response
					result[0] = (double) Math.round(precidtedResponse.get(0));
				} else {
					// otherwise, output is the index that has the highest value
					result[0] = (double) precidtedResponse.maxValueIndex();
				}
				break;
			case LogisticRegression:
				LRrowVector.assign(ArrayUtils.addAll(intercept, dataRow));
				precidtedResponse = LRmodel.classifyFull(LRrowVector);
				result[0] = (double) precidtedResponse.maxValueIndex();
				break;
			case NaiveBayes:
				precidtedResponse = NBmodel.classifyFull(dataRowVector);
				result[0] = (double) precidtedResponse.maxValueIndex();
				break;
		}
		return new InEvent("MLPredictedStream", System.currentTimeMillis(), result);
	}

	/*
	 * (non-Javadoc)
	 * @see org.wso2.siddhi.core.query.processor.transform.TransformProcessor#
	 * processEvent(org.wso2.siddhi.core.event.in.InListEvent)
	 */
	@Override
	protected InStream processEvent(InListEvent inListEvent) {
		InListEvent transformedListEvent = new InListEvent();
		for (Event event : inListEvent.getEvents()) {
			if (event instanceof InEvent) {
				transformedListEvent.addEvent((Event) processEvent((InEvent) event));
			}
		}
		return transformedListEvent;
	}

	@Override
	protected Object[] currentState() {
		return new Object[] { parameterPositions };
	}

	/*
	 * Reset Lists and HashMaps
	 * (non-Javadoc)
	 * @see org.wso2.siddhi.core.query.processor.transform.TransformProcessor#
	 * restoreState(java.lang.Object[])
	 */
	@Override
	protected void restoreState(Object[] objects) {
		if (objects.length > 0 && objects[0] instanceof Map) {
			parameterPositions = (Map<String, Integer>) objects[0];
			features = new ArrayList<String>();
		}
	}

	/*
	 * Check whether the file path is valid
	 */
	protected boolean isFilePath(String path) {
		File file = new File(path);
		if (file.exists()) {
			return true;
		} else {
			return false;
		}
	}

	@Override
	public void destroy() {
	}
}