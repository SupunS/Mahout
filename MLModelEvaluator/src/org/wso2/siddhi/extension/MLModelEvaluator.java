package org.wso2.siddhi.extension;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.log4j.Logger;
import org.apache.mahout.classifier.mlp.MultilayerPerceptron;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
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
	private Double [] result = new Double[1];
	private Vector LRrowVector;
	private int outputNodes;
	
	private enum Algorithm {NaiveBayes, LogisticRegression, MultilayerPerceptron};
	private Algorithm algorithm;
	private MultilayerPerceptron MLPmodel;
	private OnlineLogisticRegression LRmodel;
	private StandardNaiveBayesClassifier NBmodel;
	
	public MLModelEvaluator() {
	}

	@Override
	protected void init(Expression[] parameters, List<ExpressionExecutor> expressionExecutors,
	                    StreamDefinition inStreamDefinition, StreamDefinition outStreamDefinition,
	                    String elementId, SiddhiContext siddhiContext) {
		Logger logger = Logger.getLogger(MLModelEvaluator.class);
		for (Expression parameter : parameters) {
			if (parameter instanceof Variable) {
				Variable var = (Variable) parameter;
				String attributeName = var.getAttributeName();
				parameterPositions.put(attributeName, inStreamDefinition.getAttributePosition(attributeName));
			}
		}
		// check whether the first and second arguments in siddhi query are strings
		if (parameters[0] instanceof StringConstant && parameters[0] instanceof StringConstant) {
			Expression expression = parameters[0];
			ExpressionExecutor executor = ExecutorParser.parseExpression(expression, null,elementId, false,siddhiContext);
			modelPath = (String) executor.execute(null);
			expression = parameters[1];
			executor = ExecutorParser.parseExpression(expression, null, elementId, false, siddhiContext);
			algo = (StringConstant) executor.execute(null);
			String logMessage = null;

			// if a valid model file path is given
			if (isFilePath(modelPath)) {
				if (Algorithm.MultilayerPerceptron.toString().equalsIgnoreCase(algo.toString())) {
					MLPmodel = ModelInitializer.InitializeMultilayerPerceptron(modelPath);
					algorithm= Algorithm.MultilayerPerceptron;
					Matrix[] weights = MLPmodel.getWeightMatrices();
					outputNodes= weights[weights.length - 1].numRows();
				} else if (Algorithm.LogisticRegression.toString().equalsIgnoreCase(algo.toString())) {
					algorithm= Algorithm.LogisticRegression;
					LRmodel=ModelInitializer.InitializeLogisticRegression(modelPath);
					LRrowVector = new RandomAccessSparseVector(LRmodel.numFeatures());
				} else if (Algorithm.NaiveBayes.toString().equalsIgnoreCase(algo.toString())) {
					algorithm= Algorithm.NaiveBayes;
					NBmodel = ModelInitializer.InitializeNaiveBayes(modelPath);
				} else {
					logger.debug("Invalid algorithm. Please provide a valid alorithm type for  \""+ modelPath+"\" model file.");
				}
				for (String field : parameterPositions.keySet()) {
					features.add(field);
				}
				logger.debug(logMessage);
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

	@Override
	protected InStream processEvent(InEvent inEvent) {
		for (String feature : features) {
			dataRow[parameterPositions.get(feature)] = (double) inEvent.getData(parameterPositions.get(feature));
		}
		dataRowVector.assign(dataRow);
		switch (algorithm) {
			case MultilayerPerceptron:
				precidtedResponse = MLPmodel.getOutput(dataRowVector);
	    		if (outputNodes == 1) {
	    			result[0] = (double) Math.round(precidtedResponse.get(0));
	    		} else {
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

	@SuppressWarnings("unchecked")
	@Override
	protected void restoreState(Object[] objects) {
		if (objects.length > 0 && objects[0] instanceof Map) {
			parameterPositions = (Map<String, Integer>) objects[0];
		}
	}

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