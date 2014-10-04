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

package org.wso2.mlearn;

import java.io.Serializable;

public class LogisticRegressionModel implements Serializable {

	private static final long serialVersionUID = 1L;

	private int numFeatures;
	private int numCategories;
	private double[][] weights;
	private double learningRate;
	private double lambda;
	private double alpha;
	private int stepOffset;
	private double decayExponent;

	public LogisticRegressionModel(int numFeatures, int numCategories, double[][] weights,
	                               double learningRate, double lambda, double alpha,
	                               int stepOffset, double decayExponent) {
		this.weights = weights;
		this.learningRate = learningRate;
		this.lambda = lambda;
		this.alpha = alpha;
		this.stepOffset = stepOffset;
		this.decayExponent = decayExponent;
		this.numFeatures = numFeatures;
		this.numCategories = numCategories;
	}

	public int getNumFeatures() {
		return this.numFeatures;
	}

	public int getNumCategories() {
		return this.numCategories;
	}

	public double[][] getWeights() {
		return this.weights;
	}

	public double getLearningRate() {
		return this.learningRate;
	}

	public double getLambda() {
		return this.lambda;
	}

	public double getAlpha() {
		return this.alpha;
	}

	public int getStepOffset() {
		return this.stepOffset;
	}

	public double getDecayExponent() {
		return this.decayExponent;
	}
}
