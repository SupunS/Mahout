package org.wso2.mahout.algorithms.regression;

import java.io.Serializable;

public class LogisticRegressionModel  implements Serializable{
	
    private static final long serialVersionUID = 1L;

    private int numFeatures;
    private int numCategories;
    private  double[][] weights;
    private  double learningRate;
    private  double lambda;
    private  double alpha;
    private  int stepOffset;
    private  double decayExponent;
    
    public LogisticRegressionModel(int numFeatures,int numCategories,double [][] weights,double learningRate,double lambda ,double alpha,int stepOffset,double decayExponent){
    	this.weights=weights;
    	this.learningRate=learningRate;
    	this.lambda=lambda;
    	this.alpha=alpha;
    	this.stepOffset=stepOffset;
    	this.decayExponent=decayExponent;
    	this.numFeatures=numFeatures;
    	this.numCategories=numCategories;
    }

    public int getNumFeatures(){
    	return this.numFeatures;
    }
    
    public int getNumCategories(){
    	return this.numCategories;
    }
    
    public double[][] getWeights(){
    	return this.weights;
    }
    
    public double getLearningRate(){
    	return this.learningRate;
    }
    
    public double getLambda(){
    	return this.lambda;
    }
    
    public double getAlpha(){
    	return this.alpha;
    }
    
    public int getStepOffset(){
    	return this.stepOffset;
    }
    
    public double getDecayExponent(){
    	return this.decayExponent;
    }
}
