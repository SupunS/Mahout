package mahout;

import mlearn.LogisticRegressionModel;

import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.sgd.L2;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class LogisticRegression {
	
		private static int numCategories;
		private static int numFeatures;
		private static List <Vector> trainingFeatureSet = new ArrayList <Vector>();
		private static List <Integer> trainingResponseSet =new ArrayList <Integer> ();
		private static List <Integer> trainingIndexSet = new ArrayList <Integer> ();
		private static List <Vector> testFeatureSet = new ArrayList <Vector>();
		private static List <Integer> testResponseSet =new ArrayList <Integer> ();
		private static List <Integer> testIndexSet = new ArrayList <Integer> ();	
		

	public static void main(String[] args) throws Exception{
		String trainDataPath =  "/home/supun/Supun/data/ForestCoverType/trainStdFull.csv";
		String testDataPath =  "/home/supun/Supun/data/ForestCoverType/validate.csv";		
		//String trainDataPath = "/home/supun/Supun/data/Influencer/trainStdFull.csv";
		//String testDataPath =  "/home/supun/Supun/data/Influencer/validate.csv";
 /*   	FileWriter writer = new FileWriter("/home/supun/Supun/data/ForestCoverType/predicted3.csv");
	    writer.append("Id");
	    writer.append(',');
	    writer.append("Cover_Type");
	    writer.append('\n');
	*/    
		Configuration configuration = new Configuration();
		FileSystem hdfs = FileSystem.get(configuration);
		
		createDataVectors(trainDataPath,hdfs);
		System.out.println("Training set size: "+trainingIndexSet.size());		
		System.out.println("Test set size: "+testIndexSet.size());		
		System.out.println("Number of Features: "+ (numFeatures));
		System.out.println("Number of Response Categories: " + numCategories);

		Random random = RandomUtils.getRandom();

		//SerializableLR model2 = new SerializableLR(numCategories, numFeatures+1, new L2(1));
		OnlineLogisticRegression  model = new OnlineLogisticRegression(numCategories, numFeatures+1, new L2(1));
   		model.learningRate(1);
   		model.lambda(0.0001);
   		model.alpha(1);
   		model.stepOffset(1000);
   	    model.decayExponent(0.001);
  	    for(int pass=0 ; pass<30 ; pass++){
   			Collections.shuffle(trainingIndexSet,random);
       		for(int k : trainingIndexSet){
       			model.train(trainingResponseSet.get(k), trainingFeatureSet.get(k));
       		}
       	}
  	/*    
  	    double [][] weights = new double[model.getBeta().numRows()][model.getBeta().numCols()];
  	  


  	    System.out.println(model.getBeta().numRows()+" , "+model.getBeta().numCols());
  	  for(int i=0 ; i<model.getBeta().numRows() ; i++){
  		  for(int j=0 ; j<model.getBeta().numCols() ; j++){
    	  	  weights[i][j]=model.getBeta().get(i, j);
      	  }
  	  }
  	  LogisticRegressionModel modelObject = new LogisticRegressionModel(model.numFeatures(),model.numCategories(),weights, model.currentLearningRate(), model.getLambda(), 1, model.getStep(), 0.01);
  	
  	  FileOutputStream fout = new FileOutputStream("/home/supun/Supun/LRobjectForestCover");
  	  ObjectOutputStream oos = new ObjectOutputStream(fout);
  	  oos.writeObject(modelObject);
  	  oos.close();
  	  fout.close();
  
  	  FileInputStream fin = new FileInputStream("/home/supun/Supun/ModelFiles/ForestCoverLR");
	  ObjectInputStream ois = new ObjectInputStream(fin);
	  LogisticRegressionModel LRmodelObject = (LogisticRegressionModel) ois.readObject();
		double [][] modelWeights = null;
    	OnlineLogisticRegression LRmodel = new OnlineLogisticRegression(LRmodelObject.getNumCategories(), LRmodelObject.getNumFeatures(), new L2(1));
    	LRmodel.learningRate(LRmodelObject.getLearningRate());
    	LRmodel.lambda(LRmodelObject.getLambda());
    	LRmodel.alpha(LRmodelObject.getAlpha());
    	LRmodel.stepOffset(LRmodelObject.getStepOffset());
    	LRmodel.decayExponent(LRmodelObject.getDecayExponent());
    	modelWeights=LRmodelObject.getWeights();
    	fin.close();
    	ois.close();
		  
    	for(int i=0 ; i<modelWeights.length ; i++){
	  		for(int j=0 ; j<modelWeights[0].length ; j++){
	  			LRmodel.setBeta(i,j, modelWeights[i][j]);
	  		}
	  	}

  */ 		int total=0;
       	int correct=0;   	    
       	for(int i:testIndexSet){
           	Vector precidtedResponse = model.classifyFull(testFeatureSet.get(i));
           	if(precidtedResponse.maxValueIndex()==testResponseSet.get(i)){
           		correct++;
           	}
           	total++;
       	}
       	System.out.println("Accuracy = " +correct*1.0/total*100 +" %");
       	model.close();
       	System.err.println("Done!");
}


	private static void createDataVectors(String inputData, FileSystem fileSystem) throws Exception {
		// find total number of records
		String line;
		FSDataInputStream lineNumberStream = fileSystem.open(new Path(inputData));
		BufferedReader lineNumberReader = new BufferedReader(new InputStreamReader(lineNumberStream));
		int size = -1;
		while (lineNumberReader.readLine() != null) {
			size++;
		}
		lineNumberStream.close();
		
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
			
			for (int i = 1; i <= numFeatures; i++) {
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
		for(int lineNo=(int)(size*0.9) ; lineNo<size ; lineNo++){
			line = dataReader.readLine();
			values = line.split(",");
			rawVector = new RandomAccessSparseVector(numFeatures+1);
			// set the intercept term to 1
			dataRaw[0]=1;
			
			for (int i = 1; i < numFeatures+1; i++) {
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
}
