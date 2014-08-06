package mahout;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.sgd.L2;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

public class CrossValidator {
	static List <Vector> featureSet = new ArrayList <Vector>();
	static List <Integer> responseSet =new ArrayList <Integer> ();
	static List <Integer> indexSet = new ArrayList <Integer> ();		
	static int numCategories;
	static int numFeatures;
	static int size = 0;
	static List <List <Vector>> trainSets = new ArrayList <List<Vector>>();
	static List <List <Vector>> validateSets = new ArrayList <List<Vector>>();
	static List <List <Integer>> trainResponseSets = new ArrayList <List<Integer>>();
	static List <List <Integer>> validateResponseSets = new ArrayList <List<Integer>>();

	public static void main(String[] args) throws Exception{		
		long startTime=System.currentTimeMillis();
		DecimalFormat decimalFormatter =  new DecimalFormat("#.##");
		String inputData="/home/supun/Supun/data/Influencer/train.csv";
		Configuration configuration = new Configuration();
		FileSystem hdfs = FileSystem.get(configuration);
		// count number of records
		FSDataInputStream lineNumberStream = hdfs.open(new Path(inputData));
		BufferedReader lineNumberReader = new BufferedReader(new InputStreamReader(lineNumberStream));
		lineNumberReader.readLine();
		while (lineNumberReader.readLine() != null) {
			indexSet.add(size);
			size++;
		}
		lineNumberStream.close();
		
		createDataVectors(inputData,hdfs);
		Random random = RandomUtils.getRandom();
		Collections.shuffle(indexSet, random);
		partition();

		//train
		System.out.print("Cross Validation in progress. Please wait..");
		Vector precidtedResponse;

       	double[] decayExponentSet={1,0.1,0.01,0.001,0.0001};
       	double[] learningRateSet={1,0.1,0.01,0.001,0.0001};
    	double[] lambdaSet={1,0.1,0.01,0.001,0.0001};
       	double[] bestParameters = new double[5];
       	double highestAvg=0;
       	for(double learningRate : learningRateSet){
       		for(double lambda : lambdaSet){
           		for(double decayExponent : decayExponentSet){
                   	double sum=0;
               		for(int k=0 ; k<10 ; k++){
            	   		int total=0;
            	       	int correct=0;
            			OnlineLogisticRegression  model = new OnlineLogisticRegression(numCategories, numFeatures+1, new L2(1));
            			model.learningRate(learningRate);
            	   		model.lambda(lambda);
            	   		model.alpha(1);
            	   		model.stepOffset(10000);
            	   	    model.decayExponent(decayExponent);
            	   	    int j=0;
            			for(int pass=0 ; pass<10 ; pass++){
            	       		for(int instance = 0 ; instance < trainSets.get(k).size() ; instance++ ){
            	       			model.train(trainResponseSets.get(k).get(instance), trainSets.get(k).get(instance));
            	       		}
            	       	}
            	       	for(int i = 0 ; i<validateSets.get(k).size() ; i++){
            	           	precidtedResponse = model.classifyFull(validateSets.get(k).get(i));
            	           	if(precidtedResponse.maxValueIndex()==validateResponseSets.get(k).get(i)){
            	           		correct++;
            	           	}
            	           	total++;
            	       	}
            	       	//System.out.println("Acc: " + correct*1.0/total*100);
            	       	sum=sum+correct*1.0/total*100;
            	       	model.close();
               		}
            		//System.out.println("Average: " +sum/10);
            		if(highestAvg<sum/10){
            			highestAvg=sum/10;
            			bestParameters[0]=learningRate;
            			bestParameters[1]=lambda;
            			bestParameters[4]=decayExponent;   			
            		}
           		}
           		System.out.print(".");
           	}
       	}
       	System.out.println("Accuracy: " + highestAvg);
    	System.out.println("\nLearning Rate: "+bestParameters[0]);
       	System.out.println("Lambda: "+bestParameters[1]);
       	System.out.println("Decay: "+bestParameters[4]);
       	System.err.println("Done!\nTime elapsed: "+decimalFormatter.format((System.currentTimeMillis()-startTime)*1.0/1000/60)+" mins.");
	}
	
	private static void createDataVectors(String inputData,FileSystem hdfs) throws Exception{
		System.out.println("Creating  the data vectors...");
		FSDataInputStream dataStream = hdfs.open(new Path(inputData));
		BufferedReader dataReader = new BufferedReader(new InputStreamReader(dataStream));
		List <String> classes = new ArrayList<String>();
		numFeatures = dataReader.readLine().split(",").length-1;
		double[] dataRaw = new double[numFeatures+1];
		Vector rawVector;
		String[] values;
		String line;
		// create a training data vector
		for(int lineNo=0 ; lineNo<size ; lineNo++){
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
			responseSet.add(Integer.parseInt(values[0]));
			featureSet.add(rawVector);
		}
		numCategories=classes.size();
		System.out.println("Creating  the data vectors completed!");
	}
	
	private static void partition(){
		System.out.println("Partitioning data...");
		int lowerBound=0;
		int sampleSize=size/10;
		int upperBound;
		for(int k=0 ; k<10 ; k++){
			List <Vector> tempTrainSet = new ArrayList <Vector>();
			List <Integer> tempTrainResponseSet = new ArrayList <Integer>();
			List <Vector> tempValidateSet = new ArrayList <Vector>();
			List <Integer> tempValidateResponseSet = new ArrayList <Integer>();
    		upperBound=lowerBound+sampleSize;
    		for (int index =0 ; index<size ; index++){
    			if(index>=lowerBound && index<upperBound){
    				tempValidateSet.add(featureSet.get(indexSet.get(index)));
    				tempValidateResponseSet.add(responseSet.get(indexSet.get(index)));
    			}else{
    				tempTrainSet.add(featureSet.get(indexSet.get(index)));
    				tempTrainResponseSet.add(responseSet.get(indexSet.get(index)));
    			}
    		}
    		trainSets.add(tempTrainSet);
    		validateSets.add(tempValidateSet);    		
    		trainResponseSets.add(tempTrainResponseSet);
    		validateResponseSets.add(tempValidateResponseSet);
    		lowerBound=lowerBound+sampleSize;
		}
		System.out.println("Partitioning data completed!");
	}
}
