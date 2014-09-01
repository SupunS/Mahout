package mahout;

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
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class CopyOfLogisticRegression {
	
		private static int numCategories;
		private static int numFeatures;
		private static List <Vector> trainingFeatureSet = new ArrayList <Vector>();
		private static List <Integer> trainingResponseSet =new ArrayList <Integer> ();
		private static List <Integer> trainingIndexSet = new ArrayList <Integer> ();
		private static List <Vector> testFeatureSet = new ArrayList <Vector>();
		private static List <Integer> testResponseSet =new ArrayList <Integer> ();
		private static List <Integer> testIndexSet = new ArrayList <Integer> ();		

	public static void main(String[] args) throws Exception{
		//String trainDataPath =  "/home/supun/Supun/data/ForestCoverType/train.csv";
		//String testDataPath =  "/home/supun/Supun/data/ForestCoverType/validate.csv";		
		String trainDataPath = "/home/supun/Supun/data/Influencer/train.csv";
		String testDataPath =  "/home/supun/Supun/data/Influencer/validate.csv";
 /*   	FileWriter writer = new FileWriter("/home/supun/Supun/data/ForestCoverType/predicted3.csv");
	    writer.append("Id");
	    writer.append(',');
	    writer.append("Cover_Type");
	    writer.append('\n');
	*/    
		Configuration configuration = new Configuration();
		FileSystem hdfs = FileSystem.get(configuration);
		
		createTrainDataVector(trainDataPath,hdfs);
		System.out.println("Training set size: "+trainingIndexSet.size());
		
		createTestDataVector(testDataPath,hdfs);
		System.out.println("Test set size: "+testIndexSet.size());		
		System.out.println("Number of Features: "+ (numFeatures-1));
		System.out.println("Number of Response Categories: " + numCategories);

		Random random = RandomUtils.getRandom();

		for(int run=0; run<10 ; run++){
    		OnlineLogisticRegression model = new OnlineLogisticRegression(numCategories, numFeatures, new L2(1));
    		model.learningRate(0.1);
    		model.lambda(0.0001);
    		model.alpha(1);
    		model.stepOffset(1000);
    	    model.decayExponent(0.01);

    	    for(int pass=0 ; pass<10 ; pass++){
    			Collections.shuffle(trainingIndexSet,random);
        		for(int k : trainingIndexSet){
        			model.train(trainingResponseSet.get(k), trainingFeatureSet.get(k));
        		}
        	}    		
    		int total=0;
        	int correct=0;
    	    
        	for(int i:testIndexSet){
            	Vector precidtedResponse = model.classifyFull(testFeatureSet.get(i));
            	//writer.append(testResponseSet.get(i)+","+precidtedResponse.get(1)+"\n");
            	
/*            	if(precidtedResponse.maxValueIndex()==0){
            		writer.append(testResponseSet.get(i)+",7\n");            		
            	}else{
            		writer.append(testResponseSet.get(i)+","+precidtedResponse.maxValueIndex()+"\n");
            	}
*/           	if(precidtedResponse.maxValueIndex()==testResponseSet.get(i)){
            		correct++;
            	}
            	total++;
        	}
        	//writer.close();
        	System.out.println("Accuracy = " +correct*1.0/total*100 +" %");
        	model.close();
		}
    	System.err.println("Done!");
	}
	
	
	private static void createTrainDataVector(String trainData, FileSystem fileSystem) throws Exception {
		FSDataInputStream dataStream = fileSystem.open(new Path(trainData));
		BufferedReader dataReader = new BufferedReader(new InputStreamReader(dataStream));		
		List <String> classes = new ArrayList<String>();
		String line;
		
		/*ignore the first line (headers)
		 *  Set the number of features in the data
		 */
		numFeatures = dataReader.readLine().split(",").length;
		double[] dataRaw = new double[numFeatures];
		Vector rawVector;
		int index=0;
		// create a vector from the data
		while ((line = dataReader.readLine()) != null) {
			String[] values = line.split(",");
			rawVector = new RandomAccessSparseVector(numFeatures);
			// find distinguished categories
			if(!classes.contains(values[0])){
				classes.add(values[0]);
			}			
			// set the intercept term to 1
			dataRaw[0]=1;
			
			for (int i = 1; i < numFeatures; i++) {
				dataRaw[i] = Double.parseDouble(values[i]);
			}
			rawVector.assign(dataRaw);
			trainingResponseSet.add(Integer.parseInt(values[0]));
			trainingFeatureSet.add(rawVector);
			trainingIndexSet.add(index++);
		}
		// set the number of categories
		numCategories=classes.size();
		dataReader.close();
		dataStream.close();
	}
	
	private static void createTestDataVector(String testData, FileSystem fileSystem) throws Exception {
		testIndexSet.clear();
		testFeatureSet.clear();
		testResponseSet.clear();
		
		FSDataInputStream dataStream = fileSystem.open(new Path(testData));
		BufferedReader dataReader = new BufferedReader(new InputStreamReader(dataStream));
		String line;
		
		/*ignore the first line (headers)
		 *  create a vector with the size of number of features in the data
		 */
		double[] dataRaw = new double[dataReader.readLine().split(",").length];
		Vector rawVector;
		int index=0;
		String[] values;
		// create a vector from the data
		while ((line = dataReader.readLine()) != null) {
			values = line.split(",");
			rawVector = new RandomAccessSparseVector(numFeatures);
			// set the intercept term to 1
			dataRaw[0]=1;
			
			for (int i = 1; i < numFeatures; i++) {
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
