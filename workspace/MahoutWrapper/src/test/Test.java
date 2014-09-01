package test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.sgd.L2;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.wso2.mahout.algorithms.bayesian.NaiveBayes;
import org.wso2.mahout.algorithms.clustering.Kmeans;
import org.wso2.mahout.algorithms.clustering.Kmeans.distanceMeasure;

public class Test {
	private static List <Vector> trainingFeatureSet = new ArrayList <Vector>();
	private static List <Integer> trainingResponseSet =new ArrayList <Integer> ();
	private static List <Integer> trainingIndexSet = new ArrayList <Integer> ();
	static Matrix matrix;
	private static int numCategories;
	private static int numFeatures;
	
	public static void main(String[] args) throws IOException {
		
		Configuration configuration = new Configuration();
		FileSystem hdfs;
        try {
	        hdfs = FileSystem.get(configuration);
	        createDataVectors("/home/supun/Supun/data/Influencer/trainStdFull.csv",hdfs);
        } catch (Exception e) {
	        e.printStackTrace();
        }
        
/*		LogisticRegression LR = new LogisticRegression(numCategories, numFeatures, new L2(1));
		LR.setParameters(0.01, 0.0001, 1, 0.001, 1000);
		LR.train(trainingIndexSet, trainingResponseSet, trainingFeatureSet, 30);
		LR.export("/home/supun/Supun/LRModel");
		
		LogisticRegression LR2 = new LogisticRegression("/home/supun/Supun/LRModel");
		LR2.test(trainingResponseSet, trainingFeatureSet);

        
        MultilayerPerceptron MLP = new MultilayerPerceptron(numCategories, numFeatures);
        MLP.setParameters(0.1);
        MLP.train(trainingIndexSet, trainingResponseSet, trainingFeatureSet, 10);
        MLP.export("/home/supun/Supun/MLPModel");
        
        MultilayerPerceptron MLP2 = new MultilayerPerceptron("/home/supun/Supun/MLPModel");
        MLP2.test(trainingResponseSet, trainingFeatureSet);
*/
        
        Kmeans cluster= new Kmeans(5,distanceMeasure.MAHALANOBIS_DISTANCE);
        cluster.run(trainingFeatureSet, 5);
        //cluster.getOutput();

        
 /*     NaiveBayes NB =new NaiveBayes();
        NB.train(trainingIndexSet, trainingResponseSet, trainingFeatureSet, 100);
        NB.test(trainingResponseSet, trainingFeatureSet);
*/
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	private static void createDataVectors(String inputData, FileSystem fileSystem) throws Exception {
		FSDataInputStream dataStream = fileSystem.open(new Path(inputData));
		BufferedReader dataReader = new BufferedReader(new InputStreamReader(dataStream));		
		List <String> classes = new ArrayList<String>();		
		
		/*ignore the first line (headers)
		 *  Set the number of features in the data
		 */
		numFeatures = dataReader.readLine().split(",").length-1;
		String line;
		double[] dataRaw = new double[numFeatures];
		Vector rawVector;
		int index=0;
		String[] values;
		// create a training data vector
		while((line = dataReader.readLine())!=null){
			values = line.split(",");
			rawVector = new RandomAccessSparseVector(numFeatures);
			
			// find distinguished categories
			if(!classes.contains(values[0])){
				classes.add(values[0]);
			}
			//convert the values in to a double array
			for (int i = 0; i < numFeatures; i++) {
				dataRaw[i] = Double.parseDouble(values[i+1]);
			}
			//assign the double array to a vector
			rawVector.assign(dataRaw);
			
			trainingResponseSet.add(Integer.parseInt(values[0]));
			
			//add the vector to the list
			trainingFeatureSet.add(rawVector);
			trainingIndexSet.add(index++);
		}
		numCategories=classes.size();
		dataReader.close();
		dataStream.close();
	}

}
