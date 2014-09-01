package mahout;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.mlp.MultilayerPerceptron;
import org.apache.mahout.classifier.mlp.NeuralNetwork.TrainingMethod;
import org.apache.mahout.clustering.Model;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class MultilayerPerceptronService {

	private static int numCategories;
	private static int numFeatures;
	private static List <Vector> trainingDataSet = new ArrayList <Vector>();
	private static List <Integer> trainingIndexSet = new ArrayList <Integer> ();
	private static List <Vector> testFeatureSet = new ArrayList <Vector>();
	private static List <Integer> testResponseSet =new ArrayList <Integer> ();
	private static List <Integer> testIndexSet = new ArrayList <Integer> ();
	
    public static void main(String[] args) throws Exception{
    	//String trainDataPath =  "/home/supun/Supun/data/ForestCoverType/train.csv";
		//String testDataPath =  "/home/supun/Supun/data/ForestCoverType/validate.csv";
        //String trainDataPath = "/home/supun/Supun/data/Iris/iris.csv";
        String trainDataPath = "/home/supun/Supun/data/Influencer/trainOrgStd.csv";
		String testDataPath =  "/home/supun/Supun/data/Influencer/validate.csv";

        Configuration configuration = new Configuration();
        FileSystem hdfs = FileSystem.get(configuration);
       
        createDataVectors(trainDataPath,hdfs);
		System.out.println("Training set size: "+trainingIndexSet.size());
		System.out.println("Test set size: "+testIndexSet.size());
		
		//Train		   
        MultilayerPerceptron model = new MultilayerPerceptron();
        model.setLearningRate(01);
        model.addLayer(numFeatures, false, "Sigmoid");
        model.addLayer(numFeatures/3, false, "Sigmoid");        
        if(numCategories==2){
        	model.addLayer(1, true, "Sigmoid");
        }else{
        	model.addLayer(numCategories, true, "Sigmoid");
        }
        model.setTrainingMethod(TrainingMethod.GRADIENT_DESCENT);
        Random random = RandomUtils.getRandom();
        for(int pass=0 ; pass<100 ; pass++){
        	Collections.shuffle(trainingIndexSet,random);
            for(int index : trainingIndexSet){
                    model.trainOnline(trainingDataSet.get(index));
            }
        }
        model.setModelPath("/home/supun/Supun/InfluencerModel");
	    model.writeModelToFile();
	    MultilayerPerceptron newModel = new MultilayerPerceptron("/home/supun/Supun/InfluencerModel");
	    System.out.println(newModel.getModelType());
	    Matrix[] weights = newModel.getWeightMatrices();
	    System.out.println(weights[weights.length-1].numRows());
	    
        //Validate    
        int total=0;
        int correct=0;
        for(int index : testIndexSet){
        	Vector precidtedResponse = model.getOutput(testFeatureSet.get(index));
        	if(numCategories==2){
        		if(Math.round(precidtedResponse.get(0))==testResponseSet.get(index)){
                    correct++;
                }
        	}else{
                if(precidtedResponse.maxValueIndex()==testResponseSet.get(index)){
                    correct++;
                }
        	}
            total++;
        }
        System.out.println("Accuracy: " + correct*1.0/total*100+" %");
        model.close();
        System.err.println("Done!");
    }

    private static void createDataVectors(String trainData, FileSystem fileSystem) throws Exception {
		FSDataInputStream dataStream = fileSystem.open(new Path(trainData));
		BufferedReader dataReader = new BufferedReader(new InputStreamReader(dataStream));		
		List <String> classes = new ArrayList<String>();
		String line;
		int size=0;

		// find distinguished response classes and the dataset size
		BufferedReader linesReader = new BufferedReader(new InputStreamReader(fileSystem.open(new Path(trainData))));
		linesReader.readLine();
		while ((line=linesReader.readLine()) != null){
			String response = line.split(",")[0];
			if(!classes.contains(response)){
				classes.add(response);
			}
			size++;
		}
		numCategories = classes.size();
		linesReader.close();

		// create the training data vector
		numFeatures = dataReader.readLine().split(",").length-1;
		double[] dataRaw;
		Vector dataRawVector;
		int index=0;
		String[] values;
		
		if(numCategories==2){
			dataRaw = new double[numFeatures+1];
			for(int lineNo=0 ; lineNo<(int)(size*0.7) ; lineNo++){
				line = dataReader.readLine();
				values = line.split(",");
				//set the features
				for (int i = 1; i <= numFeatures; i++) {
					dataRaw[i-1] = Double.parseDouble(values[i]);
				}
				dataRawVector = new RandomAccessSparseVector(numFeatures+1);
				//set the response
	    		dataRaw[numFeatures] = Integer.parseInt(values[0]);
				dataRawVector.assign(dataRaw);
				trainingDataSet.add(dataRawVector);
				trainingIndexSet.add(index++);
			}
		}else{
			//set the training set
			dataRaw = new double[numFeatures+numCategories];
			for(int lineNo=0 ; lineNo<(int)(size*0.7) ; lineNo++){
				line = dataReader.readLine();
				values = line.split(",");		

				//set the features
				for (int i = 1; i <= numFeatures; i++) {
					dataRaw[i-1] = Double.parseDouble(values[i]);
				}		
				//set the responses
				dataRawVector = new RandomAccessSparseVector(numFeatures+numCategories);
				for (int j = 0; j < numCategories; j++) {
	    			if(Integer.parseInt(values[0])==j){
	    				dataRaw[numFeatures+j] = 1;
	    			}
	    			else{
	    				dataRaw[numFeatures+j] = 0;
	    			}
	    		}
				dataRawVector.assign(dataRaw);
				trainingDataSet.add(dataRawVector);
				trainingIndexSet.add(index++);
			}
		}

		//Set the validate set
		index=0;
		dataRaw = new double[numFeatures];
		for(int lineNo=(int)(size*0.7) ; lineNo<size ; lineNo++){
			line = dataReader.readLine();
			values = line.split(",");
			dataRawVector = new RandomAccessSparseVector(numFeatures);				
			for (int i = 1; i <= numFeatures; i++) {
				dataRaw[i-1] = Double.parseDouble(values[i]);
			}
			dataRawVector.assign(dataRaw);
			testResponseSet.add(Integer.parseInt(values[0]));
			testFeatureSet.add(dataRawVector);
			testIndexSet.add(index++);
		}
		
		dataReader.close();
		dataStream.close();
	}
}
