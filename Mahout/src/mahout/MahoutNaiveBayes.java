package mahout;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.test.TestNaiveBayesDriver;
import org.apache.mahout.classifier.naivebayes.training.TrainNaiveBayesJob;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class MahoutNaiveBayes {
	public static void main(String [] args) throws Exception{
		String parentPath= "/home/supun/Supun/data/ForestCoverType";
		//String parentPath= "/home/supun/Supun/data/Iris/";
		//String parentPath= "/home/supun/Supun/data/Influencer/";

		String trainData = parentPath +"/trainStdFull.csv";
		String testData = parentPath + "/validate.csv";
		String seqPath = parentPath + "/resources/sequence/";
		String modelPath = parentPath + "/resources/model";
		String labelIndexPath = parentPath + "/resources/labelIndex";
		String validateResultsPath = parentPath + "/resources/results";

		Configuration configuration = new Configuration();
		FileSystem hdfs = FileSystem.get(configuration);

		// create training and validating sequence files
	/*	String trainSeqFile = createSequenceFiles(trainData, seqPath + "/trainSeq", configuration, hdfs);
		String testSeqFile = createSequenceFiles(testData, seqPath + "/testSeq", configuration, hdfs);

		// Train
		TrainNaiveBayesJob trainer = new TrainNaiveBayesJob();
		trainer.setConf(configuration);
		String[] trainingParameters = { "-i", trainSeqFile, "-o", modelPath, "-li", labelIndexPath, "-ow", "-el" };
		trainer.run(trainingParameters);

		// Validate
		TestNaiveBayesDriver test = new TestNaiveBayesDriver();
		test.setConf(configuration);
		String[] testingParameters = { "-i", testSeqFile, "-m", modelPath, "-l", labelIndexPath, "-o", validateResultsPath, "-ow"};
		test.run(testingParameters);*/

		// Test (predict)
/*		FileWriter writer = new FileWriter("/home/supun/Supun/data/Influencer/predicted3.csv");
		writer.append("Id");
		writer.append(',');
		writer.append("Cover_Type");
		writer.append('\n');*/
		int Id=1;
		NaiveBayesModel model= NaiveBayesModel.materialize(new Path(modelPath),configuration);
		StandardNaiveBayesClassifier classifier = new StandardNaiveBayesClassifier(model);
		List <Vector> testDataSet=createDataVector(testData, hdfs);
		for(Vector dataRow : testDataSet){
			if(classifier.classifyFull(dataRow).maxValueIndex()==0){
				//writer.append(Id+",7\n");
				System.out.println(classifier.classifyFull(dataRow).maxValueIndex()+"-->"+classifier.classifyFull(dataRow));
			}else{
				//writer.append(Id+","+classifier.classifyFull(dataRow).maxValueIndex()+"\n");
				System.out.println(classifier.classifyFull(dataRow).maxValueIndex()+"-->"+classifier.classifyFull(dataRow));
			}
			Id++;
		}
		//writer.close();
		System.out.println("Done!");
	}


	private static String createSequenceFiles(String inputData, String seqPath,Configuration configuration, FileSystem fileSystem) throws Exception {
		Writer trainDataWriter = new SequenceFile.Writer(fileSystem, configuration,new Path(seqPath), Text.class,VectorWritable.class);
		Text key = new Text();
		VectorWritable writableVector = new VectorWritable();
		String line;
		int count=0;
		/*
		 * find the number of features.
		 * ignore the first line (headers) of data.
		 */
		FSDataInputStream inStream = fileSystem.open(new Path(inputData));
		BufferedReader dataReader = new BufferedReader(new InputStreamReader(inStream));

		int featureCount = dataReader.readLine().split(",").length-1;
		double[] dataRow = new double[featureCount];
		Vector rowVector = new RandomAccessSparseVector(featureCount);

		/*
		 * create the vector and write to the sequence file
		 */
		while ((line = dataReader.readLine()) != null) {
			String[] values = line.split(",");
			// set response variable class as key
			key.set("/" + values[0] + "/" + values[0]);
			// create a data array for a given row
			for (int i = 1; i < values.length; i++) {
				dataRow[i-1] = Double.parseDouble(values[i]);
			}
			// assign it to the vector
			rowVector.assign(dataRow);
			writableVector.set(rowVector);
			trainDataWriter.append(key, writableVector);
			count++;
		}
		inStream.close();
		trainDataWriter.close();
		System.out.println("Wrote " + count + " entries.");
		return seqPath;
	}

	private static List<Vector> createDataVector(String inputData, FileSystem fileSystem) throws Exception {
		FSDataInputStream dataStream = fileSystem.open(new Path(inputData));
		BufferedReader dataReader = new BufferedReader(new InputStreamReader(dataStream));
		List <Vector> testDataSet = new ArrayList <Vector> ();
		String line;
		// ignore the first line (headers)
		dataReader.readLine();
		while ((line = dataReader.readLine()) != null) {
			String[] values = line.split(",");
			Vector tmpVector = new RandomAccessSparseVector(values.length-1);
			double[] tmp = new double[values.length-1];
			for (int i = 1; i < values.length; i++) {
				tmp[i-1] = Double.parseDouble(values[i]);
			}
			tmpVector.assign(tmp);
			testDataSet.add(tmpVector);
		}
		dataReader.close();
		dataStream.close();
		return testDataSet;
	}
}
