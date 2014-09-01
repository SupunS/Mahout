package org.wso2.mlearn;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.concurrent.Callable;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.naivebayes.test.TestNaiveBayesDriver;
import org.apache.mahout.classifier.naivebayes.training.TrainNaiveBayesJob;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.wso2.mlearn.MachineLearningService;
import org.wso2.mlearn.Options;

public class NaiveBayesService implements Callable<String>{
	private String inputPath;
	private String outputPath;
	private Configuration configuration;
	private FileSystem hdfs;

	public NaiveBayesService(String inputPath,String outputPath, Options [] options,Configuration configuration, FileSystem hdfs)
			throws Exception {
		this.inputPath=inputPath;
		this.outputPath=outputPath;
		this.configuration=configuration;
		this.hdfs=hdfs;
	}

	@Override
	public String call(){
		String jobId = String.valueOf(Thread.currentThread().getId());
		MachineLearningService.currentJobs.put(jobId,"active");
		String seqPath = outputPath + "/NaiveBayesModel/sequenceFiles/";
		String modelPath = outputPath + "/NaiveBayesModel/model";
		String labelIndexPath = outputPath + "/NaiveBayesModel/labelIndex";
		String validateResultsPath = outputPath + "/NaiveBayesModel/results";

		// create training and validating sequence files
		try {
			createSequenceFiles(inputPath, seqPath + "inSeq", configuration, hdfs);
		} catch (Exception e) {
			System.out.println("Failed to create the sequence files!\n"+e.getMessage());
		}

		// train
		TrainNaiveBayesJob trainer = new TrainNaiveBayesJob();
		trainer.setConf(configuration);
		String[] trainingParameters = { "-i", seqPath + "inSeqTrain", "-o", modelPath, "-li", labelIndexPath, "-ow", "-el" };
		try {
			trainer.run(trainingParameters);
		} catch (Exception e) {
			System.out.println("Navie Bayes model creation failed!\n"+e.getMessage());
		}

		// validate
		TestNaiveBayesDriver test = new TestNaiveBayesDriver();
		test.setConf(configuration);
		String[] testingParameters = { "-i", seqPath + "inSeqTest", "-m", modelPath, "-l", labelIndexPath, "-o", validateResultsPath, "-ow"};
		try {
			test.run(testingParameters);
		} catch (Exception e) {
			System.out.println("Model validation failed!\n"+e.getMessage());
		}
		MachineLearningService.currentJobs.put(jobId,"completed");
		return jobId;
	}


	public static void createSequenceFiles(String inputData, String seqPath,Configuration configuration, FileSystem fileSystem) throws Exception {
		Writer trainDataWriter = new SequenceFile.Writer(fileSystem, configuration, new Path(seqPath + "Train"), Text.class, VectorWritable.class);
		Writer validateDataWriter = new SequenceFile.Writer(fileSystem, configuration, new Path(seqPath + "Test"), Text.class, VectorWritable.class);
		Text key = new Text();
		VectorWritable writableVector = new VectorWritable();

		// find total number of records
		int count = 0;
		String line;
		FSDataInputStream lineNumberStream = fileSystem.open(new Path(inputData));
		BufferedReader lineNumberReader = new BufferedReader(new InputStreamReader(lineNumberStream));
		int size = -1;
		while (lineNumberReader.readLine() != null) {
			size++;
		}
		lineNumberStream.close();

		/*find the number of features.
		 * ignore the first line (headers) of data.
		 */
		FSDataInputStream inStream = fileSystem.open(new Path(inputData));
		BufferedReader dataReader = new BufferedReader(new InputStreamReader(inStream));
		int featureCount = dataReader.readLine().split(",").length;
		double[] dataRow = new double[featureCount];
		Vector rowVector = new RandomAccessSparseVector(featureCount);

		//create the vector and write to the sequence file
		while ((line = dataReader.readLine()) != null) {
			String[] values = line.split(",");
			// set response variable class as key
			key.set("/" + values[0] + "/" + values[0]);
			// create a data array for a given row
			for (int i = 1; i < values.length; i++) {
				dataRow[i] = Double.parseDouble(values[i]);
			}
			// assign it to the vector
			rowVector.assign(dataRow);
			writableVector.set(rowVector);
			if (count <= size * 0.7) {
				trainDataWriter.append(key, writableVector);
			} else {
				validateDataWriter.append(key, writableVector);
			}
			count++;
		}
		inStream.close();
		trainDataWriter.close();
		validateDataWriter.close();
	}
}
