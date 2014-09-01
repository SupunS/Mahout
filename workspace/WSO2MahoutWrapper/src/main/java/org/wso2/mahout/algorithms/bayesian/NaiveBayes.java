package org.wso2.mahout.algorithms.bayesian;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.io.Text;
import org.apache.log4j.Logger;
import org.apache.mahout.classifier.naivebayes.test.TestNaiveBayesDriver;
import org.apache.mahout.classifier.naivebayes.training.TrainNaiveBayesJob;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.wso2.mahout.algorithms.Supervised;

public class NaiveBayes implements Supervised {

	private Configuration configuration;
	FileSystem fileSystem;
	private Logger logger;
	private String output="/home/supun/Supun/";
	private String sequenceFile=output+"/inputSequence";
	private String modelFile=output+"/NaiveBayesModel";
	private String labelIndexFile=output+"/labelIndex";
	private String summaryFile=output+"/summary";

	public NaiveBayes() {
		logger = Logger.getLogger(NaiveBayes.class);
		configuration = new Configuration();
		try {
	        fileSystem = FileSystem.get(configuration);
        } catch (IOException e) {
        	logger.error("Failed to load fileSystem configurations!");
        }
	}
	
    public void train(List<Integer> indexSet, List<Integer> responseSet, List<Vector> featureSet, int passes) {
		TrainNaiveBayesJob trainer = new TrainNaiveBayesJob();
		trainer.setConf(configuration);
		//create the sequence file using the input data
		createSequenceFiles(responseSet,featureSet,sequenceFile);
		//set the path to sequence file, path to the output model file and the labeled-index file
		String[] parameters = { "-i", sequenceFile , "-o", modelFile, "-li", labelIndexFile, "-ow", "-el" };
		try {
			//train the model using the data in the sequence file
			trainer.run(parameters);
		} catch (Exception e) {
			logger.error("Failed to train a Naive Bayes model!");
		}	    
    }


    public void test(List<Integer> responseSet, List<Vector> featureSet) {
		TestNaiveBayesDriver test = new TestNaiveBayesDriver();
		test.setConf(configuration);
		//set the path to sequence file, model file, labeled-index file and the path to the output summary file
		String[] parameters = { "-i", sequenceFile, "-m", modelFile, "-l", labelIndexFile, "-o", summaryFile, "-ow"};
		try {
			//test the model against the data in the sequence file
			test.run(parameters);
		} catch (Exception e) {
			logger.error("Failed to validate the created Naive Bayes model!");
		}	    
    }


    public void export(String exportPath) {
	    // TODO Auto-generated method stub	    
    }
	
	private void createSequenceFiles(List<Integer> responseSet, List<Vector> featureSet,String output) {
		Writer sequenceFileWriter;
        try {
        	sequenceFileWriter = new SequenceFile.Writer(fileSystem, configuration, new Path(output), Text.class, VectorWritable.class);
			Text key = new Text();
			VectorWritable writableVector = new VectorWritable();

			//create the vector and write to the sequence file
			for (int index=0 ; index<featureSet.size() ; index++) {
				// set response category as the key (i.e. name and value of the key)
				key.set("/" + responseSet.get(index) + "/" + responseSet.get(index));
				// assign the features to a writable vector
				writableVector.set(featureSet.get(index));
				//append the feature vector to the sequence file, having response as the key
				sequenceFileWriter.append(key, writableVector);
			}
			sequenceFileWriter.close();
        } catch (IOException e) {
        	logger.error("Failed to create sequence files!");
        }
	}
}
