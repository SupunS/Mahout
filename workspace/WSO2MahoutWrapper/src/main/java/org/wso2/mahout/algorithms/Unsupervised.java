package org.wso2.mahout.algorithms;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.log4j.Logger;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public abstract class Unsupervised {
	
	Logger logger = Logger.getLogger(Unsupervised.class);
	
	/*
	 * Creates the sequence file from the input data, to be used by the clustering algorithm
	 */
	public void createSequenceFiles(List<Vector> featureSet,Path output,Configuration configuration,FileSystem fileSystem) {
		Writer sequenceFileWriter;
        try {
        	sequenceFileWriter = new SequenceFile.Writer(fileSystem, configuration, output , Text.class, VectorWritable.class);
			VectorWritable writableVector = new VectorWritable();
			for (int index=0 ; index<featureSet.size() ; index++) {
				//Create a named vector with row number as the key, and the feature set as the values
				NamedVector dataVector = new NamedVector(featureSet.get(index), String.valueOf(index));
				//append it to the sequence file
				writableVector.set(dataVector);
				sequenceFileWriter.append(new Text(dataVector.getName()), writableVector);
			}
			sequenceFileWriter.close();
        } catch (IOException e) {
        	e.printStackTrace();
        	logger.error("Failed to create sequence files.");
        }
        logger.info("Successfully wrote sequence files to: " +output.toString());
	}
	
	public abstract void run(List <Vector> featureSet, int passes);
	
	public abstract void getOutput();
}
