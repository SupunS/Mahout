package org.wso2.mahout.algorithms.clustering;

import java.io.IOException;
import java.util.List;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.io.Text;
import org.apache.log4j.Logger;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.classify.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.distance.ChebyshevDistanceMeasure;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.MahalanobisDistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.common.distance.MinkowskiDistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.common.distance.TanimotoDistanceMeasure;
import org.apache.mahout.common.distance.WeightedManhattanDistanceMeasure;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.wso2.mahout.algorithms.Unsupervised;

public class Kmeans extends Unsupervised {

	private Configuration configuration;
	private FileSystem fileSystem;
	private Logger logger;
	private String output="/home/supun/Supun/Kmeans";
	private int k=3;
	private Path inputSequence = new Path(output+"/inputSequence");
	private Path initialClusters = new Path(output+"/clustersIn/part-00000");
	private Path outputClusters = new Path(output+"/clustersOut");
	private Kluster cluster;
	private DistanceMeasure meassure;
	public enum distanceMeasure {CHEBYSHEV_DISTANCE , COSINE_DISTANCE , EUCLIDEAN_DISTANCE , MAHALANOBIS_DISTANCE , 
		MANHATTAN_DISTANCE , MINKOWSKI_DISTANCE , SQUARED_EUCLIDEAN_DISTANCE , TANIMOTO_DISTANCE , WEIGHTED_MANHATTAN_DISTANCE};

	public Kmeans(int clusters,distanceMeasure meassure) {
		logger = Logger.getLogger(Kmeans.class);
		configuration = new Configuration();
		try {
	        fileSystem = FileSystem.get(configuration);
	        this.k=clusters;			
			switch(meassure){
				case CHEBYSHEV_DISTANCE:
					this.meassure = new ChebyshevDistanceMeasure();
					break;
				case COSINE_DISTANCE:
					this.meassure = new CosineDistanceMeasure();
					break;
				case EUCLIDEAN_DISTANCE:
					this.meassure = new EuclideanDistanceMeasure();
					break;
				case MAHALANOBIS_DISTANCE:
					this.meassure = new MahalanobisDistanceMeasure();
					break;
				case MANHATTAN_DISTANCE:
					this.meassure = new ManhattanDistanceMeasure();
					break;
				case MINKOWSKI_DISTANCE:
					this.meassure = new MinkowskiDistanceMeasure();
					break;
				case SQUARED_EUCLIDEAN_DISTANCE:
					this.meassure = new SquaredEuclideanDistanceMeasure();
					break;
				case TANIMOTO_DISTANCE:
					this.meassure = new TanimotoDistanceMeasure();
					break;
				case WEIGHTED_MANHATTAN_DISTANCE:
					this.meassure = new WeightedManhattanDistanceMeasure();
					break;
				default:
					this.meassure = new EuclideanDistanceMeasure();
					break;
			}
        } catch (IOException e) {
        	logger.error("Failed to create filesystem configurations.");
        }		
	}
	
	/*
	 * Create K-clusters taking k-random points from the input data as the initial centroids.
	 */
	@Override
    public void run(List<Vector> featureSet, int passes) {
		//create sequence file from the input data 
		createSequenceFiles(featureSet,inputSequence,configuration,fileSystem);
		Random random = new Random();
		Writer writer;
        try {
        	writer = new SequenceFile.Writer(fileSystem, configuration,initialClusters, Text.class, Kluster.class);
        	//initialize the centroids of the k-clusters
        	for (int count = 0; count < k; count++) {
        		//get a random feature set from the input data
        		Vector feautreVector = featureSet.get(random.nextInt(featureSet.size()));
        		//set the above data vector as the initial center point of the cluster
        		cluster = new Kluster(feautreVector, count, meassure);
        		//write it to the initial-clusters file
        		writer.append(new Text(cluster.getIdentifier()), cluster);
        	}
			writer.close();
			//cluster the input data
			KMeansDriver.run(configuration, inputSequence , initialClusters, outputClusters, 0.001, passes, true, 0.1, false);
	        logger.info("Successfully created clusters. Saved to file: "+outputClusters+"/"+Kluster.CLUSTERED_POINTS_DIR+"/part-m-00000");
        } catch (Exception e) {
	        logger.error("Failed to create clusters.");
        }
    }

	/*
	 * Print the clusters to which each data set belongs.
	 */
	@Override
    public void getOutput() {
		SequenceFile.Reader reader;
		//set the path of the final clusters file
		String output=outputClusters+"/"+Kluster.CLUSTERED_POINTS_DIR+"/part-m-00000";
        try {
        	//read the file
	        reader = new SequenceFile.Reader(fileSystem, new Path(output), configuration);
			IntWritable key = new IntWritable();
			WeightedVectorWritable value = new WeightedPropertyVectorWritable();
			//for each line, print the data set and the cluster it belongs
			while (reader.next(key,value)) {
				System.out.println(" --> cluster "+ key.toString());
			}
			reader.close();
        } catch (IOException e) {
        	logger.error("Failed to read clusters from file: "+output);
        }
    }

	@Override
	public void createSequenceFiles(List<Vector> featureSet,Path output,Configuration configuration,FileSystem fileSystem) {
		Writer sequenceFileWriter;
        try {
        	sequenceFileWriter = new SequenceFile.Writer(fileSystem, configuration, output , Text.class, VectorWritable.class);
			VectorWritable writableVector = new VectorWritable();
			//create the vector and write to the sequence file
			for (int index=0 ; index<featureSet.size() ; index++) {
				NamedVector dataVector = new NamedVector(featureSet.get(index), String.valueOf(index));
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
}
