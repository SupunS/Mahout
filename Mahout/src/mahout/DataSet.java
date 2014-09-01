package mahout;

import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.math.Vector;

public class DataSet {
	List <Vector> featureSet = new ArrayList <Vector>();
	List <Integer> responseSet =new ArrayList <Integer> ();
	List <Integer> indexSet = new ArrayList <Integer> ();
	
	public void addFeatures(Vector features){
		this.featureSet.add(features);
	}
	
	public List <Vector> getFeatureSet(){
		return this.featureSet;
	}
	
	public void addResponse(Integer response){
		this.responseSet.add(response);
	}
	
	public List <Integer> getResponseSet(){
		return this.responseSet;
	}
	
	public void addIndex(int index){
		this.indexSet.add(index);
	}
	
	public List <Integer> getIndexSet(){
		return this.indexSet;
	}
}
