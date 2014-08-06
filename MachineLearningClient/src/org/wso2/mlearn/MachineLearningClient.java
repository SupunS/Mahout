package org.wso2.mlearn;

import java.io.BufferedReader;
import java.io.InputStreamReader;

import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.DefaultHttpClient;

public class MachineLearningClient {

	public static void main(String [] args) throws Exception{

		//String inPath="/home/supun/Supun/data/Influencer/trainOrgStd.csv";
		String inPath =  "/home/supun/Supun/data/ForestCoverType/train.csv";
		String outPath="/home/supun/Supun/ModelFiles/";
		String algorithm="Multilayer Perceptron";
		//String algorithm="Logistic Regression";
		//String algorithm="Naive bayes";
		String [][] options={{"learningRate","1"},{"decay","0.01"},{"lambda","0.0001"}};
		String opts="\"options\":[";
		
		//assign all the options to the parameters
		for(int i=0 ; i<options.length ; i++){
			if(i!=0){
				opts=opts+",";
			}
			opts=opts+"{\"option\":\""+options[i][0]+"\",\"value\":\""+options[i][1]+"\"}";
		}
		opts=opts+"]";
		String parameters= "{\"Parameters\":{\"inPath\":\""+inPath+"\",\"outPath\":\""+outPath+"\",\"algorithm\":\""+algorithm+"\","+opts+"}}";
		System.out.println(parameters);
		
		//create a client and post the parameters to the service
		DefaultHttpClient httpclient = new DefaultHttpClient();
		HttpPost post = new HttpPost("http://localhost:9766/MachineLearningService-1.0.0/services/submit");

		StringEntity entity = new StringEntity(parameters);
		entity.setContentType("application/json");
		post.setEntity(entity);
		HttpResponse response = httpclient.execute(post);

		System.out.println(response.getStatusLine());
		BufferedReader br = new BufferedReader(new InputStreamReader(response.getEntity().getContent()));
		String output;
		while ((output = br.readLine()) != null) {
			System.out.println(output);
		}
		httpclient.getConnectionManager().shutdown();
	}
}
