package org.wso2.cep.data.publisher;

import java.io.File;
import java.io.FileReader;
import java.util.List;

import org.apache.log4j.Logger;
import org.supercsv.io.CsvListReader;
import org.supercsv.prefs.CsvPreference;
import org.wso2.carbon.databridge.agent.thrift.DataPublisher;

public class MLDataPublisher {
	
	Logger log = Logger.getLogger("org.wso2.cep.data.publisher");
	   
	public static void main(String[] args) throws Exception {	
		setTrustStoreParams();
		MLDataPublisher mldb =new MLDataPublisher();
		mldb.sendData();
	}	   
	  
	public void sendData() throws Exception{
		DataPublisher dataPublisher = new DataPublisher("tcp://localhost:7611","admin","admin");		   
		   
		String jsonObject = "{ 'name': 'InfluencerInputStream'," +
				"'version': '1.0.0'," +
				"'nickName': ''," +
				" 'description': ''," +
				" 'payloadData': [" +
				" {'name': 'A_follower_count','type': 'DOUBLE'}," +
				" {'name': 'A_following_count','type': 'DOUBLE'}," +
				" {'name': 'A_listed_count','type': 'DOUBLE'}," +
				" {'name': 'A_mentions_received','type': 'DOUBLE'}," +
				" {'name': 'A_retweets_received','type': 'DOUBLE'}," +
				" {'name': 'A_mentions_sent','type': 'DOUBLE'}," +
				" {'name': 'A_retweets_sent','type':'DOUBLE'}," +
				" {'name': 'A_posts','type': 'DOUBLE'}," +
				" {'name': 'A_network_feature_1','type': 'DOUBLE' }," +
				" {'name': 'A_network_feature_2','type': 'DOUBLE'}," +
				" {'name': 'A_network_feature_3','type': 'DOUBLE'}," +
				" {'name': 'B_follower_count','type': 'DOUBLE'}," +
				" {'name': 'B_following_count','type': 'DOUBLE'}," +
				" {'name': 'B_listed_count','type': 'DOUBLE'}," +
				" {'name': 'B_mentions_received','type': 'DOUBLE'}," +
				" {'name': 'B_retweets_received','type': 'DOUBLE'}," +
				" {'name': 'B_mentions_sent','type': 'DOUBLE'}," +
				" {'name': 'B_retweets_sent','type': 'DOUBLE'}," +
				" {'name': 'B_posts','type': 'DOUBLE' }," +
				" {'name': 'B_network_feature_1','type': 'DOUBLE' }," +
				" {'name': 'B_network_feature_2','type': 'DOUBLE'}," +
				" {'name': 'B_network_feature_3','type': 'DOUBLE'}" +
				"] }";
		
		String streamId1 = dataPublisher.defineStream(jsonObject);
		log.info("1st stream created!: " + streamId1);
		
		/* Read from a file and
		 *   send a data stream every 3 seconds
		 */
    	File dataFile = new File("/home/supun/Supun/data/Influencer/validate.csv");
    	CsvListReader reader = new CsvListReader(new FileReader(dataFile), CsvPreference.STANDARD_PREFERENCE);    	
    	List<String> row;
    	int rowNo=1;    	
		while((row=reader.read())!=null){			
			Double [] dataArray = new Double[row.size()-1];
			if(rowNo>1){
				for(int i=1 ; i<row.size() ; i++){
					String bodyCell = row.get(i);
					dataArray[i-1]=Double.valueOf(bodyCell);
				}
				dataPublisher.publish(streamId1, null, null,dataArray);
				Thread.sleep(2000);
			}
			if(rowNo==11){
				break;
			}

			System.out.println(rowNo);
			rowNo++;
    	}		
        log.info("Event published to 1st stream");
        dataPublisher.stop();
        reader.close();
	}
	
	
	public static void setTrustStoreParams() {
        File filePath = new File("src/main/resources");
        if (!filePath.exists()) {
            filePath = new File("resources");
        }
        String trustStore = filePath.getAbsolutePath();
        System.setProperty("javax.net.ssl.trustStore", trustStore + "/client-truststore.jks");
        System.setProperty("javax.net.ssl.trustStorePassword", "wso2carbon");

    }
}
