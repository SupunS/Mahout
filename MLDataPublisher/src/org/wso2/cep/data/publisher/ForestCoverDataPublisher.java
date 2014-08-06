package org.wso2.cep.data.publisher;

import java.io.File;
import java.io.FileReader;
import java.util.List;

import org.apache.log4j.Logger;
import org.supercsv.io.CsvListReader;
import org.supercsv.prefs.CsvPreference;
import org.wso2.carbon.databridge.agent.thrift.DataPublisher;

public class ForestCoverDataPublisher {
	
	Logger log = Logger.getLogger("org.wso2.cep.data.publisher");
	   
	public static void main(String[] args) throws Exception {	
		setTrustStoreParams();
		ForestCoverDataPublisher mldb =new ForestCoverDataPublisher();
		mldb.sendData();
	}	   
	  
	public void sendData() throws Exception{
		DataPublisher dataPublisher = new DataPublisher("tcp://localhost:7611","admin","admin");		   
		   
		String jsonObject = "{ 'name': 'ForestCoverInputStream'," +
				"'version': '1.0.0'," +
				"'nickName': ''," +
				" 'description': ''," +
				" 'payloadData': [" +
				" {'name': 'Elevation','type': 'DOUBLE'}," +
				" {'name': 'Aspect','type': 'DOUBLE'}," +
				" {'name': 'Slope','type': 'DOUBLE'}," +
				" {'name': 'Horizontal_Distance_To_Hydrology','type': 'DOUBLE'}," +
				" {'name': 'Vertical_Distance_To_Hydrology','type': 'DOUBLE'}," +
				" {'name': 'Horizontal_Distance_To_Roadways','type':'DOUBLE'}," +
				" {'name': 'Hillshade_9am','type': 'DOUBLE'}," +
				" {'name': 'Hillshade_Noon','type': 'DOUBLE' }," +
				" {'name': 'Hillshade_3pm','type': 'DOUBLE'}," +
				" {'name': 'Horizontal_Distance_To_Fire_Points','type': 'DOUBLE'}," +
				" {'name': 'Wilderness_Area1','type': 'DOUBLE'}," +
				" {'name': 'Wilderness_Area2','type': 'DOUBLE'}," +
				" {'name': 'Wilderness_Area3','type': 'DOUBLE'}," +
				" {'name': 'Wilderness_Area4','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type1','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type2','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type3','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type4','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type5','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type6','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type7','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type8','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type9','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type10','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type11','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type12','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type13','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type14','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type15','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type16','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type17','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type18','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type19','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type20','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type21','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type22','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type23','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type24','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type25','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type26','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type27','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type28','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type29','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type30','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type31','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type32','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type33','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type34','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type35','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type36','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type37','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type38','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type39','type': 'DOUBLE'}," +
				" {'name': 'Soil_Type40','type': 'DOUBLE'}" +
				"] }";
		
		String streamId1 = dataPublisher.defineStream(jsonObject);
		log.info("1st stream created!: " + streamId1);
		
		/* Read from a file and
		 *   send a data stream every 3 seconds
		 */
    	File dataFile = new File("/home/supun/Supun/data/ForestCoverType/validate.csv");
    	CsvListReader reader = new CsvListReader(new FileReader(dataFile), CsvPreference.STANDARD_PREFERENCE);    	
    	List<String> row;
    	int rowNo=1;    	
		while((row=reader.read())!=null){			
			Double [] dataArray = new Double[row.size()-1];
			if(rowNo>1){
				for(int i=1 ; i<row.size() ; i++){
					String bodyCell = row.get(i);
					dataArray[i-1]=Double.valueOf(bodyCell);
					System.out.print(dataArray[i-1]+",");
				}
				System.out.println();
				dataPublisher.publish(streamId1, null, null,dataArray);
				//Thread.sleep(1000);
			}
			if(rowNo==1000){
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
