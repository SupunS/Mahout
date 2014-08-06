package org.wso2.mlearn;

import javax.xml.bind.annotation.XmlRootElement;

@XmlRootElement(name = "Parameters")
public class Parameters {

	String inPath;
	String outPath;
	String algorithm;
	Options [] options;

	public String getinPath() {
		return inPath;
	}

	public void setinPath(String inPath) {
		this.inPath = inPath;
	}
	
	public String getoutPath() {
		return outPath;
	}

	public void setoutPath(String outPath) {
		this.outPath = outPath;
	}

	public String getalgorithm() {
		return algorithm;
	}

	public void setalgorithm(String algorithm) {
		this.algorithm = algorithm;
	}
	
	public Options[] getoptions() {
		return options;
	}

	public void setoptions(Options [] options) {
		this.options = options;
	}

	@Override
	public String toString() {
		String opts="";
		for (int i=0 ; i<options.length ; i++){
			if(i!=0){
				opts=opts+",";
			}
			opts=opts+options[i]+"\n\t\t";
		}
		return "parameters:{\n\tinPath:" + inPath + ",\n\toutPath:" + outPath + ",\n\talgorithm:" + algorithm +",\n\toptions:[" + opts + "]\n}";
	}
}
