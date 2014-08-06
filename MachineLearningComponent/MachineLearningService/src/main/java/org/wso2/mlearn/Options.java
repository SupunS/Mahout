package org.wso2.mlearn;

import javax.xml.bind.annotation.XmlRootElement;

@XmlRootElement(name = "options")
public class Options {

	String option;
	String value;

	public String getoption() {
		return option;
	}
	
	public void setoption(String option) {
		this.option = option;
	}
	
	public String getvalue() {
		return value;
	}
	
	public void setvalue(String value) {
		this.value = value;
	}
	
	public String toString() {
		return "{option:"+option+",value:"+value+"}";
	}
}
