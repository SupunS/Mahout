package mahout;

import java.io.Serializable;

import org.apache.mahout.classifier.sgd.L2;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.classifier.sgd.PriorFunction;

public class SerializableLR extends OnlineLogisticRegression implements Serializable {
	private static final long serialVersionUID = 1L;
	
	SerializableLR(final int numCategories, final int numFeatures, final PriorFunction function){
		super.numCategories=numCategories;
		super.prior=function;
	}
}
