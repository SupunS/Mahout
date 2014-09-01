package mahout;

import org.encog.Encog;
import org.encog.app.analyst.AnalystFileFormat;
import org.encog.app.analyst.EncogAnalyst;
import org.encog.app.analyst.csv.normalize.AnalystNormalizeCSV;
import org.encog.app.analyst.script.normalize.AnalystField;
import org.encog.app.analyst.wizard.AnalystWizard;
import org.encog.util.csv.CSVFormat;

import java.io.File;

public class Normalize {
    public static void dumpFieldInfo(EncogAnalyst analyst) {

        for (AnalystField field : analyst.getScript().getNormalize()
                .getNormalizedFields()) {
            StringBuilder line = new StringBuilder();
            line.append(field.getName());
            line.append(",action=");
            line.append(field.getAction());
            line.append(",min=");
            line.append(field.getActualLow());
            line.append(",max=");
            line.append(field.getActualHigh());
            System.out.println(line.toString());
        }
    }

    public static String getNormalized(final String input) {
    	File sourceFile = new File(input);
    	
    	File targetFile = new File(input.substring(0, input.lastIndexOf('/'))+"Normalized.csv");
    	System.out.println(sourceFile.getPath());
        EncogAnalyst analyst = new EncogAnalyst();
        AnalystWizard wizard = new AnalystWizard(analyst);
        wizard.wizard(sourceFile, true, AnalystFileFormat.DECPNT_COMMA);
        dumpFieldInfo(analyst);
        final AnalystNormalizeCSV norm = new AnalystNormalizeCSV();
        norm.analyze(sourceFile, true, CSVFormat.ENGLISH, analyst);
        norm.setProduceOutputHeaders(true);
        norm.normalize(targetFile);
        Encog.getInstance().shutdown();
        return targetFile.getPath();
    }
}
