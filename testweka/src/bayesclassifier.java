//import weka.core.Instance;
import weka.core.Instances;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.Writer;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.misc.VFI;
import weka.classifiers.trees.J48;
//import weka.gui.beans.InstanceEvent;
//import weka.filters.Filter;
import weka.core.SparseInstance;
public class bayesclassifier {
	public static void main(String[] args) throws Exception{

	BufferedReader reader = new BufferedReader(
            new FileReader("cmc.arff"));
Instances data = new Instances(reader);
reader.close();
// setting class attribute
data.setClassIndex(data.numAttributes() - 1);
//Removes all instances with a missing class value
//data.deleteWithMissingClass();
int i=data.numInstances();
int j=data.numAttributes()-1;
/*for(int i=0;i<data.numInstances();i++)
{
	data.instance(i).setMissing(i%j);
}*/
File file = new File("tablecmc.csv");
Writer output = null;
output = new BufferedWriter(new FileWriter(file));
output.write("probability,auc,correct,fmeasure\n");

Random randomGenerator = new Random();
int numBlock= data.numInstances()*(data.numAttributes()-1);
//Instances mdata = new Instances(data);
//System.out.println(data.instance(3).stringValue(1));
for(double prob=0;prob<=1.0;prob=prob+0.02)
{
	Instances mdata = new Instances(data);
	for (int k=0;k<i;k++) 
	{
		if (data.instance(k).stringValue(1).equals("1") || data.instance(k).stringValue(1).equals("2"))
		{
			float p = randomGenerator.nextFloat();
			if (p<=prob) mdata.instance(k).setMissing(1);
		}
	}
	Classifier cModel = (Classifier)new J48();
	Evaluation eTest = new Evaluation(mdata);
	eTest.crossValidateModel(cModel,mdata,10,mdata.getRandomNumberGenerator(1));
	double y1=eTest.areaUnderROC(0);
	double y2=eTest.correct();
	double y3=eTest.fMeasure(0);
	output.write(prob+","+y1+","+y2+","+y3+"\n");
	//mdata=data;
}
output.close();
//System.out.println(sum);

//cModel.buildClassifier(data);
//Test the model

//eTest.evaluateModel(cModel, data);
//String strSummary = eTest.toSummaryString();

//plot these with percentage of missing values 
//System.out.println(strSummary);
// Get the confusion matrix
/*double[][] cmMatrix = eTest.confusionMatrix();
for(int row_i=0; row_i<cmMatrix.length; row_i++)
{
	for(int col_i=0; col_i<cmMatrix.length; col_i++)
	{
		System.out.print(cmMatrix[row_i][col_i]);
		System.out.print("|");
	}
	System.out.println();
}*/

	}
}
