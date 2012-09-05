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
public class carprobabilityofmissing {
	public static void main(String[] args) throws Exception{

	BufferedReader reader = new BufferedReader(
            new FileReader("car.arff"));
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
File file = new File("tablecar.csv");
Writer output = null;
output = new BufferedWriter(new FileWriter(file));
output.write("probability,auc,correct,fmeasure\n");

Random randomGenerator = new Random();
int numBlock= data.numInstances()*(data.numAttributes()-1);
//Instances mdata = new Instances(data);
for(double prob=0;prob<=1.00;prob=prob+0.02)
{
	Instances mdata = new Instances(data);
	for (int k=0;k<i;k++) 
	{
		if (mdata.instance(k).stringValue(5).equals("low") || mdata.instance(k).stringValue(5).equals("med"))
		{
			float p = randomGenerator.nextFloat();
			if (p<=prob) mdata.instance(k).setMissing(5);
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
	}
}