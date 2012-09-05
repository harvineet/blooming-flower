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
import weka.classifiers.trees.J48;
//import weka.gui.beans.InstanceEvent;
//import weka.filters.Filter;
import weka.core.SparseInstance;
public class treeclassifier {
	public static void main(String[] args) throws Exception{

	BufferedReader reader = new BufferedReader(
            new FileReader("pendigits.arff"));
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
File file = new File("tablej48.csv");
Writer output = null;
output = new BufferedWriter(new FileWriter(file));
output.write("%missing,auc,correct,fmeasure\n");

int numBlock= data.numInstances();
//Instances mdata = new Instances(data);
Random randomGenerator = new Random();
int c = randomGenerator.nextInt(j);
for(int perc=0;perc<101;perc=perc+5)
{
	Instances mdata = new Instances(data);
	int numMissing=perc*numBlock/100;
	for (int k=0;k<numMissing;k++) 
	{
		int r = randomGenerator.nextInt(i);
		
		mdata.instance(r).setMissing(c);
	}
	Classifier cModel = (Classifier)new J48();
	Evaluation eTest = new Evaluation(mdata);
	eTest.crossValidateModel(cModel,mdata,10,mdata.getRandomNumberGenerator(1));
	double y1=eTest.areaUnderROC(0);
	double y2=eTest.correct();
	double y3=eTest.fMeasure(0);
	output.write(perc+","+y1+","+y2+","+y3+"\n");
	//mdata=data;
}
output.close();

	}
}
