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
public class aucstddev {
	public static void main(String[] args) throws Exception{

	BufferedReader reader = new BufferedReader(
            new FileReader("arrhythmia.arff"));
Instances data = new Instances(reader);
reader.close();
// setting class attribute
data.setClassIndex(data.numAttributes() - 1);

int i=data.numInstances();
int j=data.numAttributes()-1;

File file = new File("tableauc.csv");
Writer output = null;
output = new BufferedWriter(new FileWriter(file));
output.write("%missing,auc,correct,fmeasure,aucsd,correctsd,fmeasuresd\n");

Random randomGenerator = new Random();
int numBlock= data.numInstances()*(data.numAttributes()-1);
//Instances mdata = new Instances(data);
//Instances traindata=null;
//Instances testdata=null;
//System.out.println(data.instance(3).stringValue(1));
for(int perc=0;perc<101;perc=perc+2)
{
	Instances mdata = new Instances(data);
	int numMissing=perc*numBlock/100;
	double y1[]= new double[10];
	double y2[]= new double[10];
	double y3[]= new double[10];
	/*mdata.deleteWithMissing(0);
	System.out.println(mdata.numInstances()+","+data.numInstances());*/
	for (int k=0;k<numMissing;k++) 
	{
		int r = randomGenerator.nextInt(i);
		int c = randomGenerator.nextInt(j);
		mdata.instance(r).setMissing(c);
		mdata.instance(r).setMissing(c);
	}
	for (int p=0;p<10;p++) 
	{
		Instances traindata= mdata.trainCV(10,p);
		Instances testdata= mdata.testCV(10,p);
		
		Classifier cModel = (Classifier)new J48();
		cModel.buildClassifier(traindata);
		Evaluation eTest1 = new Evaluation(testdata);
		eTest1.evaluateModel(cModel, testdata);
	//eTest.crossValidateModel(cModel,mdata,10,mdata.getRandomNumberGenerator(1));
		y1[p]=eTest1.areaUnderROC(0);
		y2[p]=eTest1.correct();
		y3[p]=eTest1.fMeasure(0);
	}
	double auc=0,corr=0,fm=0;
	for(int a=0;a<10;a++)
	{
	auc+=y1[a];
	corr+=y2[a];
	fm+=y3[a];
	}
	
	double aucsd=StandardDeviation(y1,10);
	double corrsd=StandardDeviation(y2,10);
	double fmsd=StandardDeviation(y3,10);
	// plot std dev in same graph using bars
	output.write(perc+","+auc/10+","+corr+","+fm/10+","+aucsd+","+corrsd+","+fmsd+"\n");
	//mdata=data;
}
output.close();
	}
	public static double StandardDeviation(double[] values,int NumAmount)
	 
    {
		double sum=0,mean=0,sq_diff_sum=0;
        for(int i = 0; i < NumAmount; i++)
        {
        sum += values[i];
        mean = sum / NumAmount;
        sq_diff_sum = 0;
        }
 
        for(int i = 0; i <NumAmount ; ++i)
        {
           double diff = values[i] - mean;
        sq_diff_sum += diff * diff;
 
        }
        double deviance= sq_diff_sum / NumAmount;
        return Math.sqrt(deviance);
    }
	
}