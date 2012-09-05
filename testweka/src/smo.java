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
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.misc.VFI;
import weka.classifiers.trees.J48;
public class smo {
	public static void main(String[] args) throws Exception{

	BufferedReader reader = new BufferedReader(
            new FileReader("splice.arff"));
Instances data = new Instances(reader);
reader.close();
// setting class attribute
data.setClassIndex(data.numAttributes() - 1);

int i=data.numInstances();
int j=data.numAttributes()-1;


File file = new File("tablesmo.csv");
Writer output = null;
output = new BufferedWriter(new FileWriter(file));
output.write("%missing,auc1,correct1,fmeasure1,auc2,correct2,fmeasure2,auc3,correct3,fmeasure3\n");

Random randomGenerator = new Random();
int numBlock= data.numInstances()/2;
data.randomize(randomGenerator);
double num0=0,num1=0,num2=0;
/*mdata.instance(0).setMissing(0);
mdata.deleteWithMissing(0);
System.out.println(mdata.numInstances()+","+data.numInstances());*/
//Instances traindata=null;
//Instances testdata=null;
//System.out.println(data.instance(3).stringValue(1));
for(int perc=10;perc<101;perc=perc+10)
{
	Instances mdata = new Instances(data);
	int numMissing=perc*numBlock/100;
	double y11[]= new double[2];
	double y21[]= new double[2];
	double y31[]= new double[2];
	double y12[]= new double[2];
	double y22[]= new double[2];
	double y32[]= new double[2];
	double y13[]= new double[2];
	double y23[]= new double[2];
	double y33[]= new double[2];
	for (int p=0;p<2;p++) 
	{
		Instances traindata= mdata.trainCV(2,p);
		Instances testdata= mdata.testCV(2,p);
		num0=0;num1=0;num2=0;
		for (int t=0;t<numBlock;t++)
		{
			if (traindata.instance(t).classValue()==0) num0++;
		  	if (traindata.instance(t).classValue()==1) num1++;
		  	if (traindata.instance(t).classValue()==2) num2++;
		}
		//System.out.println(mdata.instance(1000).classValue());
		Instances trainwithmissing=new Instances(traindata);
		Instances testwithmissing=new Instances(testdata);
		for (int q=0;q<j;q++)
		{
			for (int k=0;k<numBlock;k++) 
			{	
				float toss = randomGenerator.nextFloat();//System.out.println(toss);
				if (toss<=(float)perc/100)
				{
					trainwithmissing.instance(k).setMissing(q);
					testwithmissing.instance(k).setMissing(q);
				}
			}	
		}
		//trainwithmissing.deleteWithMissing(0);System.out.println(traindata.numInstances()+","+trainwithmissing.numInstances());
		Classifier cModel = (Classifier)new SMO(); //try for different classifiers and datasets
		cModel.buildClassifier(trainwithmissing);
		Evaluation eTest1 = new Evaluation(trainwithmissing);
		eTest1.evaluateModel(cModel, testdata);
	//eTest.crossValidateModel(cModel,mdata,10,mdata.getRandomNumberGenerator(1));
		y11[p]=num0/numBlock*eTest1.areaUnderROC(0)+num1/numBlock*eTest1.areaUnderROC(1)+num2/numBlock*eTest1.areaUnderROC(2);//System.out.println(y11[p]);
		y21[p]=eTest1.correct();
		y31[p]=num0/numBlock*eTest1.fMeasure(0)+num1/numBlock*eTest1.fMeasure(1)+num2/numBlock*eTest1.fMeasure(2);
		
		Classifier cModel2 = (Classifier)new SMO();
		cModel2.buildClassifier(traindata);
		Evaluation eTest2 = new Evaluation(traindata);
		eTest2.evaluateModel(cModel2, testwithmissing);
		y12[p]=num0/numBlock*eTest2.areaUnderROC(0)+num1/numBlock*eTest2.areaUnderROC(1)+num2/numBlock*eTest2.areaUnderROC(2);
		y22[p]=eTest2.correct();
		y32[p]=num0/numBlock*eTest2.fMeasure(0)+num1/numBlock*eTest2.fMeasure(1)+num2/numBlock*eTest2.fMeasure(2);
	
		Classifier cModel3 = (Classifier)new SMO();
		cModel3.buildClassifier(trainwithmissing);
		Evaluation eTest3 = new Evaluation(trainwithmissing);
		eTest3.evaluateModel(cModel3, testwithmissing);
		y13[p]=num0/numBlock*eTest3.areaUnderROC(0)+num1/numBlock*eTest3.areaUnderROC(1)+num2/numBlock*eTest3.areaUnderROC(2);
		y23[p]=eTest3.correct();
		y33[p]=num0/numBlock*eTest3.fMeasure(0)+num1/numBlock*eTest3.fMeasure(1)+num2/numBlock*eTest3.fMeasure(2);
		//System.out.println(num0+","+num1+","+num2+"\n");
	}
	double auc1=(y11[0]+y11[1])/2;
	double auc2=(y12[0]+y12[1])/2;
	double auc3=(y13[0]+y13[1])/2;
	double corr1=(y21[0]+y21[1])/i;
	double corr2=(y22[0]+y22[1])/i;
	double corr3=(y23[0]+y23[1])/i;
	double fm1=(y31[0]+y31[1])/2;
	double fm2=(y32[0]+y32[1])/2;
	double fm3=(y33[0]+y33[1])/2;
	output.write(perc+","+auc1+","+corr1+","+fm1+","+auc2+","+corr2+","+fm2+","+auc3+","+corr3+","+fm3+"\n");//System.out.println(num0);
	//mdata=data;
	
}
output.close();
	}
}