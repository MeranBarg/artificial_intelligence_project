package ai;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class weka {
	public static void main(String[] args) throws Exception {
	
    Instances train;
	BufferedReader buffer = null;
	buffer= new BufferedReader (new FileReader("vote.arff"));
	train = new Instances (buffer);
	train.setClassIndex(train.numAttributes() - 1);
	System.out.println("Number of starting features: "+train.numAttributes());
	buffer.close();
	
	// train classification using WEKA APIs
	NaiveBayes nB = new NaiveBayes();
	nB.buildClassifier(train);
	Evaluation ev = new Evaluation(train);
	ev.crossValidateModel(nB, train, 10, new Random(1));
	System.out.println(ev.toSummaryString("\nThe results are:\n",false));	
	double correct = ev.correct()/(ev.incorrect()+ ev.correct());
	System.out.println(correct);
	
	// Printing initial features 
	System.out.println("\nStarting features");
	for(int i=0; i<train.numAttributes(); i++)
		System.out.println(train.attribute(i));
	
	
	int random = (int )(Math.random() * (train.numAttributes() - 1) + 1);
	String randomString = new String();
	randomString = Integer.toString(random);
	String[] opts = new String[]{"-R", randomString};
	Remove remove = new Remove();
	remove.setOptions(opts);
	remove.setInputFormat(train);
	Instances newData = Filter.useFilter(train,remove);
	train.setClassIndex(newData.numAttributes() - 1);
	System.out.println("Number of starting features: "+newData.numAttributes());
	
	
	
	// train classification using WEKA APIs
	NaiveBayes nB1 = new NaiveBayes();
	nB.buildClassifier(newData);
	Evaluation ev1 = new Evaluation(newData);
	ev1.crossValidateModel(nB1, newData, 10, new Random(1));
	System.out.println(ev1.toSummaryString("\nThe results are:\n",false));		   
	
	// Printing initial features 
	System.out.println("\nStarting features");
	for(int i=0; i<newData.numAttributes(); i++)
		System.out.println(newData.attribute(i));
	
}}
