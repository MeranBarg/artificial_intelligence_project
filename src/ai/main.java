package ai;

//The packages below are used for reading the weka files!
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

//The packages below are used to evaluate the solution
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
//The packages below are used to select certain features to add or remove

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
public class main {
	public static void main(String[] args) throws  Exception{
	//As a first step, the weka file must be read for training
    Instances train;
			BufferedReader buffer = null;
			buffer= new BufferedReader (new FileReader("vote.arff"));
			train = new Instances (buffer);
			buffer.close();
			String optimalSolution = generateInitialNode(train.numAttributes());

			System.out.println("INITIAL FEATURES: ");
			String[] opts = new String[]{"-R", optimalSolution};
			Remove remove = new Remove();
			remove.setOptions(opts);
			remove.setInputFormat(train);
			Instances newData = Filter.useFilter(train,remove);
			train.setClassIndex(newData.numAttributes() - 1);
			System.out.println("Number of initial features: "+newData.numAttributes());	
			NaiveBayes nB = new NaiveBayes();
			nB.buildClassifier(train);
			Evaluation ev = new Evaluation(train);
			ev.crossValidateModel(nB, train, 10, new Random(1));

			Scanner input = new Scanner(System.in); 
			System.out.println("Please enter the temperature: ");
			double temperature = input.nextDouble();
			System.out.println("Please enter the cooling rate: ");
			double rate = input.nextDouble();

	
		//	System.out.print(getEvaluation(train));
			System.out.println("HILL CLIMBING: ");
			HCAlgorithm(optimalSolution, train);
			System.out.println("SIMULATED ANNEALING: ");
			SAAlgorithm(optimalSolution,train,temperature,rate);
}
	
	
	public static double getEvaluation(Instances train) throws Exception{
		train.setClassIndex(train.numAttributes() - 1);
		NaiveBayes nB = new NaiveBayes();
		nB.buildClassifier(train);
		Evaluation ev = new Evaluation(train);
		ev.crossValidateModel(nB, train, 10, new Random(1));
		return ev.pctCorrect();
		

	}
	
	public static double selectFeatures(String featureOptions, Instances train)throws Exception{

		Remove remove = new Remove();
		String[] opts = new String[]{"-R", featureOptions};	
		remove.setOptions(opts);
		remove.setInputFormat(train);
		Instances newSolution = Filter.useFilter(train,remove);
		newSolution.setClassIndex(newSolution.numAttributes() - 1);
		
		return getEvaluation(newSolution);
	}
	
	public static String generateInitialNode ( int featureNum){
	 	ArrayList<Integer> delFeatures = new ArrayList<>();
	 	Random random = new Random();
		int numOfFeaturesToDelete = random.nextInt(featureNum-1) + 1;
		
	
		String deselectedFeatures ="";
		int initialDelete = random.nextInt(featureNum -1) + 1;

		delFeatures.add(initialDelete);
		String initialString = Integer.toString(initialDelete);
		 deselectedFeatures = deselectedFeatures.concat(initialString);
		
		
		for (int i =0;i<numOfFeaturesToDelete-1;i++)
		{
			int successiveDelete = random.nextInt(featureNum-1) + 1;
			if (!delFeatures.contains(successiveDelete)){
				delFeatures.add(successiveDelete);
				 String successiveString = Integer.toString(successiveDelete);
				 successiveString =","+successiveString;
				 deselectedFeatures = deselectedFeatures.concat(successiveString);
				
			}
			
		}

		return deselectedFeatures;	  
  }
	public static String generateNextNode(int featureNum, String currentNode){
		//In this function we may want to add or delete columns from the current node to possibly generate a node with
		//a higher accuracy!
	 	ArrayList<String> delFeatures = new ArrayList<>();
		Random random = new Random();
		String nextNode="";
		//WE NEED TO SPLIT THE STRING TO REMOVE THE COMAS
		String[] splittedString = currentNode.split(",");
		//HOWEVER, WE MAY WANT TO REMOVE OR ADD FEATURES
		int flag = 0; //if flag == 1 then we want to add features to the current node and remove if else
		int length = splittedString.length;
		for(int i = 0; i<length;i++){
			delFeatures.add(splittedString[i]);
		}
		
		if(length == featureNum-1){
			flag=0;
		}
		else if(length< featureNum-1){
			flag=1;
		}
		
		if(flag == 1){
			int numOfFeaturesToDelete = random.nextInt(4) + 1;
			while( (numOfFeaturesToDelete + (length)) > featureNum -2){
				numOfFeaturesToDelete--;
			}
			for (int i =0;i<numOfFeaturesToDelete-1;i++){
				int successiveDelete = random.nextInt(featureNum-1) + 1;
				String successiveString = Integer.toString(successiveDelete);
				if(!delFeatures.contains(successiveString)){
					successiveString= ","+successiveString;
					nextNode = currentNode.concat(successiveString);
				}
				
			}
			
		}
		else{
			int reAddFeature = random.nextInt(length);
			int i=0;
			if(reAddFeature == 0){
				nextNode = splittedString[1];
				i=2;
			}
			else{
				nextNode = splittedString[0];
				i = 1;
			}
			for(int j = i; j<length;j++){
				if(j==reAddFeature)
					continue;
				nextNode=nextNode+","+splittedString[j]; 

				
			}
		}
		
		
		
	return nextNode;	
	}
public static void HCAlgorithm(String optimalSolution,Instances train)throws Exception{
	//As a first step, we will initialize a solution by calling the function "generateInitialNode" and assuming
	//that it's the optimal solution.

	int featureNum=train.numAttributes(),stuck=0,generationNum=0;
	double evOptimalSolution = selectFeatures(optimalSolution,train);
	boolean flag = false;
	
	//Now we move on to the butter of the search algorithm!
	do{
		String newSolution = generateNextNode(featureNum,optimalSolution);
		double evNewSolution = selectFeatures(newSolution,train);
		//NOW WE START TO COMPARE
		generationNum++;
		if(evNewSolution>evOptimalSolution){
		evOptimalSolution = evNewSolution;
		optimalSolution = newSolution;
		stuck = 0; //NOT STUCK AT A LOCAL OPTIMUM
		//WE'RE HUNGRY FOR BETTER SOLUTIONS!
		flag = true;
		}
		
		//The next state is no better!
		else{
			if(stuck==70){
				flag = false;
			}
			else{
				flag = true;
				stuck++;
			}
		}
		
		String evaluation = String.format("%.1f",evOptimalSolution);
		System.out.println("Iteration:"+generationNum+"	"+"Evaluation--->"+evaluation+"%"); 
		//System.out.println(evaluation);
		
	
}while(flag);
	String[] opts = new String[]{"-R", optimalSolution};
	Remove remove = new Remove();
	remove.setOptions(opts);
	remove.setInputFormat(train);
	Instances newData = Filter.useFilter(train,remove);
	train.setClassIndex(newData.numAttributes() - 1);
	System.out.println("Number of final features: "+newData.numAttributes());	
	NaiveBayes nB = new NaiveBayes();
	nB.buildClassifier(train);
	Evaluation ev = new Evaluation(train);
	ev.crossValidateModel(nB, train, 10, new Random(1));
	System.out.println(ev.toSummaryString("\nThe results are:\n",false));	

	
}
public static void SAAlgorithm(String optimalSolution, Instances train, double temperature, double coolingRate)throws Exception{
	//As a first step, we will initialize a solution by calling the function "generateInitialNode" and assuming
	//that it's the optimal solution.

	int featureNum=train.numAttributes(),stuck=0,generationNum=0;
	double evOptimalSolution = selectFeatures(optimalSolution,train);
	
	boolean flag = true;
	double variableTemp = temperature;
	
	//Now we move on to the butter of the search algorithm!
	while(variableTemp>1 && flag==true){
		generationNum++;
		String newSolution = generateNextNode(featureNum,optimalSolution);
		double evNewSolution = selectFeatures(newSolution,train);
		//NOW WE START TO COMPARE
		//generationNum++;
		if(evNewSolution>evOptimalSolution){
		evOptimalSolution = evNewSolution;
		optimalSolution = newSolution;
		//stuck = 0; //NOT STUCK AT A LOCAL OPTIMUM
		//WE'RE HUNGRY FOR MORE BETTER SOLUTIONS!
		flag = true;
		}
		
		//The next state is probably better or probably not!
		else{
			Random random = new Random();
			double diff = evOptimalSolution - evNewSolution;
			double acceptanceProbability = Math.exp(-diff/variableTemp);
			double randomDouble = (0.01)*random.nextInt(10000);
			if(acceptanceProbability>randomDouble){
				evOptimalSolution = evNewSolution;
				optimalSolution = newSolution;	
				flag=true;
			}
			else if(stuck==70){
				flag = false;
				
			}
			else{
				flag = true;
				stuck++;
				
			}
		}
		
		String evaluation = String.format("%.1f",evOptimalSolution);
		//System.out.println(evaluation);

		System.out.println("Iteration:"+generationNum+"	"+"Evaluation--->"+evaluation+"%"); 

		variableTemp= (1-coolingRate)*variableTemp;
}
	String[] opts = new String[]{"-R", optimalSolution};
	Remove remove = new Remove();
	remove.setOptions(opts);
	remove.setInputFormat(train);
	Instances newData = Filter.useFilter(train,remove);
	train.setClassIndex(newData.numAttributes() - 1);
	System.out.println("Number of final features: "+newData.numAttributes());	
	NaiveBayes nB = new NaiveBayes();
	nB.buildClassifier(train);
	Evaluation ev = new Evaluation(train);
	ev.crossValidateModel(nB, train, 10, new Random(1));
	System.out.println(ev.toSummaryString("\nThe results are:\n",false));	
}
}
