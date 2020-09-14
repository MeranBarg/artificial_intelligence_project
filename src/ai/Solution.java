package ai;


import weka.core.Instances;

public class Solution {
	double accuracy;
	Instances newSol;
	public Solution(double accuracy, Instances newSol) {
		this.accuracy = accuracy;
		this.newSol = newSol;
	}
	public double getAccuracy() {
		return accuracy;
	}
	public void setAccuracy(double accuracy) {
		this.accuracy = accuracy;
	}
	public Instances getNewSol() {
		return newSol;
	}
	public void setNewSol(Instances newSol) {
		this.newSol = newSol;
	}
}
