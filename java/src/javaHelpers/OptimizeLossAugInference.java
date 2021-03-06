package javaHelpers;

import ilog.concert.IloException;
import ilpInference.InferenceWrappers;
import ilpInference.YZPredicted;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.stats.Counter;
import javaHelpers.FindMaxViolatorHelperAll.LabelWeights;
import javaHelpers.LossLagrangian.Region;

public class OptimizeLossAugInference {

	static int MAX_ITERS_SUB_DESCENT = 10;
	
	public static ArrayList<YZPredicted> optimizeLossAugInferenceDD(ArrayList<DataItem> dataset, 
			LabelWeights [] zWeights, double simFracParam) throws IOException, IloException{

		// Initialize t = 0 and Lambda^0
		// repeat
		//		~Y* = optLossLag(Lambda, Y)
		//		~Y'* = optModelLag(Lambda, X)
		//		if (~Y* == ~Y'*)
		//			return ~Y*
		//		
		//		for(i = 1 to N) {
		//			for(l = 1 to L) {
		//				lambda^(t+1)_i (l) = lambda^t_i (l) - eta^t (~y*_{i,l} - ~y'*_{i,l})
		// until some stopping condition is met
		// return ~Y*

		//TODO: check if zWeights.length also includes the nil label
		
		ArrayList<YZPredicted> YtildeStar;
		ArrayList<YZPredicted> YtildeDashStar;
		double Lambda[][] = new double[dataset.size()][zWeights.length];
		int t = 0;

		// init Lambda
		for(int i = 0; i < dataset.size(); i ++){
			for(int l = 1; l < zWeights.length; l ++){
				Lambda[i][l] = 0.0;
			}
		}


		ArrayList<Region> regions = LossLagrangian.readRegionFile("regions_coeff_binary.txt");
		while(true){

			long startiter = System.currentTimeMillis();
			t++;

			//******* Loss Lag *****************************************************
			YtildeStar = LossLagrangian.optLossLag(dataset, zWeights.length-1, regions, Lambda);
			
			long endlosslag = System.currentTimeMillis();
			double timelosslag = (double)(endlosslag - startiter) / 1000.0;
			System.out.println("Ajay: Time taken loss lag : " + timelosslag + " s.");
			
			/// ****** Model Lag ***************************************************
			//YtildeDashStar = ModelLagrangian.optModelLag(dataset, zWeights, Lambda);
			YtildeDashStar = ModelLagrangian.optModelLag_lpsolve(dataset, zWeights, Lambda);

			long endmodellag = System.currentTimeMillis();
			double timemodellag = (double) (endmodellag - endlosslag) / 1000.0;
			System.out.println("Ajay: Time taken model lag : " + timemodellag + " s.");
			
			double fracSame = fractionSame(YtildeStar, YtildeDashStar);
			
			System.out.println("-------------------------------------");
			System.out.println("Subgradient-descent: In Iteration " + t + ": Between Ytilde and YtildeStar, fraction of same labels is : " + fracSame);
			System.out.println("-------------------------------------");
			
			// Stopping condition for the subgradient descent algorithm
			if(fracSame > simFracParam || t > MAX_ITERS_SUB_DESCENT) { // || both YtildeStar and YtildeDashStar are equal
				System.out.println("Met the stopping criterion. !!");
				System.out.println("Fraction of same labels is : " + fracSame + "; Num of iters completed : " + t);
				break; 
			}
				

			double eta = 1.0 / Math.sqrt(t);
			for(int i = 0; i < dataset.size(); i ++){
				for(int l = 1; l < zWeights.length; l ++){

					double ystar_il = YtildeStar.get(i).getYPredicted().getCount(l);
					double ydashstar_il = YtildeDashStar.get(i).getYPredicted().getCount(l);		

					Lambda[i][l] = Lambda[i][l] -  ( eta * (ystar_il - ydashstar_il));
				}
			}
			
			long endIter = System.currentTimeMillis();
			double timeiter = (double)(endIter - startiter)/ 1000.0;
			System.out.println(" Time taken for iteration " + t + " :" + timeiter + " s.");
		}

		return YtildeDashStar;

	}
	
	static double fractionSame(ArrayList<YZPredicted> YtildeStar, ArrayList<YZPredicted> YtildeDashStar){
		
		double fracSame = 0.0;
		
		if(YtildeStar.size() != YtildeDashStar.size()){
			System.out.println("SOME ERROR!!!! THE SIZES of YtildeStar and YtildeDashStar are not the same");
			System.exit(0);
		}
		
		for(int i = 0; i < YtildeStar.size(); i ++){
			YZPredicted ytilde = YtildeStar.get(i);
			YZPredicted ytildedash = YtildeDashStar.get(i);
			if(isSame(ytilde, ytildedash))
				fracSame++;
		}
			
		
		fracSame = fracSame / YtildeStar.size();
		
		return fracSame;
		
	}
	
	static boolean isSame(YZPredicted ytilde, YZPredicted ytildedash){

		Set<Integer> y =  ytilde.getYPredicted().keySet();
		Set<Integer> ydash = ytildedash.getYPredicted().keySet();

		if(y.size() != ydash.size()) return false;

		for(Integer l: ydash) {
			if(! y.contains(l)) {
				return false;
			}
		}

		return true;
	}
	
	public static void main(String args[]) throws NumberFormatException, IOException, IloException{
		
		String currentParametersFile = args[0];
		String datasetFile = args[1];
		double simFracParam = Double.parseDouble(args[2]);
		
		LabelWeights [] zWeights = Utils.initializeLabelWeights(currentParametersFile);
		ArrayList<DataItem> dataset = Utils.populateDataset(datasetFile);
		
		ArrayList<YZPredicted> yzPredictedAll = optimizeLossAugInferenceDD(dataset, zWeights, simFracParam);
	
	}
	

}
