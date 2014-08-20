package javaHelpers;

import ilpInference.YZPredicted;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import edu.stanford.nlp.util.Triple;
import net.sf.javailp.Linear;
import net.sf.javailp.OptType;
import net.sf.javailp.Problem;
import net.sf.javailp.Result;
import net.sf.javailp.Solver;
import net.sf.javailp.SolverFactory;
import net.sf.javailp.SolverFactoryLpSolve;
import net.sf.javailp.Term;
import javaHelpers.FindMaxViolatorHelperAll.LabelWeights;

public class LossLagrangian1stVer {
	
	static class Region {
		double [] p1;
		double [] p2;
		double [] p3;
		double [] coeff; // alpha, beta, gamma
		
		public void processLine(String line, int var){
			String vals[];
			switch (var){
				// P1
				case 1: 
					p1 = new double[3];
					vals = line.split(" ");
					p1[0] = Double.parseDouble(vals[0]);
					p1[1] = Double.parseDouble(vals[1]);
					p1[2] = Double.parseDouble(vals[2]);
					break;
					
				// P2	
				case 2:
					p2 = new double[3];
					vals = line.split(" ");
					p2[0] = Double.parseDouble(vals[0]);
					p2[1] = Double.parseDouble(vals[1]);
					p2[2] = Double.parseDouble(vals[2]);
					break;
				
				// P3	
				case 3: 
					p3 = new double[3];
					vals = line.split(" ");
					p3[0] = Double.parseDouble(vals[0]);
					p3[1] = Double.parseDouble(vals[1]);
					p3[2] = Double.parseDouble(vals[2]);
					break;
				
				// Coeff	
				case 4:
					coeff = new double[3];
					vals = line.split(" ");
					coeff[0] = Double.parseDouble(vals[0]);
					coeff[1] = Double.parseDouble(vals[1]);
					coeff[2] = Double.parseDouble(vals[2]);
					break;
			}
		}
	}
	
	public static ArrayList<Region> readRegionFile(String regionFile) throws NumberFormatException, IOException{
		ArrayList<Region> regions = new ArrayList<Region>();
		BufferedReader br = new BufferedReader(new FileReader(new File(regionFile)));
		
		int numRegions = Integer.parseInt(br.readLine()); // Number of regions
		
		for(int i = 0 ; i < numRegions; i ++){
			Region r = new Region();
			r.processLine(br.readLine(), 1); // P1
			r.processLine(br.readLine(), 2); // P2
			r.processLine(br.readLine(), 3); // P3
			r.processLine(br.readLine(), 4); // alpha, beta, gamma
			regions.add(r);		
		}
		
		br.close();
		return regions;
	}
	
	public static SolverFactory initializeLPsolver(){
		SolverFactory factory = new SolverFactoryLpSolve();
		factory.setParameter(Solver.VERBOSE, 0);
		factory.setParameter(Solver.TIMEOUT, 100); // set timeout to 100 seconds

		return factory;
		
//		Problem problem = new Problem();
//		
//		Linear objective = new Linear();

	}
	
	public static Linear createObjectiveFunction(double objectiveCoeffs[][], double constant, int datasetSz, int numLabels){
		Linear objective = new Linear();
		
		for(int i = 0; i < datasetSz; i ++){
			for(int l = 1; l <= numLabels; l++){
				String var = new StringBuffer("ytilde_"+i+"_"+l).toString();
				objective.add(objectiveCoeffs[i][l], var);
			}
		}

		return objective;
	}
	
	public static Linear createConstraint(double[][] Constraint_Coeff, int datasetSz, int numLabels){
		Linear constraint = new Linear();
		
		for(int i = 0; i < datasetSz; i ++){
			for(int l = 1; l <= numLabels; l++){
				String var = new StringBuffer("ytilde_"+i+"_"+l).toString();
				constraint.add(Constraint_Coeff[i][l], var);
			}
		}
		
		return constraint;
		
	}
	
	public static ArrayList<YZPredicted> optLossLag(ArrayList<DataItem> dataset, int numPosLabels, 
			ArrayList<Region> regions, double[][] Lambda) throws IOException{
		ArrayList<YZPredicted> YtildeStar= new ArrayList<YZPredicted>();
		
		// For r \in regions
		// 		create and solve the ILP/LP problem 
		//		Return the max value and the corresponding YtildeStar
		
		SolverFactory factory = initializeLPsolver();
				
		for(Region r : regions){
			
			// For every Region r solve the LP / ILP problem
			
			double objectiveCoeffs[][] = new double[dataset.size()][numPosLabels+1]; // N * L matrix of coeff one for each ytilde_i,l variable (NOTE: one extra variable for label since 0th index is nil-label and is not considered
			double constant = 0.0;
			for(int i = 0; i < dataset.size(); i ++){
				int ygold[] = dataset.get(i).ylabel;
				int yi[] = initVec(ygold, numPosLabels+1); //(See prev note for numPoslabels+1)
				for(int l = 1; l <= numPosLabels; l ++){
					objectiveCoeffs[i][l] = Lambda[i][l] 						// Lambda_i,l
									  + (r.coeff[0] * (1 - yi[l])) 		// alpha_r * (1 - y_i,l)
									  - (r.coeff[1] * yi[l]);			// beta_r * y_i,l
					
					constant += r.coeff[1] *  yi[l];					// \sum_i,l beta_r * y_i,l
				}
			}
			constant += r.coeff[2]; // constant += gamma_r
			
			// Construct the objective function
			Linear objective = createObjectiveFunction(objectiveCoeffs, constant, dataset.size(), numPosLabels);
			System.out.println("Constructed the objective function");
			// Constraint 1 : Line AB and point C
			// lp1 = r.p1 (A), lp2 = r.p2 (B);		tript = r.p3 (C)
			double[][] Constraint1_Coeff = computeConstraintCoeff(r.p1, r.p2, r.p3, dataset, numPosLabels);
			Linear constraint1 = createConstraint(Constraint1_Coeff, dataset.size(), numPosLabels);
			System.out.println("Constructed Constraint 1");
			// Constraint 2 : Line BC and point A
			// lp1 = r.p2 (B), lp2 = r.p3 (C); 		tript = r.p1 (A)
			double[][] Constraint2_Coeff = computeConstraintCoeff(r.p2, r.p3, r.p1, dataset, numPosLabels);
			Linear constraint2 = createConstraint(Constraint2_Coeff, dataset.size(), numPosLabels);
			System.out.println("Constructed Constraint 2");
			// Constraint 3 : Line AC and point B
			// lp1 = r.p1 (A), lp2 = r.p3 (C);      tript = r.p2 (B)
			double[][] Constraint3_Coeff = computeConstraintCoeff(r.p1, r.p3, r.p2, dataset, numPosLabels);
			Linear constraint3 = createConstraint(Constraint3_Coeff, dataset.size(), numPosLabels);
			System.out.println("Constructed Constraint 3");
			// Solve the optimisation objective function (as an LP/ILP)
			Problem p = new Problem();
			p.setObjective(objective, OptType.MAX);
			
			p.add(constraint1, ">=", -Constraint1_Coeff[0][0]);
			p.add(constraint2, ">=", -Constraint2_Coeff[0][0]);
			p.add(constraint3, ">=", -Constraint3_Coeff[0][0]);
			
			for(Object var : p.getVariables()){
				p.setVarType(var, Double.class);
				p.setVarUpperBound(var, 1.0);
				p.setVarLowerBound(var, 0.0);
			}
			
			System.out.println("Constructed the LP for the region");
			
			Solver solver = factory.get();
			Result result = solver.solve(p);
			
			System.out.println(result);
		}
		
		return YtildeStar;
	}

	public static double[][] computeConstraintCoeff(double[] A, double[] B, double[] C, ArrayList<DataItem> dataset, int numLabels ){
		
		double varCoeffs[][] = new double[dataset.size()][numLabels+1];
		
		double k1 = ( (B[0] - A[0])*(C[1] - A[1]) ) - ( (B[1] - A[1])*(C[0] - A[0]) ); // (Bx - Ax)(Cy - Ay) - (By - Ay)(Cx - Ax)
		double k2 = ( B[0] - A[0] ); // Bx - Ax
		double k3 = ( B[1] - A[1] ); // By - Ay
		
		// Store the constant in varCoeffs[0][0] which is anyway not used
		varCoeffs[0][0] = 0;
		
		for(int i = 0; i < dataset.size(); i ++){
			int ygold[] = dataset.get(i).ylabel;
			int yi[] = initVec(ygold, numLabels+1); 
			for(int l = 1; l <= numLabels; l ++){
				varCoeffs[i][l] = (-k1*k2*yi[l] - k1*k3*(1 - yi[l])); // -k1*k2*y_i,l - k1*k3*(1-y_i,l) 
				
				varCoeffs[0][0] += k1 * k2 * yi[l]; //Constant in the constraint
			}
		}
		
		varCoeffs[0][0] += (-k1 * k2 * A[1] + k1 * k3 * A[0]); //Constant in the constraint
		
		return varCoeffs;
	}
	
	public static int[] initVec(int ygold[], int sz){
		int yi[] = new int[sz];
		Arrays.fill(yi, 0);
		
		for(int y : ygold)
			yi[y] = 1;

		return yi;
	}
	
	public static void main(String args[]) throws NumberFormatException, IOException{
		ArrayList<Region> regions = readRegionFile("/home/ajay/Research/SVMs/latentssvm_relext.mosek.dd/java/src/regions_coeff_binary.txt");
		
		String datasetFile ="/home/ajay/Research/SVMs/latentssvm_relext.mosek.dd/dataset/reidel_trainSVM.data";
		ArrayList<DataItem> dataset = Utils.populateDataset(datasetFile);
	
		System.out.println("Loaded the dataset!");
		
		double Lambda[][] = new double[dataset.size()][52];

		// init Lambda
		for(int i = 0; i < dataset.size(); i ++){
			for(int j = 1; j < 51; j ++){
				Lambda[i][j] = 0.0;
			}
		}
		
		optLossLag(dataset, 51, regions, Lambda);
		
		/*System.out.println(regions.size());
		for(Region r : regions){
			System.out.println(r.p1[0] + " "+ r.p1[1] + " "+ r.p1[2]);
			System.out.println(r.p2[0] + " "+ r.p2[1] + " "+ r.p2[2]);
			System.out.println(r.p3[0] + " "+ r.p3[1] + " "+ r.p3[2]);
			System.out.println(r.coeff[0] + " " + r.coeff[1] + " " + r.coeff[2]);
		}*/
		
		
	}
}
