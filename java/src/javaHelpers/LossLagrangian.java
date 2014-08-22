package javaHelpers;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import ilog.concert.IloModeler;
import ilog.concert.IloNumVar;
import ilog.concert.IloNumVarType;
import ilog.cplex.IloCplex;
import ilpInference.YZPredicted;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import edu.stanford.nlp.util.Pair;
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

public class LossLagrangian {
	
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
		
		/*System.out.println(regions.size());
		for(Region r : regions){
			System.out.println(r.p1[0] + " "+ r.p1[1] + " "+ r.p1[2]);
			System.out.println(r.p2[0] + " "+ r.p2[1] + " "+ r.p2[2]);
			System.out.println(r.p3[0] + " "+ r.p3[1] + " "+ r.p3[2]);
			System.out.println(r.coeff[0] + " " + r.coeff[1] + " " + r.coeff[2]);
		}*/
		
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
	
	static ArrayList<IloNumVar[]> initVariablesForLP(IloCplex model, int datasetsize, int numPosLabels) throws IloException{
		
		ArrayList<IloNumVar[]> variables = new ArrayList<IloNumVar[]>();
		
		for(int i = 0; i < datasetsize; i ++){
			IloNumVar[] vars_i = model.numVarArray(numPosLabels, 0, 1);
			variables.add(vars_i);
		}
		
		return variables;
		
	}
	
	static double buildCplexModel(IloCplex cplexModel, ArrayList<IloNumVar[]> variables, Region r,
			ArrayList<DataItem> dataset, int numPosLabels, double[][] Lambda) throws IloException{
		
		double A[] = r.p1;
		double B[] = r.p2;
		double C[] = r.p3;
		double alpha_r = r.coeff[0];
		double beta_r = r.coeff[1];
		double gamma_r = r.coeff[2];

		double regionConstant = gamma_r;
		
		IloLinearNumExpr objective = cplexModel.linearNumExpr();
		IloLinearNumExpr constraint1 = cplexModel.linearNumExpr();
		double cons_constraint1 = 0.0;
		IloLinearNumExpr constraint2 = cplexModel.linearNumExpr();
		double cons_constraint2 = 0.0;
		IloLinearNumExpr constraint3 = cplexModel.linearNumExpr();
		double cons_constraint3 = 0.0;

		for(int i = 0; i < dataset.size(); i ++){ // For every datum in training dataset

			int yi[] = initVec(dataset.get(i).ylabel, numPosLabels);
			for(int l = 1; l <= numPosLabels; l ++){ // For every label-id

				////////////////////////////// ~y_i,l //////////////////////////////////////////////////

				double coeff = Lambda[i][l] 			// Lambda_i,l
						+ (alpha_r * (1 - yi[l])) 	// alpha_r * (1 - y_i,l)
						- (beta_r * yi[l]);			// beta_r * y_i,l

				// Add ~y_i,l to the objective function
				//String varName = new StringBuffer("ytilde_"+i+"_"+l).toString();
				//IloNumVar var = model.numVar(0, 1, varType, varName);
				IloNumVar var = variables.get(i)[l-1];
				objective.addTerm(coeff, var);

				regionConstant += r.coeff[1] *  yi[l];					// \sum_i,l beta_r * y_i,l

				// Add ~y_i,l to the first constraint
				Pair<Double, Double> c1_coeffs = computeConstraintCoeff(A, B, C, yi[l]);
				constraint1.addTerm(c1_coeffs.first(), var);
				cons_constraint1 += c1_coeffs.second();

				// Add ~y_i,l to the second constraint
				Pair<Double, Double> c2_coeffs = computeConstraintCoeff(B, C, A, yi[l]);
				constraint2.addTerm(c2_coeffs.first(), var);
				cons_constraint2 += c2_coeffs.second();

				// Add ~y_i,l to the third constraint
				Pair<Double, Double> c3_coeffs = computeConstraintCoeff(A, C, B, yi[l]);
				constraint3.addTerm(c3_coeffs.first(), var);
				cons_constraint3 += c3_coeffs.second();

				////////////////////////////// ~y_i,l //////////////////////////////////////////////////

			} // END: For every label-id

		} // END: For every datum in training dataset

		Triple<Double, Double, Double> c1 =  compute_k1k2k3 (A, B, C);
		cons_constraint1 += ((-c1.first() * c1.second() * A[1]) + (c1.first() * c1.third() * A[0]));

		Triple<Double, Double, Double> c2 =  compute_k1k2k3 (B, C, A);
		cons_constraint2 += ((-c2.first() * c2.second() * B[1]) + (c2.first() * c2.third() * B[0]));

		Triple<Double, Double, Double> c3 =  compute_k1k2k3 (A, C, B);
		cons_constraint3 += ((-c3.first() * c3.second() * A[1]) + (c3.first() * c3.third() * A[0]));

		// Create the LP/ILP problem object and add the objective and constraints to it. 
		cplexModel.addMaximize(objective);
		cplexModel.addGe(constraint1, -cons_constraint1);
		cplexModel.addGe(constraint2, -cons_constraint2);
		cplexModel.addGe(constraint3, -cons_constraint3);

		return regionConstant;
	}
	
	public static ArrayList<YZPredicted> optLossLag(ArrayList<DataItem> dataset, int numPosLabels, 
			ArrayList<Region> regions, double[][] Lambda) throws IOException, IloException{
		ArrayList<YZPredicted> YtildeStar= new ArrayList<YZPredicted>();
		
		long startFunction = System.currentTimeMillis();
		
		double maxObjectiveValueAllRegions = Double.NEGATIVE_INFINITY;
		int maxRegion = -1;
		
		for(int rid = 0; rid < regions.size(); rid ++){ // For each region
			
			Region r = regions.get(rid);
			IloCplex cplexModel = new IloCplex();
			IloNumVarType varType   = IloNumVarType.Float; 
			
			long startilpsol = System.currentTimeMillis();
			
			// Build the LP problem model
			ArrayList<IloNumVar[]> variables = initVariablesForLP(cplexModel,  dataset.size(), numPosLabels);
			double regionObjValue = buildCplexModel(cplexModel, variables, r, dataset, numPosLabels, Lambda); // return value is constant for a region in the LP formula. Added to the solved value of max objective
			
			// Solve the LP
			if ( cplexModel.solve() ) {
				
				regionObjValue += cplexModel.getObjValue();
				
				System.out.println("Solution status = " + cplexModel.getStatus());
				System.out.println(" cost = " + regionObjValue);
				
				if(regionObjValue > maxObjectiveValueAllRegions){
					maxRegion = rid;
					maxObjectiveValueAllRegions = regionObjValue;
				}
					
			}
			
			long endilpsol = System.currentTimeMillis();
			double time = (double)(endilpsol - startilpsol) / 1000.0;
			System.out.println("Log: Total time for the LP problem : " + time + " s");
						
			cplexModel.end();

		} // END: For each region
			
		System.out.println("Max region : " + maxRegion + "; Max objective : " + maxObjectiveValueAllRegions);

		System.out.println("Solving LP for max region");
		Region r = regions.get(maxRegion);
		IloCplex cplexModel = new IloCplex();
		IloNumVarType varType   = IloNumVarType.Float;

		// Build the LP problem model for the max region 
		ArrayList<IloNumVar[]> variables = initVariablesForLP(cplexModel,  dataset.size(), numPosLabels);
		buildCplexModel(cplexModel, variables, r, dataset, numPosLabels, Lambda);
		
		// Solve the cplex model
		cplexModel.solve();
		
		// Update the variables to be returned
		for (int i = 0; i < variables.size(); i++) {
			double ytildestar_i_approx[] =  cplexModel.getValues(variables.get(i));
			addToYtildeStar(YtildeStar, ytildestar_i_approx);
			
		}
		
		cplexModel.end();
		
		long endFunction = System.currentTimeMillis();
		double FunctionTime = (double)(endFunction - startFunction) / 1000.0; 
		System.out.println("Log: Total time in function optLossLag - " + FunctionTime + " s");
		
		return YtildeStar;
	}
	
	static void addToYtildeStar(ArrayList<YZPredicted> YtildeStar, double ytildestar_i_approx[]){
		YZPredicted ytildestar_i = new YZPredicted(0);
		
		for(int i = 0; i < ytildestar_i_approx.length; i ++){
			double ytilde_l = ytildestar_i_approx[i];
			if(ytilde_l > 0.5)
				ytildestar_i.yPredicted.incrementCount(i+1);
				
		}
		
		YtildeStar.add(ytildestar_i);
		return;
	}

	static void writeToFile (Linear objective, 
			Linear constraint1,double cons_constraint1, 
			Linear constraint2, double cons_constraint2, 
			Linear constraint3, double cons_constraint3,
			String type) throws IOException {
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("optimization.txt")));
		
		Term t = objective.get(0);
		bw.write("Max: " + t.getCoefficient() + " " + t.getVariable() );
		for(int i = 1; i < objective.size(); i ++) {
			t = objective.get(i);
			bw.write(" + " + t.getCoefficient() + " " + t.getVariable());
		}
		bw.write(";\n");
		
		t = constraint1.get(0);
		bw.write(t.getCoefficient() + " " + t.getVariable());
		for(int i = 1; i < constraint1.size(); i ++) {
			t = constraint1.get(i);
			bw.write(" + " + t.getCoefficient() + " " + t.getVariable());
		}
		bw.write("  >= " + cons_constraint1 + " ;\n");
		
		t = constraint2.get(0);
		bw.write(t.getCoefficient() + " " + t.getVariable());
		for(int i = 1; i < constraint2.size(); i ++) {
			t = constraint2.get(i);
			bw.write(" + " + t.getCoefficient() + " " + t.getVariable());
		}
		bw.write("  >= " + cons_constraint2 + " ;\n");
		
		t = constraint3.get(0);
		bw.write(t.getCoefficient() + " " + t.getVariable());
		for(int i = 1; i < constraint3.size(); i ++) {
			t = constraint3.get(i);
			bw.write(" + " + t.getCoefficient() + " " + t.getVariable());
		}
		bw.write("  >= " + cons_constraint3 + " ;\n");
		
		System.out.println("Added the obj and constraints in the file");
		
		//if(type.equals("sec")){
			for(int i = 0; i < objective.size(); i ++){
				bw.write(objective.get(i).getVariable() + " <= 1;\n");
				//bw.write(objective.getVariables().get(i) + " >= 0;\n");
				System.out.println(i);
			}
		//}
		
		bw.write(type + " ");
		bw.write(" " + objective.get(0).getVariable());
		for(int i = 1; i < objective.size(); i ++){
			bw.write(", " + objective.get(i).getVariable());
		}
		bw.write(";\n");
		
		bw.close();
	}
	
	public static Triple<Double, Double, Double> compute_k1k2k3(double[] A, double[] B, double[] C){
	
		double k1 = ( (B[0] - A[0])*(C[1] - A[1]) ) - ( (B[1] - A[1])*(C[0] - A[0]) ); // (Bx - Ax)(Cy - Ay) - (By - Ay)(Cx - Ax)
		double k2 = ( B[0] - A[0] ); // Bx - Ax
		double k3 = ( B[1] - A[1] ); // By - Ay
		
		return new Triple<Double, Double, Double>(k1, k2, k3);
	}
	
	public static Pair<Double, Double> computeConstraintCoeff(double[] A, double[] B, double[] C, int y_il ){
		
		
		double k1 = ( (B[0] - A[0])*(C[1] - A[1]) ) - ( (B[1] - A[1])*(C[0] - A[0]) ); // (Bx - Ax)(Cy - Ay) - (By - Ay)(Cx - Ax)
		double k2 = ( B[0] - A[0] ); // Bx - Ax
		double k3 = ( B[1] - A[1] ); // By - Ay
		
		double coeff = -k1*k2*y_il - k1*k3*(1 - y_il); // -k1*k2*y_i,l - k1*k3*(1-y_i,l) 
				
		double constant = k1 * k2 * y_il; //Constant in the constraint
		
		
		//varCoeffs[0][0] += (-k1 * k2 * A[1] + k1 * k3 * A[0]); //Constant in the constraint
		
		return new Pair<Double, Double>(coeff, constant);
		
	}
	
	public static int[] initVec(int ygold[], int sz){
		int yi[] = new int[sz+1]; // 0 position is nil label and is not filled; so one extra element is  created ( 1 .. 51)
		Arrays.fill(yi, 0);
		
		for(int y : ygold)
			yi[y] = 1;

		return yi;
	}
	
	public static void main(String args[]) throws NumberFormatException, IOException, IloException{
		ArrayList<Region> regions = readRegionFile("../regions_coeff_binary.txt");
		
		String datasetFile ="../dataset/reidel_trainSVM.data";
		ArrayList<DataItem> dataset = Utils.populateDataset(datasetFile);
	
		System.out.println("Log: Loaded the dataset!");
		
		double Lambda[][] = new double[dataset.size()][52];

		// init Lambda
		for(int i = 0; i < dataset.size(); i ++){
			for(int j = 1; j < 51; j ++){
				Lambda[i][j] = Math.random();
			}
		}
		
		ArrayList<YZPredicted> YtildeStar = optLossLag(dataset, 51, regions, Lambda);
		
		int numPosLabels = 0;
		for(int i = 0; i < YtildeStar.size(); i++){
			YZPredicted ytildestar_i = YtildeStar.get(i);
			numPosLabels += ytildestar_i.yPredicted.size();
		}
		
		System.out.println("Total number of positive labels : " + numPosLabels);	
		
	}
}
