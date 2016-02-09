/*******************************************************************************************************/
/*                                                                                                     */
/* main.cpp                                                                                            */
/* iTEBD                                                                                               */
/* version 0.1																					                                          	   */
/*                                                                                                     */
/* This code is iTEBD algorithm for Infinite-Size Q Lattice Systems (Ising and Heisenberg) in 1D	     */
/* Copyright (C) 2016  Jozef Genzor <jozef.genzor@gmail.com>                                           */
/*                                                                                                     */
/*                                                                                                     */
/* This file is part of iTEBD.                                                                         */
/*                                                                                                     */
/* iTEBD is free software: you can redistribute it and/or modify                                       */
/* it under the terms of the GNU General Public License as published by                                */
/* the Free Software Foundation, either version 3 of the License, or                                   */
/* (at your option) any later version.                                                                 */
/*                                                                                                     */
/* iTEBD is distributed in the hope that it will be useful,                                            */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of                                      */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                       */
/* GNU General Public License for more details.                                                        */
/*                                                                                                     */
/* You should have received a copy of the GNU General Public License                                   */
/* along with iTEBD.  If not, see <http://www.gnu.org/licenses/>.                                      */
/*                                                                                                     */
/*******************************************************************************************************/

/*
 * This code is an implementation of the iTEBD algorithm based on the article: 
 *
 * Classical simulation of infinite-size quantum lattice systems in one spatial dimension
 * Phys. Rev. Lett 98, 070201 (2007), http://arxiv.org/abs/cond-mat/0605597
 *
 *
 * For linear algebra (Singular Value Decomposition (SVD) in particular), 
 * Eigen library (3.2.7) is called. 
 *
 * Compilation under Mac:
 *
 * g++ -m64 -O3 -I/.../Eigen main.cpp -o main.x
 *
 */

/*******************************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "SVD"

#define NDEBUG
#include <assert.h>

using namespace Eigen;

#define LIMIT_M  1.E-15
#define LIMIT_E  1.E-15

/*******************************************************************************************************/

void mps_update(const int & d, const int & D, const double & epsilon, const int & xi, int * const xi_p, 
				double * const G_A, double * const G_B, 
				double * const lambda_A, const double * const lambda_B, 
				double **U);

MatrixXd psi_expansion(const int & d, const int & xi, const int & xi_p, 
					   const double * const G_A, const double * const G_B, 
					   const double * const lambda_A, const double * const lambda_B);

void magnetization_z(const int & d, const int & xi, const int & xi_p, const double * const G_A, 
					 const double * const lambda_A, const double * const lambda_B, double * const mag);
void magnetization_x(const int & d, const int & xi, const int & xi_p, const double * const G_A, 
					 const double * const lambda_A, const double * const lambda_B, double * const mag_x);

void normalization(const int & dim, double * const lambda);

void energy_calculation(const int & IH, const int & d, const double & h, const int & xi, const int & xi_p, 
					   const double * const G_A, const double * const G_B, 
					   const double * const lambda_A, const double * const lambda_B, double * const energy);

double entropy_calculation(const int & dim, const double * const lambda_B);

/*******************************************************************************************************/

int main () 
{
	clock_t start, end;
	double  runTime;
	start = clock(); //time measurement start
	
	int IH_r; 
	double initial_r; 
	double final_r; 
	double step_r; 
	double tau_r;
	int D_r;
	int N_r;
	double epsilon_r;
	
	FILE* par;
	par = fopen("INIT.txt", "r");
	
	if ( par == NULL ) 
	{ 
		printf("Can't open the input file \"INIT.txt\"\n");
		return 1;
	}
	
	fscanf(par, "%d%*[^\n]", &IH_r); // for Ising (0), for Heisenberg (1)
	fscanf(par, "%lf%*[^\n]", &initial_r); 
	fscanf(par, "%lf%*[^\n]", &final_r);  
	fscanf(par, "%lf%*[^\n]", &step_r);  
	fscanf(par, "%lf%*[^\n]", &tau_r);  
	fscanf(par, "%d%*[^\n]", &D_r);		  
	fscanf(par, "%d%*[^\n]", &N_r);		
	fscanf(par, "%lf%*[^\n]", &epsilon_r);  
	
	fclose(par);
	
	const int IH = IH_r; //model: Ising (0), Heisenberg (1)
	const double initial = initial_r; //external magnetic field
	const double final = final_r;
	const double step = step_r;
	const double tau = tau_r; //imaginary time step
	const int D = D_r; //max Schmidt rank xi
	const int N = N_r; //max number of iterations
	const double epsilon = epsilon_r; 
	
	if ( IH != 0 && IH != 1 ) 
	{
		printf("choose correct value for Ising (0) or Potts (1) in INIT.txt\n");
		abort();
	}
	
	if ( IH==0 ) 
	{
		printf("# 1D quantum Ising model\n");
	}
	else 
	{
		printf("# 1D quantum Heisenberg model\n");
	}
	printf("# tau = %1.1E\t\tD = %d\t\tN = %d\t\tepsilon = %1.1E\n", tau, D, N, epsilon);

	
	FILE* fw;
	if ((fw = fopen("DATA.txt", "w")) == NULL)  
	{
		printf("Can't open the output file \"DATA.txt\"\n");
		return 1;
	}
	fclose(fw);
	
	fw = fopen("DATA.txt","a");	
	if ( IH==0 ) 
	{
		fprintf(fw, "# 1D quantum Ising model\n");
	}
	else 
	{
		fprintf(fw, "# 1D quantum Heisenberg model\n");
	}
	fprintf(fw, "# tau = %1.1E\t\tD = %d\t\tN = %d\t\tepsilon = %1.1E\n", tau, D, N, epsilon);
	fprintf(fw, "# h\t\tground-state energy\tsigma_x\t\t\tsigma_z\t\t\tentropy\t\t\t\titer\n");
	fclose(fw);
	
	const int d =   2;   //physical dimension
	
	double  *G_A;        //even MPS \Lambda^A here represented as 1 dim array
	G_A = (double  *)malloc(d*D*D*sizeof(double ));
	double  *G_B;        //odd MPS \Lambda^B here represented as 1 dim array
	G_B = (double  *)malloc(d*D*D*sizeof(double ));
	double  *lambda_A;   //even MPS \lambda^A 
	lambda_A = (double  *)malloc(D*sizeof(double ));
	double  *lambda_B;   //odd MPS \lambda^A 
	lambda_B = (double  *)malloc(D*sizeof(double )); 
	
	double  *U[d*d];                                   
	for (int j=0; j<d*d; j++) 
	{
		U[j]=(double  *)malloc((d*d)*sizeof(double ));
	}
	
	for (int i=0; i<d*d; i++) 
	{
		for (int j=0; j<d*d; j++) 
		{
			U[i][j] = 0.0;
		}
	}
		
	for (double h = initial; h <= final; h += step) 
	{		
		if (IH == 0) 
		{
			/*     non-unitary evolution (in imaginary time)     *
			 *     for quantum Ising model                      */
			
			double s;
			s = sqrt(1 + h*h);
			
			U[0][0] =   (exp(s*tau)*(-h+s) - exp(-s*tau)*(-h-s))/(2*s);
			U[0][3] = - sinh(s*tau)/s; //(exp(-s*tau) - exp(s*tau))/(2*s);
			U[1][1] =   cosh(tau);
			U[1][2] = - sinh(tau);
			U[2][1] = - sinh(tau);
			U[2][2] =   cosh(tau); 
			U[3][0] = - sinh(s*tau)/s; //(exp(-s*tau) - exp(s*tau))/(2*s);
			U[3][3] =   (exp(s*tau)*(h+s) - exp(-s*tau)*(h-s))/(2*s);
		}
		else 
		{
			/*     non-unitary evolution (in imaginary time)		  *
			 *     for quantum Heisenberg model                      */
			
			U[0][0] =   exp(-tau*(1+h));
			U[1][1] =   exp(tau)*cosh(2*tau);
			U[1][2] = - exp(tau)*sinh(2*tau);
			U[2][1] = - exp(tau)*sinh(2*tau);
			U[2][2] =   exp(tau)*cosh(2*tau); 
			U[3][3] =   exp(-tau*(1-h));
		}

		int xi   = 1; // Schmidt rank, goes up to D - for odd and even part of the chain
		int xi_p = 1;
		
		/* Initialization of G_A and G_B */
		
		for (int i=0; i<d; i++) 
		{
			for (int a=0; a<xi; a++) 
			{
				for (int b=0; b<xi; b++) 
				{
					G_A[xi*xi*i + xi*a + b] = 0.0;
					G_B[xi*xi*i + xi*a + b] = 0.0;
				}
			}
		}
		
		if (IH == 0) 
		{
			//Ising model
			 G_A[0] = 1.0;
			 G_B[0] = 1.0;
		}
		else 
		{
			// Heisenberg model
			G_A[1] = 1.0;
			G_A[0] = 1.0;
			G_B[0] = 1.0;
		}
		
		/* Initialization of lambda_A and lambda_B */
		
		lambda_A[0] = 1.0;
		lambda_B[0] = 1.0; 
		
		double  energy   = 1;
		double  energy_p = 0;
		
		double  mag_z =   1;
		double  mag_z_p = 0;
		double  mag_x = 1;
		double  mag_x_p = 0;
		
		double entropy = 0;
		
		int iter= 0;
		
		const int Min =  100;
		
		printf("# Iter\th\t\tXi\tXi'\tground-state energy\tsigma_x\t\t\tsigma_z\t\t\tentropy\n");
		
		while ( ( ( iter < N ) && ( (fabs(mag_z - mag_z_p) > LIMIT_M) || (fabs(mag_x - mag_x_p) > LIMIT_M) ) ) || (iter < Min))
		{
			/*
			 // The imaginary-time step tau may be adjusted here as needed. 
			 if (i > Min) 
			 {
			 // tau = ***your formula here***
			 //printf("tau = %.16f\n", tau);
			 if (IH == 0) 
			 {
			 //    non-unitary evolution (in imaginary time)   
			 //    for quantum Ising model                  
			 double s;
			 s = sqrt(1 + h*h);
			 U[0][0] =   (exp(s*tau)*(-h+s) - exp(-s*tau)*(-h-s))/(2*s);
			 U[0][3] = - sinh(s*tau)/s; //(exp(-s*tau) - exp(s*tau))/(2*s);
			 U[1][1] =   cosh(tau);
			 U[1][2] = - sinh(tau);
			 U[2][1] = - sinh(tau);
			 U[2][2] =   cosh(tau); 
			 U[3][0] = - sinh(s*tau)/s; //(exp(-s*tau) - exp(s*tau))/(2*s);
			 U[3][3] =   (exp(s*tau)*(h+s) - exp(-s*tau)*(h-s))/(2*s);
			 }
			 else 
			 {
			 //     non-unitary evolution (in imaginary time)    
			 //     for quantum Heisenberg model                     
			 U[0][0] =   exp(-h-tau);
			 U[1][1] =   exp(tau)*cosh(2*tau);
			 U[1][2] = - exp(tau)*sinh(2*tau);
			 U[2][1] = - exp(tau)*sinh(2*tau);
			 U[2][2] =   exp(tau)*cosh(2*tau); 
			 U[3][3] =   exp(h-tau);
			 }
			 }
			 */
						
			mps_update(d, D, epsilon, xi, &xi_p, G_A, G_B, lambda_A, lambda_B, U);
			normalization(xi_p, lambda_A);
			
			
			mps_update(d, D, epsilon, xi_p, &xi, G_B, G_A, lambda_B, lambda_A, U);
			normalization(xi, lambda_B);
						
			energy_p = energy;
			energy_calculation(IH, d, h, xi, xi_p, G_A, G_B, lambda_A, lambda_B, &energy); 
			
			mag_z_p = mag_z;
			magnetization_z(d, xi, xi_p, G_A, lambda_A, lambda_B, &mag_z);
			mag_x_p = mag_x;
			magnetization_x(d, xi, xi_p, G_A, lambda_A, lambda_B, &mag_x);
			
			entropy = entropy_calculation(xi, lambda_B);

			printf("%d\t%1.6E\t%d\t%d\t%.16f\t%.16f\t%.16f\t%.16f\n", iter+1, h, xi, xi_p, energy, mag_x, mag_z, entropy);
						
			iter += 1;
		}
		fw = fopen("DATA.txt","a");
		fprintf(fw,"%1.6E\t%1.16E\t%1.16E\t%1.16E\t%1.16E\t\t%d", h, energy, mag_x, mag_z, entropy, iter);
		fprintf(fw,"\n");
		fclose(fw);
	}
	
	free((void *) G_A);
	G_A = NULL;
	free((void *) G_B);
	G_B = NULL;
	free((void *) lambda_A);
	lambda_A = NULL;
	free((void *) lambda_B);
	lambda_B = NULL;
	
	for (int i=0; i<d*d; i++) 
	{
		free(U[i]);       
		U[i]=NULL;	      
	}
	
	end = clock();        //time measurement end
	runTime = (end - start) / (double ) CLOCKS_PER_SEC ;
	printf ("Run time is %f seconds\n", runTime);
	
	fw = fopen("DATA.txt","a");
	fprintf(fw, "# Time = %.6f seconds\n", runTime);
	fclose(fw);
	
    return 0;
}

/*******************************************************************************************************/
/*******************************************************************************************************/

void mps_update(const int & d, const int & D, const double & epsilon, const int & xi, int * const xi_p, 
				double * const G_A, double * const G_B, 
				double * const lambda_A, const double * const lambda_B, 
				double **U)
{
		
	MatrixXd psi = psi_expansion(d, xi, *xi_p, G_A, G_B, lambda_A, lambda_B); 
	
	MatrixXd Theta(xi*d, xi*d);
	
	double sum;
	
	for (int a=0; a<xi; a++) 
	{
		for (int i=0; i<d; i++) 
		{
			for (int j=0; j<d; j++) 
			{
				for (int c=0; c<xi; c++) 
				{
					sum = 0;
					for (int k=0; k<d; k++) 
					{
						for (int l=0; l<d; l++) 
						{
							sum += U[d*i+j][d*k+l] * psi(d*a + k, xi*l + c);
						}
					}
					Theta(d*a + i, xi*j + c) = sum;
				}
			}
		}
	}
	
	JacobiSVD<MatrixXd> svd(Theta, ComputeThinU | ComputeThinV);
	
	if ( d * xi > D ) 
	{
		(*xi_p) = D;
	}
	else 
	{
		(*xi_p) = d * xi;
	}
	
	for (int b=0; b<(*xi_p); b++) 
	{
		//printf("singularValues[%d] = %.16f\n", b, svd.singularValues()(b));
		if ( svd.singularValues()(b) < epsilon ) 
		{
			(*xi_p) = b;
			break;
		}
		lambda_A[b] = svd.singularValues()(b);
	}
	
	for (int i=0; i<d; i++) 
	{
		for (int a=0; a<xi; a++) 
		{
			for (int b=0; b<(*xi_p); b++) 
			{
				G_A[(*xi_p)*xi*i + (*xi_p)*a + b] = svd.matrixU()(d*a + i, b);
			}
		}
	}
		
	for (int j=0; j<d; j++) 
	{
		for (int b=0; b<(*xi_p); b++) 
		{
			for (int c=0; c<xi; c++) 
			{
				G_B[xi*(*xi_p)*j + xi*b + c] = svd.matrixV()(xi*j + c, b); //V is transposed here
			}
		}
	}
	
	for (int i=0; i<d; i++) 
	{
		for (int a=0; a<xi; a++) 
		{
			for (int b=0; b<(*xi_p); b++) 
			{
				G_A[(*xi_p)*xi*i + (*xi_p)*a + b] /= lambda_B[a];
			}
		}
	}
	
	for (int j=0; j<d; j++) 
	{
		for (int b=0; b<(*xi_p); b++) 
		{
			for (int c=0; c<xi; c++) 
			{
				G_B[xi*(*xi_p)*j + xi*b + c] /= lambda_B[c];
			}
		}
	}
}

MatrixXd psi_expansion(const int & d, const int & xi, const int & xi_p, 
					   const double * const G_A, const double * const G_B, 
					   const double * const lambda_A, const double * const lambda_B)
{
	MatrixXd psi(d*xi, d*xi);
	
	double sum; 
	
	for (int a=0; a<xi; a++) 
	{
		for (int i=0; i<d; i++) 
		{
			for (int j=0; j<d; j++) 
			{
				for (int c=0; c<xi; c++) 
				{
					sum = 0;
					for (int b=0; b<xi_p; b++) 
					{
						sum += lambda_B[a] * G_A[xi_p*xi*i + xi_p*a + b] * lambda_A[b] * G_B[xi_p*xi*j + xi*b + c] * lambda_B[c];
					}
					psi(d*a + i, xi*j + c) = sum;
				}
			}
		}
	}
	
	return psi; 
}

void magnetization_z(const int &d, const int & xi, const int & xi_p, const double * const G_A, const double * const lambda_A, const double * const lambda_B, double * const mag)
{
	double  norm = 0;
	
	for (int a=0; a<xi; a++) 
	{
		for (int b=0; b<xi_p; b++) 
		{
			for (int i=0; i<d; i++) 
			{
				norm += lambda_B[a]*lambda_B[a]*lambda_A[b]*lambda_A[b]*G_A[xi_p*xi*i + xi_p*a + b]*G_A[xi_p*xi*i + xi_p*a + b];
			}
		}
	}
	
	(*mag) = 0;
	
	for (int a=0; a<xi; a++) 
	{
		for (int b=0; b<xi_p; b++) 
		{
			for (int i=0; i<d; i++) 
			{
				(*mag) += (1-2*i)*lambda_B[a]*lambda_B[a]*lambda_A[b]*lambda_A[b]*G_A[xi_p*xi*i + xi_p*a + b]*G_A[xi_p*xi*i + xi_p*a + b];
			}
		}
	}
	
	(*mag) /= norm;
}

void magnetization_x(const int & d, const int & xi, const int & xi_p, const double * const G_A, const double * const lambda_A, const double * const lambda_B, double * const mag_x)
{
	double  norm = 0;
	
	for (int a=0; a<xi; a++) 
	{
		for (int b=0; b<xi_p; b++) 
		{
			for (int i=0; i<d; i++) 
			{
				norm += lambda_B[a]*lambda_B[a]*lambda_A[b]*lambda_A[b]*G_A[xi_p*xi*i + xi_p*a + b]*G_A[xi_p*xi*i + xi_p*a + b];
			}
		}
	}
	
	(*mag_x) = 0;
	
	for (int a=0; a<xi; a++) 
	{
		for (int b=0; b<xi_p; b++) 
		{
			(*mag_x) += lambda_B[a]*lambda_B[a]*lambda_A[b]*lambda_A[b]*G_A[xi_p*xi*0 + xi_p*a + b]*G_A[xi_p*xi*1 + xi_p*a + b] + 
			lambda_B[a]*lambda_B[a]*lambda_A[b]*lambda_A[b]*G_A[xi_p*xi*1 + xi_p*a + b]*G_A[xi_p*xi*0 + xi_p*a + b];
		}
	}
	
	(*mag_x) /= norm;
}

void normalization(const int & dim, double * const lambda)
{
	double  sum = 0;

	for (int b=0; b<dim; b++) 
	{
		sum += lambda[b]*lambda[b];
	}
	
	sum = sqrt(sum);
	
	for (int b=0; b<dim; b++) 
	{
		lambda[b] /= sum;
	}
}

/*******************************************************************************************************/

void energy_calculation(const int & IH, const int & d, const double & h, const int & xi, const int & xi_p, 
					   const double * const G_A, const double * const G_B, 
					   const double * const lambda_A, const double * const lambda_B, double * const energy)
{
	MatrixXd psi = psi_expansion(d, xi, xi_p, G_A, G_B, lambda_A, lambda_B); 
	
	double norm=0;
	
	for (int a=0; a<xi; a++) 
	{
		for (int c=0; c<xi; c++) 
		{
			for (int i=0; i<d; i++) 
			{
				for (int j=0; j<d; j++) 
				{
					norm += psi(d*a + i, xi*j + c) * psi(d*a + i, xi*j + c);
				}
			}
		}
	}
	
	double  *H[d*d];                                   
	for (int j=0; j<d*d; j++) 
	{
		H[j]=(double  *)malloc((d*d)*sizeof(double ));
	}
	
	for (int i=0; i<d*d; i++) 
	{
		for (int j=0; j<d*d; j++) 
		{
			H[i][j] = 0;
		}
	}
	
	if ( IH == 0) 
	{
		//Ising model (Hamiltonian)
		H[0][0] =  h;
		H[0][3] =  1;
		H[1][2] =  1;
		H[2][1] =  1;
		H[3][0] =  1;
		H[3][3] = -h; 
	}
	else 
	{
		//Heisenberg model (Hamiltonian)
		H[0][0] =   1 + h;
		H[1][1] = - 1;
		H[1][2] =   2;
		H[2][1] =   2;
		H[2][2] = - 1;
		H[3][3] =   1 - h;
	}

	double  sum = 0;
	
	for (int a=0; a<xi; a++) 
	{
		for (int c=0; c<xi; c++) 
		{
			for (int k=0; k<d; k++) 
			{
				for (int l=0; l<d; l++) 
				{
					for (int i=0; i<d; i++) 
					{
						for (int j=0; j<d; j++) 
						{
							sum += psi(d*a + k, xi*l + c) * H[d*k + l][d*i + j] * psi(d*a + i, xi*j + c) ; 
						}
					}
				}
			}
		}
	}
	
	for (int i=0; i<d*d; i++) 
	{
		free(H[i]);       
		H[i]=NULL;	      
	}
	
	if ( IH == 0) 
	{
		*energy = sum / norm;
	}
	else 
	{
		// The division by 4 is due to our choice of the Hamiltonian for the Heisenberg model; for comparison, see Eq. (3.1) in
		// F. Franchini, Notes on Bethe ansatz techniques, International School for Advanced Studies-Trieste, Lecture Notes (2011)
		// https://people.sissa.it/~ffranchi/BAnotes.pdf
		// The exact value for ground-state energy when h=0 is E_0 = 1/4 - ln(2), see Eq. (3.88)
		*energy = sum / (4*norm); 
	}	
}

double entropy_calculation(const int & dim, const double * const lambda_B)
{
	double entropy = 0;
	
	for (int i=0; i<dim; i++) 
	{
		//entropy += - lambda_B*lambda_B[i]*log2(lambda_B[i]*lambda_B[i]);
		entropy += - lambda_B[i]*lambda_B[i]*log(lambda_B[i]*lambda_B[i]); //natural logarithm 
	}
	
	return entropy;
}

