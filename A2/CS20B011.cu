// SMIT BAGUL
// CS20B011
// Transpose kernel uses shared memory
// Multiplication kernel uses memory coalescing

#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

/*Matrix transpose using shared memory*/
__global__ void Transpose(int *D, int *T,int r, int q )
{
	__shared__ int arr[32*33];    // launch with 32 and 32
	unsigned r1 = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned c1 = threadIdx.x + blockIdx.x*blockDim.x;

	unsigned r2 = blockIdx.x*blockDim.x + threadIdx.y;
	unsigned c2 = blockIdx.y*blockDim.y + threadIdx.x;

	unsigned tmp = threadIdx.x + threadIdx.y*33;
	unsigned tmp2 = threadIdx.y + threadIdx.x*33;

	if(r1<r && c1<q)
	{
		arr[tmp] = D[r1*q+c1]; 
	}

	__syncthreads();

	if(r2<q && c2<r)
	{
		T[r2*r+c2] = arr[tmp2];
	}
}

// __global__ void Transpose(int *D, int *T, int r, int q){                             //Naive transpose
// 	unsigned val = (blockDim.y*blockDim.x*(gridDim.y*blockIdx.x + blockIdx.y) + threadIdx.x * blockDim.y + threadIdx.y);
// 	unsigned row = val/q;
// 	unsigned col = val%q; 
// 	if(row < r && col < q)
// 	{
// 		T[col*r + row] = D[row*q + col];
// 	}
// }

/*This function calculates AB in a column access manner,i.e. in matrix multiplication A is accessed row wise and B 
is accessed column wise, but here we use Atranspose so we can access A too columnwise for memory coalescing 
so AT*B gives AB if we access elemennts of A in columnwise manner*/
__global__ void Multiplication(int *A, int *B, int *T,int *C, int *D, int *T2,int *E, int p, int q, int r)
{
	unsigned val = blockDim.y*blockDim.x*(gridDim.y*blockIdx.x + blockIdx.y) + threadIdx.x * blockDim.y + threadIdx.y;
	unsigned col1 = val/r;
	unsigned col2 = val%r;

	if(col1<p && col2<r)
	{
		T[col1*r+col2] = 0;
		T2[col1*r+col2] = 0;

		for(int i=0;i<q;i++)
		{
			T[col1*r+col2] += A[i*p+col1]*B[i*r+col2];            //both accesses are columnwise for memory coalescing
			T2[col1*r+col2] += C[i*p+col1]*D[i*r+col2];
		}

		// T2[col1*r+col2] = 0;                              //gives more time
		// for(int i=0;i<q;i++)
		// {
		// 	T2[col1*r+col2] += C[i*p+col1]*D[i*r+col2];
		// }
	}

	//Addition in same kernel gives worse performance
	// if(col1<p && col2<r)
	//  {
	//		E[col1*r+col2] = T[col1*r+col2] + T2[col1*r+col2]
	//	}
}

__global__ void Addition(int *A, int *B, int *E, int p, int r)
{
	unsigned val = blockDim.y*blockDim.x*(gridDim.y*blockIdx.x + blockIdx.y) + threadIdx.x * blockDim.y + threadIdx.y;
	unsigned row = val/r;
	unsigned col = val%r;

	if(row<p && col<r)
	{
		E[row*r+col] = A[row*r+col]+B[row*r+col];
	}
}


// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE, int *h_matTmp){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	int *d_matrixDT, *d_matrixAB, *d_matrixCDT, *d_matrixAT, *d_matrixCT;

	cudaMalloc(&d_matrixDT, q * r * sizeof(int));
	cudaMalloc(&d_matrixAB, p * r * sizeof(int));
	cudaMalloc(&d_matrixCDT, p * r * sizeof(int));
	cudaMalloc(&d_matrixAT, p * q * sizeof(int));
	cudaMalloc(&d_matrixCT, p * q * sizeof(int));

	//cudaMemset(d_matrixAB, 0, p * r * sizeof(int));
	//cudaMemset(d_matrixCDT, 0, p * r * sizeof(int));

	int gridDimx, gridDimy, gridDim1;

	gridDim1 = (p/32) + (p%32!=0);
	gridDimx = (q/32) + (q%32!=0);
	gridDimy = (r/32) + (r%32!=0);
	dim3 grid1(gridDimx,gridDimy,1);
	dim3 grid0(gridDimx,gridDim1,1);
	dim3 block1(32,32,1);
	Transpose<<<grid1,block1>>> (d_matrixD,d_matrixDT,r,q);     //transpose D to calculate CDT
	Transpose<<<grid0,block1>>> (d_matrixA,d_matrixAT,p,q);     //transpose A for columnwise access while computing AB
	Transpose<<<grid0,block1>>> (d_matrixC,d_matrixCT,p,q);     //transpose C for columnwise access while computing CDT
	cudaDeviceSynchronize();

	gridDimx = (p/32) + (p%32!=0);
	gridDimy = (r/32) + (r%32!=0);
	dim3 grid2(gridDimx,gridDimy,1);
	dim3 block2(32,32,1);
	Multiplication<<<grid2,block2>>>(d_matrixAT,d_matrixB,d_matrixAB,d_matrixCT,d_matrixDT,d_matrixCDT,d_matrixE,p,q,r);
	cudaDeviceSynchronize();

	Addition<<<grid2,block2>>>(d_matrixAB,d_matrixCDT,d_matrixE,p,r);
	cudaDeviceSynchronize();

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_matTmp, d_matrixDT, q * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE, *matTmp;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));
	matTmp = (int*) malloc(q * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE, matTmp);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);
	//writeMatrix(outputFilePtr, matTmp, q, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}