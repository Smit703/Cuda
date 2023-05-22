/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-1
 * Description: Computation of a matrix C = Kronecker_prod(A, B.T)
 *              where A and B are matrices of dimension (m, n) and
 *              the output is of the dimension (m * n, m * n). 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
using namespace std;

ofstream outfile; // The handle for printing the output

//C[m*n*n*r1 + m*n*c2 + m*c1 + r2] = A[n*r1+c1] * B[n*r2+c2];

__global__ void per_row_AB_kernel(long int *A, long int *B, long int *C,long int m, long int n)
{
    unsigned long row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long ai,bi,ci,r1,r2;

    if(row < m*m)
    {    
        r1 = row/m;
        r2 = row%m;
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<n;j++)
            {
                ci = (m*n*n*r1) + (m*n*j) + (m*i) + r2;
                ai = n*r1 + i;
                bi = n*r2 + j;
                C[ci] = A[ai]*B[bi];
            }
        }

    }
}

__global__ void per_column_AB_kernel(long int *A, long int *B, long int *C,long int m, long int n)
{
    unsigned long column = blockDim.y*blockDim.x*blockIdx.x + threadIdx.x * blockDim.y + threadIdx.y;
    unsigned long ai,bi,ci,c1,c2;

    if(column < n*n)
    {
        c1 = column/n;
        c2 = column%n;
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<m;j++)
            {
                ci = (m*n*n*i) + (m*n*c2) + (m*c1) + j;
                ai = n*i + c1;
                bi = n*j + c2;
                C[ci] = A[ai]*B[bi];
            }
        }
    }
    
}

__global__ void per_element_kernel(long int *A, long int *B, long int *C,long int m, long int n)
{
    unsigned long id = (blockDim.y*blockDim.x*(gridDim.y*blockIdx.x + blockIdx.y) + threadIdx.x * blockDim.y + threadIdx.y);
    unsigned long row = id/(m*n);
	unsigned long column = id%(m*n);
    unsigned long ai,bi,ci,r1,c1,r2,c2;


    if(id<m*m*n*n)
    {   
        r1 = row/n; 
        c1 = row%n;
        r2 = column/n;
        c2 = column%n;

        ai = r1*n+c1;
        bi = r2*n+c2;
        ci = (r1*n+c2)*m*n + c1*m+r2;
        C[ci] = A[ai] * B[bi];
    }
}

/**
 * Prints any 1D array in the form of a matrix
 **/
void printMatrix(long int *arr, long int rows, long int cols, char* filename){
    outfile.open(filename);
    for(long int i = 0; i < rows; i++){
        for(long int j = 0; j < cols; j++){
            outfile<<arr[i * cols + j]<<" ";
        }
        outfile<<"\n";
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    long int m,n;	
    cin>>m>>n;	

    // Host_arrays 
    long int *h_a,*h_b,*h_c;

    // Device arrays 
    long int *d_a,*d_b,*d_c;
	
    // Allocating space for the host_arrays 
    h_a = (long int *) malloc(m * n * sizeof(long int));
    h_b = (long int *) malloc(m * n * sizeof(long int));	
    h_c = (long int *) malloc(m * m * n * n * sizeof(long int));	

    // Allocating memory for the device arrays 
    cudaMalloc(&d_a, m * n * sizeof(long int));
	cudaMalloc(&d_b, m * n * sizeof(long int));
	cudaMalloc(&d_c, m * m * n * n * sizeof(long int));


    // Read the input matrix A 
    for(long int i = 0; i < m * n; i++) {
        cin>>h_a[i];
    }

    //Read the input matrix B 
    for(long int i = 0; i < m * n; i++) {
        cin>>h_b[i];
    }

    // Transfer the input host arrays to the device 
    cudaMemcpy(d_a, h_a, m * n * sizeof(long int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, m * n * sizeof(long int), cudaMemcpyHostToDevice);

    long int gridDimx, gridDimy;
    
    // Launch the kernels
    /**
     * Kernel 1 - per_row_AB_kernel
     * To be launched with 1D grid, 1D block
     * Each thread should process a complete row of A, B
    **/

    double starttime = rtclock();
    gridDimx = (m*m)/1024 + ((m*m)%1024!=0);
    dim3 grid1(gridDimx,1,1);
	dim3 block1(1024,1,1);
    per_row_AB_kernel<<<grid1,block1>>>(d_a,d_b,d_c,m,n); 
    cudaDeviceSynchronize();                                                     

    double endtime = rtclock(); 
	printtime("GPU Kernel-1 time: ", starttime, endtime);  
    cudaMemcpy(h_c, d_c, m * n * m * n * sizeof(long int), cudaMemcpyDeviceToHost);
    printMatrix(h_c, m * n, m * n,"kernel1.txt");
    cudaMemset(d_c, 0, m * n * m * n * sizeof(long int));

    /**
     * Kernel 2 - per_column_AB_kernel
     * To be launched with 1D grid, 2D block
     * Each thread should process a complete column of  A, B
     **/
    
    // --> Set the launch configuration 

    starttime = rtclock(); 
    gridDimx = (n*n)/1024 + ((n*n)%1024!=0);
    dim3 grid2(gridDimx,1,1);
	dim3 block2(32,32,1);
    per_column_AB_kernel<<<grid2,block2>>>(d_a,d_b,d_c,m,n);
    cudaDeviceSynchronize(); 

    endtime = rtclock(); 
  	printtime("GPU Kernel-2 time: ", starttime, endtime);  
    cudaMemcpy(h_c, d_c, m * n * m * n * sizeof(long int), cudaMemcpyDeviceToHost);
    printMatrix(h_c, m * n, m * n,"kernel2.txt");
    cudaMemset(d_c, 0, m * n * m * n * sizeof(long int));

    /**
     * Kernel 3 - per_element_kernel
     * To be launched with 2D grid, 2D block
     * Each thread should process one element of the output 
     **/
    long int tmpn = n*n;
    long int tmpm = m*m;
    gridDimx = tmpn/16 + (tmpn%16!=0);
    gridDimy = tmpm/64 + (tmpm%64!=0);
    dim3 grid3(gridDimx,gridDimy,1);
    dim3 block3(64,16,1);

    starttime = rtclock();  
    per_element_kernel<<<grid3,block3>>>(d_a,d_b,d_c,m,n);
    cudaDeviceSynchronize();                                                         
    endtime = rtclock();  
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }     
	printtime("GPU Kernel-3 time: ", starttime, endtime);  
    cudaMemcpy(h_c, d_c, m * n * m * n * sizeof(long int), cudaMemcpyDeviceToHost);
    printMatrix(h_c, m * n, m * n,"kernel3.txt");

    return 0;
}