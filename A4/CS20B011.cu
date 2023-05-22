// NOTE : had to change compilation instruction to : nvcc "$ROLLNO.cu" -arch=sm_70 --std=c++11 -o main : in given evaluation script

#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define max_N 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;


__global__ void Slotting(int N, int *centre, int *facility, int *fac_ids, int *capacity, int R, int *req_id, int *req_cen, int *req_fac, int *start, int *slots, int *succ, int *offset)
{
    int centre_id = blockIdx.x*1000+blockIdx.y;
    int facility_id = threadIdx.x;

    int curr_cap[25];
    for(int i=0;i<25;i++)
    {
        curr_cap[i] = 0;
    }
    int max_cap = capacity[offset[centre_id]+facility_id];

    // 1 thread processing each facility
    if(centre_id<N && facility_id<facility[centre_id])          
    {
        for(int i=0;i<R;i++)
        {
            if(centre_id!=req_cen[i] || facility_id!=req_fac[i])
            {
                continue;
            }

            bool isfree = true;
            int j = 0;
            int k = start[i];
            int s = slots[i];
            while(j<s)
            {
                if(curr_cap[k]==max_cap || k>24)
                {
                    isfree = false;
                    break;
                }
                k++;
                j++;
            } 

            if(isfree)
            {
                atomicAdd(&succ[centre_id],1);
                int kk = start[i];
                for(int ii=0;ii<s;ii++)
                {
                    curr_cap[kk]++;
                    kk++;
                }
            }  
        }
    }
}



int main(int argc,char **argv)
{
	// variable declarations...
    int N,*centre,*facility,*capacity,*fac_ids, *succ_reqs, *tot_reqs;
    
    int *d_centre, *d_facility, *d_capacity, *d_fac_ids, *d_succ_reqs;

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &N ); // N is number of centres
	
    // Allocate memory on cpu
    centre=(int*)malloc(N * sizeof (int));  // Computer  centre numbers
    facility=(int*)malloc(N * sizeof (int));  // Number of facilities in each computer centre
    fac_ids=(int*)malloc(max_P * N  * sizeof (int));  // Facility room numbers of each computer centre
    capacity=(int*)malloc(max_P * N * sizeof (int));  // stores capacities of each facility for every computer centre 

    cudaMalloc(&d_centre,N*sizeof(int));
    cudaMalloc(&d_facility,N*sizeof(int));
    cudaMalloc(&d_fac_ids,max_P * N  * sizeof (int));
    cudaMalloc(&d_capacity,max_P * N  * sizeof (int));

    int success=0;  // total successful requests
    int fail = 0;   // total failed requests
    //int d_success;
    //cudaMalloc(&d_success,sizeof(int));
    //cudaMemcpy(d_success,&success,sizeof(int),cudaMemcpyHostToDevice);
    

    tot_reqs = (int *)malloc(N*sizeof(int));   // total requests for each centre
    memset(tot_reqs,0,N*sizeof(int));
    succ_reqs = (int *)malloc(N*sizeof(int)); // total successful requests for each centre
    memset(succ_reqs,0,N*sizeof(int));
    cudaMalloc(&d_succ_reqs,N*sizeof(int));
    cudaMemcpy(d_succ_reqs,succ_reqs,N*sizeof(int),cudaMemcpyHostToDevice);

    int *offset, *d_offset;
    offset = (int *)malloc(N*sizeof(int));
    cudaMalloc(&d_offset,N*sizeof(int));


    // Input the computer centres data
    int k1 = 0, k2 = 0;
    for(int i=0;i<N;i++)
    {
      fscanf( inputfilepointer, "%d", &centre[i] );
      fscanf( inputfilepointer, "%d", &facility[i] );
      offset[i] = k2;

      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &fac_ids[k1] );
        k1++;
      }
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &capacity[k2]);
        k2++;     
      } 
    }
    cudaMemcpy(d_offset,offset,N*sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(d_centre,centre,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_facility,facility,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fac_ids,fac_ids,max_P*N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_capacity,capacity,max_P*N*sizeof(int),cudaMemcpyHostToDevice);

    // variable declarations
    int *req_id, *req_cen, *req_fac, *req_start, *req_slots;   // Number of slots requested for every request
    int *d_req_id, *d_req_cen, *d_req_fac, *d_req_start, *d_req_slots;
    
    // Allocate memory on CPU 
	  int R;
	  fscanf( inputfilepointer, "%d", &R); // Total requests
    req_id = (int *) malloc ( (R) * sizeof (int) );  // Request ids
    req_cen = (int *) malloc ( (R) * sizeof (int) );  // Requested computer centre
    req_fac = (int *) malloc ( (R) * sizeof (int) );  // Requested facility
    req_start = (int *) malloc ( (R) * sizeof (int) );  // Start slot of every request
    req_slots = (int *) malloc ( (R) * sizeof (int) );   // Number of slots requested for every request

    cudaMalloc ( &d_req_id,(R) * sizeof (int) );
    cudaMalloc ( &d_req_cen,(R) * sizeof (int) );  
    cudaMalloc ( &d_req_fac,(R) * sizeof (int) );  
    cudaMalloc ( &d_req_start,(R) * sizeof (int) ); 
    cudaMalloc ( &d_req_slots,(R) * sizeof (int) );   
    
    // Input the user request data
    for(int j = 0; j < R; j++)
    {
       fscanf( inputfilepointer, "%d", &req_id[j]);
       fscanf( inputfilepointer, "%d", &req_cen[j]);
       fscanf( inputfilepointer, "%d", &req_fac[j]);
       fscanf( inputfilepointer, "%d", &req_start[j]);
       fscanf( inputfilepointer, "%d", &req_slots[j]);
       tot_reqs[req_cen[j]]+=1;  
    }

    cudaMemcpy(d_req_id,req_id,R*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_cen,req_cen,R*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_fac,req_fac,R*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_start,req_start,R*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_slots,req_slots,R*sizeof(int),cudaMemcpyHostToDevice);

    int Dimx = (N/1000)+(N%1000!=0);
    dim3 grid(Dimx,1000,1);
    Slotting<<<grid,30>>>(N, d_centre, d_facility, d_fac_ids, d_capacity, R, d_req_id, d_req_cen, d_req_fac, d_req_start, d_req_slots, d_succ_reqs, d_offset);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }     

    cudaMemcpy(succ_reqs,d_succ_reqs,N*sizeof(int),cudaMemcpyDeviceToHost);

    // Output
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");
    for(int i=0;i<N;i++)
    {
      success += succ_reqs[i];
    }
    fail = R - success;
    fprintf( outputfilepointer, "%d %d\n", success, fail);
    for(int j = 0; j < N; j++)
    {
        fprintf( outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j]-succ_reqs[j]);
    }
    fclose( inputfilepointer );
    fclose( outputfilepointer );

    cudaDeviceSynchronize();
	return 0;
}