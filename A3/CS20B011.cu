/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
//#include <thrust/extrema.h>
//#include <thrust/device_vector.h>
#include "graph.hpp"
 
using namespace std;


ofstream outfile; // The handle for printing the output

/*Process level n-1 and update active in degree*/
__global__ void Update_AID(int *min, int* max, int* aid, int* csr, int* offset, bool* is_active, int V)
{
    unsigned node = blockIdx.x * blockDim.x + threadIdx.x;

   
    if(node<max[0]-min[0]+1)
    {
        int i = offset[min[0]+node];
        int j = offset[min[0]+node+1];
        // if(V==20)
        // {
        //     printf("%d, %d\n",i,j);
        // }
    
        for(int k=i;k<j;k++)
        {
            //if(V==20)
            //{
            //    printf("csr %d\n",csr[k]);    
            //}

            atomicMax(max,csr[k]);
            // if(V==20 && node==0)
            // {
            //     printf("min = %d, max = %d\n",min[0],max[0]);
            // }
            if(is_active[node+min[0]]==true)
            {
                atomicAdd(&aid[csr[k]],1);
            }
        }
    }
    
}

/*Process level n update the actives and deactive acc to rule*/
__global__ void Activate_Deactivate(int* min, int* max, int* aid, int* apr,bool* is_active, int* noOfActives, int curr, int V)
{
    unsigned node = blockIdx.x * blockDim.x + threadIdx.x;

    if(node<max[0]-min[0]+1)
    {
        if(aid[min[0]+node]>=apr[min[0]+node])
        {
            if(node==0 || node==max[0]-min[0])
            {
                atomicAdd(&noOfActives[curr],1);
                is_active[min[0]+node] = true;
            }
            else
            {
                if(aid[min[0]+node-1]>=apr[min[0]+node-1] || aid[min[0]+node+1]>=apr[min[0]+node+1])
                {
                    is_active[min[0]+node] = true;
                    atomicAdd(&noOfActives[curr],1);
                }
            }
        }

        // if(aid[min[0]+node]>=apr[min[0]+node])
        // {
        //     is_active[min[0]+node] = true;
        //     atomicAdd(&noOfActives[curr],1);
        // }
        // if(node!=0 && node!=max[0]-min[0])
        // {
        //     if(!is_active[min[0]+node-1] && !is_active[min[0]+node+1] && is_active[min[0]+node])
        //     {
        //         is_active[min[0]+node] = false;
        //         atomicSub(&noOfActives[curr],1);
        //     }
        // }
    }

    
}
    

//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
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
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();

    //Variable for CSR format on host
    int *h_offset; // for csr offset (prefix sum) size V
    int *h_csrList; // for csr size E
    int *h_apr; // active point requirement
    bool *h_is_active;
    h_is_active = (bool *)malloc(V*sizeof(bool));
    memset(h_is_active,false,V*sizeof(bool));

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
        
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    bool *d_is_active;
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));
    cudaMalloc(&d_is_active,V*sizeof(bool));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    //variable for result, storing number of active vertices at each level, on device
    int *d_activeVertex;
	cudaMalloc(&d_activeVertex, L*sizeof(int));
    //cudaMemset(d_activeVertex,0,L*sizeof(int));
    cudaMemset(d_aid,0,V*sizeof(int));

    double starttime = rtclock();

    //can do all this in a kernel?? will check
    int min = 0;             //a level L will be from min to max-1
    int max = -1;
    int i = 0;
    while(h_apr[i]==0)
    {
        max++;
        h_is_active[i] = true;
        h_activeVertex[0]++;
        i++;
    }
    int *d_min,*d_max;
    cudaMalloc(&d_min,sizeof(int));
    cudaMalloc(&d_max,sizeof(int));
    cudaMemcpy(d_max,&max,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_min,&min,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_is_active,h_is_active,V*sizeof(bool),cudaMemcpyHostToDevice);
    cudaMemcpy(d_activeVertex,h_activeVertex,L*sizeof(int),cudaMemcpyHostToDevice);

    //double starttime = rtclock(); 
    int curr = 1;
    while(curr<L)
    {   
        int elemennts = max-min+1;                              //number of nodes in current level

        // if(L==5)
        // {
        //     printf("elements here - %d, min = %d, max = %d\n",elemennts,min,max);
        // }
        
        int Dim = (elemennts/512)+(elemennts%512!=0);
        int block = 512;

        ///kernel launch config <Dim,block>
        Update_AID<<<Dim,block>>>(d_min,d_max,d_aid,d_csrList,d_offset,d_is_active,V);
        cudaDeviceSynchronize();
        min = max+1;

        //thrust::device_vector<int> d_arr(h_csrList+h_offset[min],h_csrList+h_offset[max]);
        //thrust::device_vector<int>:: iterator itr = thrust::max_element(d_arr.begin(),d_arr.end());
        //max = *itr;
        cudaMemcpy(&max,d_max,sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(d_min,&min,sizeof(int),cudaMemcpyHostToDevice);
        //cudaMemcpy(h_aid,d_aid,V*sizeof(int),cudaMemcpyDeviceToHost);
        //cudaMemcpy(h_is_active,d_is_active,V*sizeof(bool),cudaMemcpyDeviceToHost);


        elemennts = max-min+1;
        Dim = (elemennts/512)+(elemennts%512!=0);
        Activate_Deactivate<<<Dim,block>>>(d_min,d_max,d_aid,d_apr,d_is_active,d_activeVertex,curr,V);
        cudaDeviceSynchronize();

        // cudaMemcpy(h_activeVertex,d_activeVertex,L*sizeof(int),cudaMemcpyDeviceToHost);
        // if(h_activeVertex[curr]==0)
        // {
        //     break;  
        // }

        curr++;
    }

    cudaMemcpy(h_activeVertex,d_activeVertex,L*sizeof(int),cudaMemcpyDeviceToHost);

    // if(V==20)
    // {
    //     for(int i=0;i<L;i++)
    //     {
    //         printf("levels - %d\n",h_activeVertex[i]);
    //     }
    // }
    

     
    double endtime = rtclock();  
    printtime("GPU Kernel time: ", starttime, endtime);  

    // --> Copy C from Device to Host
    char outFIle[30] = "./output.txt" ;
    printResult(h_activeVertex, L, outFIle);
    if(argc>2)
    {
        for(int i=0; i<L; i++)
        {
            printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
        }
    }

    return 0;
}
