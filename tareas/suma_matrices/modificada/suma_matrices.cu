#include <iostream>
#include <fstream>
#include <time.h>

using namespace std;

const int N=10;

__global__ void MatAdd( float *A, float *B, float *C, int N)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;  // Compute row index
  int i = blockIdx.y * blockDim.y + threadIdx.y;  // Compute column index
  int index=i*N+j; // Compute global 1D index

  if (i < N && j < N)
     {
	    printf( "Block (%d,%d), Thread (%d,%d), i=%d, j=%d \n",  blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,i,j);
	    C[index] = A[index] + B[index]; // Compute C element
     }
}

int main(int argc, char* argv[]){ 
  const int NN=N*N;
  /* pointers to host memory */
  /* Allocate arrays A, B and C on host*/
  float * A = (float*) malloc(NN*sizeof(float));
  float * B = (float*) malloc(NN*sizeof(float));
  float * C = (float*) malloc(NN*sizeof(float));

  /* pointers to device memory */
  float *A_d, *B_d, *C_d;
  /* Allocate arrays a_d, b_d and c_d on device*/
  cudaMalloc ((void **) &A_d, sizeof(float)*NN);
  cudaMalloc ((void **) &B_d, sizeof(float)*NN);
  cudaMalloc ((void **) &C_d, sizeof(float)*NN);

  /* Initialize arrays a and b */
   for (int i=0; i<N;i++)
    for (int j=0;j<N;j++)
      {
       A[i*N+j]=(float) i; 
       B[i*N+j]=(float) (1-i);
      };

  /* Copy data from host memory to device memory */
  cudaMemcpy(A_d, A, sizeof(float)*NN, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, sizeof(float)*NN, cudaMemcpyHostToDevice);

  /* Compute the execution configuration */
  dim3 threadsPerBlock (8, 8);
  dim3 numBlocks( ceil ((float)(N)/threadsPerBlock.x), ceil ((float)(N)/threadsPerBlock.y) );
  
  double  time=clock();
  // Kernel Launch
  MatAdd <<<numBlocks, threadsPerBlock>>> (A_d, B_d, C_d, N);
  
  cudaDeviceSynchronize();
  time=(clock()-time)/CLOCKS_PER_SEC;

  /* Copy data from deveice memory to host memory */
  cudaMemcpy(C, C_d, sizeof(float)*NN, cudaMemcpyDeviceToHost);

  /* Print c */
  for (int i=0; i<N;i++)
    for (int j=0;j<N;j++)
      cout <<"C["<<i<<","<<j<<"]="<<C[i*N+j]<<endl;
      
 
  cout <<"Kernel Execution Time="<<time<<endl;
  
  /* Free the memory */
  free(A); free(B); free(C);
  cudaFree(A_d); cudaFree(B_d);cudaFree(C_d);

}
