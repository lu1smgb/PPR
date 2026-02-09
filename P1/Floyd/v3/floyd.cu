/**
 * 
 * Programacion Paralela
 * Luis Miguel Guirado Bautista
 * Curso 2024/2025
 * Universidad de Granada
 * 
 * Floyd V3
 * Version del programa que nos proporciona la
 * media, el minimo y el maximo de todas las
 * distancias de la matriz resultante
 * 
*/

#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "Graph.h"

using namespace std;

#define BLOCKSIZE 64

//*************************************************
// Function for checking CUDA runtime API results
//*************************************************
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess)
  {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

//*************************************************
// GPU WARMUP KERNEL
// ************************************************
__global__ void warm_up_gpu()
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float i, j = 1.0, k = 2.0;
  i = j + k;
  j += i + float(tid);
}

//**************************************************************************
__global__ void floyd_kernel(int *M, const int nverts, const int k)
{
  int ij = threadIdx.x + blockDim.x * blockIdx.x;
  int i = ij / nverts;
  int j = ij - i * nverts;
  if (i < nverts && j < nverts)
  {
    int Mij = M[ij];
    if (i != j && i != k && j != k)
    {
      int Mikj = M[i * nverts + k] + M[k * nverts + j];
      Mij = (Mij > Mikj) ? Mikj : Mij;
      M[ij] = Mij;
    }
  }
}
//**************************************************************************

//**************************************************************************
//* Implementacion utilizando grid y bloques bidimensionales
__global__ void floyd_kernel_2D(int *M, const int nverts, const int k)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;                // Indice de columna
  int i = blockIdx.y * blockDim.y + threadIdx.y;                // Indice de fila
  if (i < nverts && j < nverts && i != j && i != k && j != k) { // Guarda de comprobacion de indices
    int ij = i * nverts + j;                                    // Indice global (1D)
    int Mij = M[ij];                                            // Valores candidatos
    int Mikj = M[i * nverts + k] + M[k * nverts + j];           // ...
    M[ij] = (Mij > Mikj) ? Mikj : Mij;                          // Obtenemos el minimo y lo escribimos en M
  }
}
//**************************************************************************

//**************************************************************************
//* Reduccion dividida por bloques, luego se ha de realizar la suma y division final en CPU
// Vamos a realizar la reduccion a suma, minimo y maximo a la vez
__global__ void reducir(int *M_in, const int nverts) {
  
  // Ahora se utiliza memoria compartida el triple de grande, una
  // vez por tipo de reduccion (en nuestro caso 3)
  extern __shared__ int sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Suma
  sdata[tid] = (i < nverts*nverts && M_in[i] < INF) ? M_in[i] : 0;
  // Minimo
  sdata[blockDim.x+tid] = (i < nverts*nverts && M_in[i] > 0) ? M_in[i] : INF;
  // Maximo
  sdata[2*blockDim.x+tid] = (i < nverts*nverts && M_in[i] < INF) ? M_in[i] : 0;

  __syncthreads();

  // Reduccion de los datos del bloque
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      // Suma
      sdata[tid] += sdata[tid + s];
      // Minimo
      sdata[blockDim.x+tid] = (sdata[blockDim.x+tid] > sdata[blockDim.x+tid + s]) ? sdata[blockDim.x+tid + s] : sdata[blockDim.x+tid];
      // Maximo
      sdata[2*blockDim.x+tid] = (sdata[2*blockDim.x+tid] < sdata[2*blockDim.x+tid + s]) ? sdata[2*blockDim.x+tid + s] : sdata[2*blockDim.x+tid];
    }
    __syncthreads();
  }

  if (tid == 0) {
    // Se guarda suma, minimo y maximo de cada bloque de manera consecutiva
    // en la matriz de entrada (operacion terminal)
    M_in[3 * blockIdx.x] = sdata[0];
    ////printf("sum in block %d = %d\n", blockIdx.x, sdata[0]);
    M_in[3 * blockIdx.x + 1] = sdata[blockDim.x];
    ////printf("min in block %d = %d\n", blockIdx.x, sdata[blockDim.x]);
    M_in[3 * blockIdx.x + 2] = sdata[2*blockDim.x];
    ////printf("max in block %d = %d\n", blockIdx.x, sdata[2*blockDim.x]);
  }

}
//**************************************************************************

//**************************************************************************
// ************  MAIN FUNCTION *********************************************
int main(int argc, char *argv[])
{

  double time, Tcpu, Tgpu;

  if (argc != 2)
  {
    cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
    return (-1);
  }

  // Get GPU information
  int devID = 0;
  cudaDeviceProp props;

  cout << "Using Device " << devID << endl;
  cout << "....................................................." << endl
       << endl;
  checkCuda(cudaGetDeviceProperties(&props, devID));
  cout << "****************************************************************************************" << endl;
  cout << "Using Device " << devID << ": " << props.name << "  with CUDA Compute Capability " << props.major << "." << props.minor << endl;
  cout << "****************************************************************************************" << endl
       << endl;

  checkCuda(cudaSetDevice(devID));

  cout << "********************************************* Warming up GPU!!!" << endl;
  // Warm up GPU
  warm_up_gpu<<<(10000 + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>();

  // Declaration of the Graph object
  Graph G;

  // Read the Graph
  G.lee(argv[1]);

  // cout << "The input Graph:"<<endl;
  // G.imprime();
  const int nverts = G.vertices;
  const int niters = nverts;
  const int nverts2 = nverts * nverts;

  int *c_Out_M = new int[nverts2];
  int size = nverts2 * sizeof(int);
  int *d_In_M = NULL;

  //! Modificacion:
  //! Declaramos que los datos de entrada se almacenen como memoria fijada en el host
  checkCuda(cudaMallocHost((void **)&d_In_M, size));

  // Get the integer 2D array for the dense graph
  int *A = G.Get_Matrix();

  //**************************************************************************
  // GPU phase
  //**************************************************************************

  dim3 threadsPerBlock(ceil(sqrt(BLOCKSIZE)), ceil(sqrt(BLOCKSIZE)));
  dim3 blocksPerGrid(
    ceil((float) nverts / threadsPerBlock.x),
    ceil((float) nverts / threadsPerBlock.y)
  );
  std::cout << "Tamaño de M: (" << nverts << ", " << nverts << ")" << std::endl;
  std::cout << "Tamaño de bloque: (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")" << std::endl;
  std::cout << "Tamaño de grid: (" << blocksPerGrid.x << ", " << blocksPerGrid.y << ")" << std::endl;

  time = clock();
  checkCuda(cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice));

  // Main Loop
  for (int k = 0; k < niters; k++)
  {
    // printf("CUDA kernel launch \n");
    // Kernel Launch
    floyd_kernel_2D<<<blocksPerGrid, threadsPerBlock>>>(d_In_M, nverts, k);
    checkCuda(cudaGetLastError());
  }
  checkCuda(cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost));

  Tgpu = (clock() - time) / CLOCKS_PER_SEC;

  cout << "Time spent on GPU= " << Tgpu << endl
       << endl;

  //**************************************************************************
  // CPU phase
  //**************************************************************************

  time = clock();

  // BUCLE PPAL DEL ALGORITMO
  int inj, in, kn;
  for (int k = 0; k < niters; k++)
  {
    kn = k * nverts;
    for (int i = 0; i < nverts; i++)
    {
      in = i * nverts;
      for (int j = 0; j < nverts; j++)
        if (i != j && i != k && j != k)
        {
          inj = in + j;
          A[inj] = min(A[in + k] + A[kn + j], A[inj]);
        }
    }
  }

  Tcpu = (clock() - time) / CLOCKS_PER_SEC;
  cout << "Time spent on CPU= " << Tcpu << endl
       << endl;
  cout << "....................................................." << endl
       << endl;

  cout << "Speedup TCPU/TGPU= " << Tcpu / Tgpu << endl;
  cout << "....................................................." << endl
       << endl;

  bool errors = false;
  // Error Checking (CPU vs. GPU)
  for (int i = 0; i < nverts; i++)
    for (int j = 0; j < nverts; j++)
      if (abs(c_Out_M[i * nverts + j] - G.arista(i, j)) > 0)
      {
        cout << "Error (" << i << "," << j << ")   " << c_Out_M[i * nverts + j] << "..." << G.arista(i, j) << endl;
        errors = true;
      }

  if (!errors)
  {
    cout << "....................................................." << endl;
    cout << "WELL DONE!!! No errors found ............................" << endl;
    cout << "....................................................." << endl
         << endl;
  }

  //**************************************************************************
  // Fase de reduccion, minimos y maximos
  //**************************************************************************

  //G.imprime();

  // El numero de bloques y hebras sera igual al numero de vertices
  dim3 threadsPerBlock_r(nverts, 1);
  dim3 numBlocks_r(nverts, 1);
  //std::cout << "tpB=" << threadsPerBlock_r.x << std::endl;
  //std::cout << "bpG=" << numBlocks_r.x << std::endl;
  int smemSize = threadsPerBlock_r.x * 3 * sizeof(int);
  reducir<<<numBlocks_r, threadsPerBlock_r, smemSize>>>(d_In_M, nverts);
  checkCuda(cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  long int sum = 0;
  float min = INF, max = 0;
  for (int i=0; i < numBlocks_r.x; i++) {
    sum += c_Out_M[3*i];
    min = (c_Out_M[3*i+1] < min) ? c_Out_M[3*i+1] : min;
    max = (c_Out_M[3*i+2] > max) ? c_Out_M[3*i+2] : max; 
  }
  float media = sum / numBlocks_r.x;

  std::cout << "--- Estadisticas ---" << std::endl;
  std::cout << "suma = " << sum << std::endl;
  std::cout << "media = " << media << std::endl;
  std::cout << "minimo = " << min << std::endl;
  std::cout << "maximo = " << max << std::endl;

  //! Modificacion:
  //! Tambien hay que liberar la memoria pinned
  checkCuda(cudaFreeHost(d_In_M));
  delete[] c_Out_M;
}
