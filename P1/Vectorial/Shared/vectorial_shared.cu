/**
 * 
 * Programacion Paralela
 * Luis Miguel Guirado Bautista
 * Curso 2024/2025
 * Universidad de Granada
 * 
 * Vectorial
 * Parte 2 de la practica 1
 * Implementacion del kernel con memoria global y 
 * kernel con memoria compartida
 * 
*/

#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <math.h>

using namespace std;

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

// Warmup
__global__ void warm_up_gpu()
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float i, j = 1.0, k = 2.0;
  i = j + k;
  j += i + float(tid);
}

// Kernel que calcula el vector C version __shared__
__global__ void vectorialGPU_shared(float *A, float *B, float *C, int N)
{
  extern __shared__ float cdata[]; // Bsize * sizeof(float) * 3 (A,B y C)
  
  int tid = threadIdx.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N)
  {
    cdata[tid] = A[i]; // A
    cdata[blockDim.x + tid] = B[i]; // B
    cdata[2 * blockDim.x + tid] = 0; // C
    __syncthreads();
    
    // Calculamos valor de C[i]
    for (int j = 0; j < blockDim.x; j++)
    {
      float c1 = cdata[j] * i;
      float c2 = cdata[blockDim.x + j];
      float v;
      if ( (int)ceil(c1) % 2 == 0 )
        v = c1 + c2;
      else
        v = c1 - c2;
      cdata[2*blockDim.x+tid] += v;
      __syncthreads();
    }
    C[i] = cdata[2*blockDim.x+tid];
  }
}

// Kernel de reduccion para obtener los maximos de cada bloque
// IDEA: Podriamos usar el array D para guardar los maximos de cada bloque
// para poder hallar el maximo global en la CPU antes de realizar el calculo
// de los valores de D, asi reutilizamos memoria
__global__ void reducirMaximos(float *C, float *D, int N) {
  extern __shared__ int sdata[];
  int tid = threadIdx.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  sdata[tid] = (i < N) ? C[i] : 0;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid + s] > sdata[tid]) {
        sdata[tid] = sdata[tid+s];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    D[blockIdx.x] = sdata[0];
  }
}

// Kernel de reduccion para obtener las suma de los elementos de cada bloque
__global__ void reducirSumas(float *C, float *D, int N) {
  extern __shared__ int sdata[];
  int tid = threadIdx.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  sdata[tid] = (i < N) ? C[i] : 0;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    D[blockIdx.x] = sdata[0];
  }
}

// Version secuencial/CPU del algoritmo
// Devuelve el tiempo de ejecucion
float vectorialCPU(float *A, float *B, int N, int Bsize, int NBlocks)
{
  float *C = new float[N];

  // Time measurement
  double t1 = clock();

  // Compute C[i], d[K] and mx
  for (int k = 0; k < NBlocks; k++)
  {
    int istart = k * Bsize;
    int iend = istart + Bsize;
    for (int i = istart; i < iend; i++)
    {
      C[i] = 0.0;
      for (int j = istart; j < iend; j++)
      {
        float a = A[j] * i;
        if ((int) ceil(a) % 2 == 0)
          C[i] += a + B[j];
        else
          C[i] += a - B[j];
      }
    }
  }

  double t2 = (clock() - t1) / CLOCKS_PER_SEC;
  
  //// for (int i = 0; i < N; i++) {
  ////   if (Cgpu[i] != C[i]) {
  ////     cout << "INCOHERENCIA EN Ci=" << i << endl;
  ////     cout << "GPU -> " << Cgpu[i] << "; CPU -> " << C[i] << endl;
  ////     break;
  ////   }
  //// }

  delete[] C;

  return (float) t2;

}

// Encuentra el maximo de manera secuencial / CPU
// Usado para comprobar la veracidad de los resultados de la version GPU
float maximoCPU(float *C, int N) {
  int mx = C[0];
  for (int i = 1; i < N; i++)
    mx = (C[i] > mx) ? C[i] : mx;
  return mx;
}

// Encuentra las sumas de manera secuencial / CPU
// Usado para comprobar la veracidad de los resultados de la version GPU
void sumasCPU(float *C, int N, int numBlocks, float *D) {

  float Bsize = ceil((float)N / numBlocks);

  // Compute d[K]
  for (int k = 0; k < numBlocks; k++)
  {
    int istart = k * Bsize;
    int iend = istart + Bsize;
    D[k] = 0.0;
    for (int i = istart; i < iend; i++)
    {
      D[k] += C[i];
    }
  }

}

// MAIN
int main(int argc, char *argv[])
{
  unsigned int N, M;

  // Obtenemos N
  if (argc != 3)
  {
    cerr << "Sintaxis: " << argv[0] << " <N (tamaño de los arrays)> <M (tamaño de los bloques) in {64,128,256}>" << endl;
    return (-1);
  }
  N = atoi(argv[1]);
  M = atoi(argv[2]);
  if (M != 64 && M != 128 && M != 256)
  {
    cerr << "Valor de M no valido, debe ser uno de los siguientes {64, 128, 256}" << endl;
    return -1;
  }
  int numBlocks = ceil((float)N / M);

  // Get GPU information
  int devID = 0;
  cudaDeviceProp props;
  checkCuda(cudaGetDeviceProperties(&props, devID));
  cout << "****************************************************************************************" << endl;
  cout << "Using Device " << devID << ": " << props.name << "  with CUDA Compute Capability " << props.major << "." << props.minor << endl;
  cout << "****************************************************************************************" << endl
       << endl;
  checkCuda(cudaSetDevice(devID));

  // Warm up GPU
  warm_up_gpu<<<numBlocks, M>>>();

  // Mostramos el tamaño del array y el tamaño de los bloques
  cout << "N (tamaño de los vectores) = " << N << endl;
  cout << "M (tamaño de los bloques) = " << M << endl;

  // Declaracion e inicializacion de vectores y variables
  float *A, *B, *C, *D;
  unsigned int N_byte = N * sizeof(float); // Tamaño en bytes de A,B,C
  unsigned int nB_byte = numBlocks * sizeof(float); // Tamaño en bytes de D
  // Si usamos la flag por defecto == cudaMemAttachGlobal los resultados seran incorrectos
  checkCuda(cudaMallocManaged((void **)&A, N_byte, cudaMemAttachHost));
  checkCuda(cudaMallocManaged((void **)&B, N_byte, cudaMemAttachHost));
  checkCuda(cudaMallocManaged((void **)&C, N_byte, cudaMemAttachHost));
  checkCuda(cudaMallocManaged((void **)&D, nB_byte, cudaMemAttachHost));

  // Asignacion de valores a los vectores A y B
  for (int i = 0; i < N; i++)
  {
    A[i] = 0.95 * ((2 + 3 * i % 5) / (2 + i % 7));
    B[i] = 1.5 * ((1 + i % 3) / (1 + i % 5));
  }

  //// Copiamos A y B del host al device
  //// checkCuda(cudaMemcpy(A_dev, A_host, N_byte, cudaMemcpyHostToDevice));
  //// checkCuda(cudaMemcpy(B_dev, B_host, N_byte, cudaMemcpyHostToDevice));
  //// checkCuda(cudaMemcpy(C_dev, C_host, N_byte, cudaMemcpyHostToDevice));
  //// checkCuda(cudaMemcpy(D_dev, D_host, nB_byte, cudaMemcpyHostToDevice));

  // Declaramos variables de benchmark
  float timestamp, Tcpu, Tgpu;

  // Empezamos a computar C, empezamos benchmarking
  timestamp = clock();
  vectorialGPU_shared<<<numBlocks, M, M * sizeof(float) * 3>>>(A, B, C, N);
  cudaDeviceSynchronize();

  // Terminamos benchmarking
  Tgpu = (clock() - timestamp) / CLOCKS_PER_SEC;
  cout << "Tiempo GPU (shared): " << Tgpu << " seg" << endl;

  // cout << "Cgpu" << endl;
  // for (int i = 0; i < 4; i++) {
  //   cout << C[i] << " ";
  // }
  // cout << endl;

  //// Copiamos resultado a CPU
  ////checkCuda(cudaMemcpy(C_host, C_dev, nB_byte, cudaMemcpyDeviceToHost));

  // Ejecutamos version secuencial y hacemos benchmark, tambien comprobamos coherencia de resultados
  Tcpu = vectorialCPU(A, B, N, M, numBlocks);
  cout << "Tiempo CPU: " << Tcpu << " seg" << endl;

  // Mostramos ganancia de velocidad
  float S = Tcpu/Tgpu;
  cout << "Speedup (shared): " << S << endl;

  // Calculamos maximos de cada bloque
  reducirMaximos<<<numBlocks, M, nB_byte>>>(C, D, N);
  cudaDeviceSynchronize();
  ////checkCuda(cudaMemcpy(D_host, D_dev, nB_byte, cudaMemcpyDeviceToHost));

  // Obtenemos maximo global
  float mx = 0;
  for (int i = 0; i < numBlocks; i++) {
    if (mx < D[i]) mx = D[i];
  }
  cout << "Maximo GPU (shared): " << mx << endl;
  int mx_cpu = maximoCPU(C, N);
  cout << "Maximo CPU: " << maximoCPU(C, N) << endl; // Comparamos con CPU
  if (mx_cpu != mx) {
    cout << "ERROR mx: " << mx << "!=" << mx_cpu << endl;
  }

  // Calculamos sumas de cada bloque
  reducirSumas<<<numBlocks, M, nB_byte>>>(C, D, N);
  cudaDeviceSynchronize();
  ////checkCuda(cudaMemcpy(D_host, D_dev, nB_byte, cudaMemcpyDeviceToHost));
  //// Comparamos D con CPU
  float *D_cpu = new float[numBlocks];
  sumasCPU(C, N, numBlocks, D_cpu);
  //// for (int i = 0; i < numBlocks; i++) {
  ////   if (D_cpu[i] != D[i]) {
  ////     cout << "ERROR D[" << i << "]:" << round(D[i]) << "!=" << round(D_cpu[i]) << endl;
  ////   }
  //// }

  // !!! Hemos terminado !!!
  // Liberamos memoria ////en device
  checkCuda(cudaFree(A));
  checkCuda(cudaFree(B));
  checkCuda(cudaFree(C));
  checkCuda(cudaFree(D));
  delete[] D_cpu;

  //// Liberamos memoria en host
  //// delete[] A;
  //// delete[] B;
  //// delete[] C;
  //// delete[] D;

  return 0;

}