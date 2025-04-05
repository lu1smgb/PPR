/**
 * 
 * Programacion Paralela
 * Luis Miguel Guirado Bautista
 * Curso 2024/2025
 * Universidad de Granada
 * 
 * suma_matrices_columna.cu
 * Version del programa de suma de matrices en la que cada
 * hebra se encarga de una unica columna
 * 
*/

#include <iostream>
#include <fstream>
#include <time.h>

using namespace std;

/* Tamaño de las matrices cuadradas */
const int N = 5000;

/* De floyd.cu */
__global__ void warm_up_gpu()
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float i, j = 1.0, k = 2.0;
    i = j + k;
    j += i + float(tid);
}

/* Kernel de suma de matrices cuadradas */
__global__ void MatAdd(float *A, float *B, float *C, int N)
{
    int column = blockDim.x * blockIdx.x + threadIdx.x; // Indice de columna
    if (column >= N) return; // Comprobacion de indice
    for (int row = 0; row < N; row++) { // Realiza la suma en toda la fila
        int index = row * N + column;
        C[index] = A[index] + B[index];
        ////printf("Block %d, Thread %d, row=%d, column=%d, Index %d ; A[%d]=%f B[%d] = %f\n", blockIdx.x, threadIdx.x, row, column, index, index, A[index], index, B[index]);
    }

}

int main(int argc, char *argv[])
{

    /* Reserva de memoria en CPU */
    const int NN = N * N;
    float *A = (float *)malloc(NN * sizeof(float));
    float *B = (float *)malloc(NN * sizeof(float));
    float *C = (float *)malloc(NN * sizeof(float));

    /* Reserva de memoria en GPU */
    float *A_d, *B_d, *C_d;
    cudaMalloc((void **)&A_d, sizeof(float) * NN);
    cudaMalloc((void **)&B_d, sizeof(float) * NN);
    cudaMalloc((void **)&C_d, sizeof(float) * NN);

    /* Inicializar matrices */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = (float)i;
            B[i * N + j] = (float)(1 - i);
        };

    /* Copiar datos de CPU a GPU */
    cudaMemcpy(A_d, A, sizeof(float) * NN, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(float) * NN, cudaMemcpyHostToDevice);

    /* Definimos el numero de bloques y el numero de hilos por bloque */
    float threadsPerBlock = 256;
    float numBlocks = (float)ceil(N / threadsPerBlock);

    /* Informacion de la ejecucion */
    std::cout << "N (tamaño de la matriz): " << N << std::endl;
    std::cout << "Numero de bloques: " << numBlocks << std::endl;
    std::cout << "Numero de hilos por bloque: " << threadsPerBlock << std::endl;

    /* Calentar la GPU */
    std::cout << "Calentando GPU..." << std::endl;
    warm_up_gpu<<<numBlocks, threadsPerBlock>>>();

    /* Realizamos la operacion y medimos el tiempo */
    double time = clock();
    MatAdd<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();
    time = (clock() - time) / CLOCKS_PER_SEC;

    /* Copiamos la matriz C de la GPU en la CPU */
    cudaMemcpy(C, C_d, sizeof(float) * NN, cudaMemcpyDeviceToHost);

    /* Mostramos los resultados (solo con N razonablemente pequeño, para comprobar la veracidad de la operacion) */
    if (N <= 20)
    {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                cout << "C[" << i << "," << j << "]=" << C[i * N + j] << endl;
    }

    cout << "Tiempo de ejecucion del kernel = " << time << endl;

    /* Liberamos memoria */
    free(A);
    free(B);
    free(C);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
