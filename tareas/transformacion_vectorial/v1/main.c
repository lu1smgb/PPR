#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi/mpi.h>

#define ITERATIONS 100

void generateRandomArray(int *x, const int size) {
    x = (int*) malloc(sizeof(int) * size);
    for (int i=0; i < size; i++) {
        x[i] = 1 + rand() % 10;
    }
}

int main(int argc, char **argv) {

    if (argc < 2) {
        printf("%s <Tamanio del vector>\n", argv[0]);
        return -1;
    }

    int rank, size;
    int *x, *bloque;
    const int N = atoi(argv[1]);
    generateRandomArray(x, N);

    int i, j, k;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int p = ceil(N / size);

    for (k = 0; k < ITERATIONS; k++) {

        bloque = (int*) malloc(sizeof(int) * p);
        for (i = 0; i < p; i++) {
            bloque[i] = x[p * rank + i];
        }

        unsigned int proc_izq, proc_der;
        proc_izq = (rank - 1) % p;
        proc_der = (rank + 1) % p;

        MPI_Send(&)

    }

    MPI_Finalize();
    return 0;
}