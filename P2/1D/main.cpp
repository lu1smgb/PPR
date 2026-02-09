/**
 * 
 * Programacion Paralela
 * Luis Miguel Guirado Bautista
 * Curso 2024/2025
 * Universidad de Granada
 * 
 * Practica 2
 * Descomposicion unidimensional
 * 
*/

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <mpi/mpi.h>

int main(int argc, char **argv) {

    int pid, P;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    // Dimension de la matriz y del vector
    const int N = abs(atoi(argv[1]));

    // Tamaño de los bloques en los que se va a dividir la matriz A
    const int parteEntera = N / P;
    const int resto = N % P;
    if (pid == 0) {
        std::cout << "Descomposicion 1D\n";
        std::cout << "N: " << N << " ; P: " << P << "\n";
        if (resto > 0) {
            std::cout << "[!!!] Resto: " << resto << "\n";
            std::cout << "Primeros " << resto << " procesos tomaran 1 fila adicional" << "\n";
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int *filas_bloques = new int[P]; // Numero de filas de cada bloque
    int *tam_bloques = new int[P]; // Tamanio de bloque en cada proceso
    int *offset_bloques = new int[P]; // Desplazamiento de cada A_p con respecto a A
    int *offset_yp = new int[P]; // Desplazamiento de cada y_p con respecto a y
    // Estimacion de la distribucion de filas
    for (int i=0; i < P; i++) {
        filas_bloques[i] = (i < resto) ? parteEntera+1 : parteEntera;
        tam_bloques[i] = N * filas_bloques[i];
        offset_bloques[i] = (i == 0) ? 0 : offset_bloques[i-1] + tam_bloques[i-1];
        offset_yp[i] = (i == 0) ? 0 : offset_yp[i - 1] + filas_bloques[i - 1];
    }

    // Declaramos los vectores y matrices
    float *A = new float[N*N]; // Matriz global
    float *A_p = new float[tam_bloques[pid]]; // Submatriz A de proceso
    float *y_p = new float[filas_bloques[pid]]; // Subvector y de proceso
    float *x = new float[N]; // Vector x
    float *y = new float[N]; // Vector y calculado de manera paralela
    float *y_ver = new float[N]; // Vector y calculado de manera secuencial por P0
    MPI_Barrier(MPI_COMM_WORLD);

    float tiempo_secuencial = 0;
    float tiempo_paralelo = 0;

    // Iniciamos los valores de la matriz A y del vector x en P0
    // Tambien calculamos un vector y_ver para verificar los resultados
    if (pid == 0) {

        for (int i=0; i < N; i++) {
            x[i] = (float) ( (1 + rand() % 5) + 0.01 * (rand() % 100) );
            y_ver[i] = 0;
        }

        for (int i=0; i < N; i++) {
            for (int j=0; j < N; j++) {
                int ij = i * N + j;
                A[ij] = (float) ( (1 + rand() % 5) + 0.01 * (rand() % 100) );
            }
        }

        float t1 = clock();

        for (int i=0; i < N; i++) {
            for (int j=0; j < N; j++) {
                int ij = i * N + j;
                y_ver[i] += A[ij] * x[j];
            }
        }

        float t2 = clock();
        tiempo_secuencial = (t2 - t1) / CLOCKS_PER_SEC;

        //// std::cout << "A: ";
        //// for (int i=0; i < N*N; i++) {
        ////     if (i >= 10) {std::cout << "..."; break;}
        ////     std::cout << A[i] << " ";
        //// }
        //// std::cout << std::endl;

        //// std::cout << "y_ver: ";
        //// for (int i=0; i < N; i++) {
        ////     if (i >= 10) {std::cout << "..."; break;}
        ////     std::cout << y_ver[i] << " ";
        //// }
        //// std::cout << std::endl;

        //// std::cout << "Se ha terminado de generar A y x" << std::endl;

    }
    // Procesos esperan a que esten todos los datos iniciales listos
    MPI_Barrier(MPI_COMM_WORLD);

    // Broadcast del vector x y Scatter de la matriz A
    // Empezamos medicion de tiempo
    MPI_Bcast(x, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    
    //// if (pid == 0) {
    ////     printf("Filas bloques: ");
    ////     for (int i=0; i < P; i++) {
    ////         printf("%d ", filas_bloques[i]);
    ////     }
    ////     std::cout << std::endl;
    ////     printf("Tam bloques: ");
    ////     for (int i=0; i < P; i++) {
    ////         printf("%d ", tam_bloques[i]);
    ////     }
    ////     std::cout << std::endl;
    ////     printf("Desplazam..: ");
    ////     for (int i=0; i < P; i++) {
    ////         printf("%d ", offset_bloques[i]);
    ////     }
    ////     std::cout << std::endl;
    //// }
    //// MPI_Barrier(MPI_COMM_WORLD);

    // Scatter irregular de la matriz de A entre los procesos
    MPI_Scatterv(A, tam_bloques, offset_bloques, MPI_FLOAT, A_p, tam_bloques[pid], MPI_FLOAT, 0, MPI_COMM_WORLD);
    //// MPI_Scatter(A, tamBloque, MPI_FLOAT, A_p, tamBloque, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    double t1 = MPI_Wtime();

    //// Mostramos el vector x
    //// if (pid == 0) {
    ////     std::cout << "x: ";
    ////     for (int i=0; i < N; i++) {
    ////         if (i >= 10) {std::cout << "..."; break;}
    ////         std::cout << x[i] << " ";
    ////     }
    ////     std::cout << std::endl;
    //// }
    MPI_Barrier(MPI_COMM_WORLD);

    //// Mostramos los elementos de A repartidos en cada proceso
    //// std::cout << "A_" << pid << ": ";
    //// for (int i=0; i < tam_bloques[pid]; i++) {
    ////     if (i >= 10) {std::cout << "..."; break;}
    ////     std::cout << A_p[i] << " ";
    //// }
    //// std::cout << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    ////if (pid == 0) std::cout << std::endl << "Computando vector y" << std::endl;
    // Calculamos los valores del vector y
    for (int i=0; i < filas_bloques[pid]; i++) {
        y_p[i] = 0;
        for (int j=0; j < N; j++) {
            int ij = i * N + j;
            y_p[i] += A_p[ij] * x[j];
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //// Mostramos los resultados dentro del proceso (aun no es una salida valida)
    //// std::cout << "y_" << pid << ": ";
    //// for (int i=0; i < filas_bloques[pid]; i++) {
    ////     if (i >= 10) {std::cout << "..."; break;}
    ////     std::cout << y_p[i] << " ";
    //// }
    //// std::cout << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // Reunimos los resultados de cada proceso
    double t2 = MPI_Wtime();
    tiempo_paralelo = t2 - t1;
    MPI_Gatherv(y_p, filas_bloques[pid], MPI_FLOAT, y, filas_bloques, offset_yp, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //// MPI_Gather(y_p, filasPorBloque, MPI_FLOAT, y, filasPorBloque, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Verificamos integridad del resultado
    if (pid == 0) {

        //// std::cout << "y: ";
        //// for (int i=0; i < N; i++) {
        ////     if (i >= 10) {std::cout << "..."; break;}
        ////     std::cout << y[i] << " ";
        //// }
        //// std::cout << std::endl;
        std::cout << "Tiempo secuencial: " << tiempo_secuencial << "\n";
        std::cout << "Tiempo paralelo: " << tiempo_paralelo << "\n";
        const float ganancia = tiempo_secuencial / tiempo_paralelo;
        std::cout << "Ganancia: " << ganancia << "\n";
        const float margen = 0.1;
        for (int i=0; i < N; i++) {
            float diff = std::fabs(y[i] - y_ver[i]);
            if (diff > margen) {
                printf("[X] Error de integridad en y[%d] -> %.5f != %.5f (%f)", i, y[i], y_ver[i], diff);
                break;
            }
            if (i == N-1) {
                printf("[*] Todos los resultados son coherentes (margen=%.2f)", margen);
            }
        }
        printf("\n\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Terminamos
    delete [] A;
    delete [] y;
    delete [] y_ver;
    delete [] filas_bloques;
    delete [] tam_bloques;
    delete [] offset_bloques;
    delete [] offset_yp;
    delete [] x;
    delete [] y_p;
    delete [] A_p;
    MPI_Finalize();
    return 0;
}