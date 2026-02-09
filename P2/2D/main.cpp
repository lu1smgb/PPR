/**
 * 
 * Programacion Paralela
 * Luis Miguel Guirado Bautista
 * Curso 2024/2025
 * Universidad de Granada
 * 
 * Practica 2
 * Descomposicion bidimensional
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
    const int Asize = N*N;
    const int sqrtP = sqrt(P);
    const int dimBloque = N / sqrtP;
    const int tamBloque = dimBloque * dimBloque;

    if (pid == 0) {
        std::cout << "Descomposicion 2D\n";
        std::cout << "N: " << N << " ; P: " << P << "\n";
        std::cout << "Tam. bloque: " << tamBloque << "\n";
        std::cout << "Dimension del bloque (n/sqrtP): " << dimBloque << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Declaramos los vectores y matrices de cada proceso
    float *A_p = new float[tamBloque]; // Submatriz A de proceso
    float *x_p = new float[dimBloque]; // Subvector x de proceso
    float *y_p = new float[dimBloque]; // Subvector x de proceso
    float *x; // Vector x
    float *y; // Vector y
    float *y_ver; // Vector para verificar resultados
    float *y_reduc; // Vector de recepcion en la reduccion

    float tiempo_secuencial;
    float tiempo_paralelo;

    MPI_Datatype MPI_BLOQUE;
    float *buf_envio;
    
    if (pid == 0) {

        float *A = new float[Asize];
        x = new float[N];
        y = new float[N];
        y_ver = new float[N];
        buf_envio = new float[Asize];

        for (int i=0; i < N; i++) {
            x[i] = (float) ( (1 + rand() % 5) + 0.01 * (rand() % 100) );
            y_ver[i] = 0;
        }
        
        // Iniciamos los valores de la matriz A y del vector x en P0
        // Tambien calculamos un vector y para verificar los resultados
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
        tiempo_secuencial = (t2-t1) / CLOCKS_PER_SEC;

        // Empezamos a distribuir la matriz A entre los procesos mediante bloques
        MPI_Type_vector(dimBloque, dimBloque, N, MPI_FLOAT, &MPI_BLOQUE);
        MPI_Type_commit(&MPI_BLOQUE);

        for (int i=0, posicion=0; i < P; i++) {
            int fila_P = i / sqrtP, columna_P = i % sqrtP;
            int comienzo = (fila_P*dimBloque*N) + (columna_P*dimBloque);
            ////std::cout << "Fila: " << fila_P << " ; Columna: " << columna_P << "\n";
            MPI_Pack(&A[comienzo], 1, MPI_BLOQUE, buf_envio, sizeof(float)*Asize, &posicion, MPI_COMM_WORLD);
        }

        MPI_Type_free(&MPI_BLOQUE);

        delete [] A;

    }
    // Procesos esperan a que esten todos los datos iniciales listos
    MPI_Barrier(MPI_COMM_WORLD);

    // Repartimos los bloques de A entre los procesos
    //// if (pid == 0) std::cout << "Scatter de buf_envio a A_p" << "\n";
    MPI_Scatter(buf_envio, sizeof(float)*tamBloque, MPI_PACKED, A_p, sizeof(float)*tamBloque, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Repartimos x entre los procesos de la diagonal
    // Primero tenemos que crear un comunicador que contenga a estos procesos
    //// if (pid == 0) std::cout << "Init comm_diagonal" << "\n";
    MPI_Comm comm_diagonal;
    int fila = pid / sqrtP;
    int columna = pid % sqrtP;
    int diag_color = (fila == columna) ? 1 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, diag_color, columna, &comm_diagonal);
    int pid_diag;
    if (comm_diagonal != MPI_COMM_NULL) {
        y_reduc = new float[dimBloque];
        MPI_Comm_rank(comm_diagonal, &pid_diag);
        //// std::cout << "Fila=" << fila << "; Columna=" << columna << "; IDiag=" << pid_diag << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //// std::cout << "A_" << fila << columna << ": ";
    //// for (int i=0; i < tamBloque; i++) {
    ////     if (i >= 3) {
    ////         std::cout << "..." << "\n";
    ////         break;
    ////     }
    ////     std::cout << A_p[i] << " ";
    //// }
    //// MPI_Barrier(MPI_COMM_WORLD);
    
    // Creamos los comunicadores de columnas
    //// if (pid == 0) std::cout << "Init comm_columnas" << "\n";
    MPI_Comm comm_columnas[sqrtP];
    MPI_Comm_split(MPI_COMM_WORLD, columna, fila, &comm_columnas[columna]);
    int pid_columna;
    MPI_Comm_rank(comm_columnas[columna], &pid_columna);

    // Creamos los comunicadores de filas
    //// if (pid == 0) std::cout << "Init comm_filas" << "\n";
    MPI_Comm comm_filas[sqrtP];
    MPI_Comm_split(MPI_COMM_WORLD, fila, columna, &comm_filas[fila]);
    int pid_fila;
    MPI_Comm_rank(comm_filas[fila], &pid_fila);
    //// std::cout << "Fila=" << fila << "; Columna=" << columna << "; IDcolumna=" << pid_columna << "; IDfila=" << pid_fila << "\n";
    MPI_Barrier(MPI_COMM_WORLD);

    // Scatter de x en la diagonal
    //// if (pid == 0) std::cout << "Scatter de x en la diagonal" << "\n";
    if (comm_diagonal != MPI_COMM_NULL)
        MPI_Scatter(x, dimBloque, MPI_FLOAT, x_p, dimBloque, MPI_FLOAT, 0, comm_diagonal);

    //// if (pid == 0) {
    ////     std::cout << "x: ";
    ////     for (int i=0; i < N; i++) {
    ////         if (i >= 3) {
    ////             std::cout << "..." << "\n";
    ////             break;
    ////         }
    ////         std::cout << x[i] << " ";
    ////     }
    //// }
    //// MPI_Barrier(MPI_COMM_WORLD);

    //// if (comm_diagonal != MPI_COMM_NULL) {
    ////     std::cout << "x_d" << pid_diag << ": ";
    ////     for (int i=0; i < dimBloque; i++) {
    ////         if (i >= 3) {
    ////             std::cout << "..." << "\n";
    ////             break;
    ////         }
    ////         std::cout << x_p[i] << " ";
    ////     }    
    //// }
    //// MPI_Barrier(MPI_COMM_WORLD);

    // Broadcast en cada columna, por cada elemento de la diagonal
    MPI_Barrier(comm_columnas[columna]);
    //// if (pid == 0) std::cout << "Broadcast de x_p en las columnas" << "\n";
    MPI_Bcast(x_p, dimBloque, MPI_FLOAT, columna, comm_columnas[columna]);

    //// std::cout << "x_c" << columna << ": ";
    //// for (int i=0; i < dimBloque; i++) {
    ////     if (i >= 3) {
    ////         std::cout << "..." << "\n";
    ////         break;
    ////     }
    ////     std::cout << x_p[i] << " ";
    //// }
    //// MPI_Barrier(MPI_COMM_WORLD);

    // Empezamos medicion de tiempo
    MPI_Barrier(MPI_COMM_WORLD);
    //// if (pid == 0) std::cout << "Empieza computo" << "\n";
    double t1 = MPI_Wtime();

    // Calculamos los valores del vector y
    for (int i=0; i < dimBloque; i++) {
        y_p[i] = 0;
        for (int j=0; j < dimBloque; j++) {
            int ij = i * dimBloque + j;
            //// if (ij >= tamBloque) {
            ////     printf("ij=%d, i=%d, j=%d\n", ij, i, j);
            //// }
            y_p[i] += A_p[ij] * x_p[j];
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Terminamos medicion de tiempo
    double t2 = MPI_Wtime();
    tiempo_paralelo = t2 - t1;
    //// if (pid == 0) std::cout << "Termina computo" << "\n";
    MPI_Barrier(MPI_COMM_WORLD);

    //// std::cout << "y_" << fila << columna << ": ";
    //// for (int i=0; i < dimBloque; i++) {
    ////     if (i >= 3) {
    ////         std::cout << "..." << "\n";
    ////         break;
    ////     }
    ////     std::cout << y_p[i] << " ";
    //// }
    //// MPI_Barrier(MPI_COMM_WORLD);

    // Reduction en cada fila
    //// if (pid == 0) std::cout << "Reduction de y_p a y_reduc en las filas hacia Pdiagonal" << "\n";
    //// MPI_Barrier(MPI_COMM_WORLD);
    if (comm_filas[fila] != MPI_COMM_NULL) 
        MPI_Reduce(y_p, y_reduc, dimBloque, MPI_FLOAT, MPI_SUM, fila, comm_filas[fila]);
    if (comm_diagonal != MPI_COMM_NULL) 
        MPI_Barrier(comm_diagonal);

    //// if (comm_diagonal != MPI_COMM_NULL) {
    ////     std::cout << "y_reduc" << fila << ": ";
    ////     for (int i=0; i < dimBloque; i++) {
    ////         if (i >= 3) {
    ////             std::cout << "..." << "\n";
    ////             break;
    ////         }
    ////         std::cout << y_reduc[i] << " ";
    ////     }
    //// }
    //// MPI_Barrier(MPI_COMM_WORLD);

    // Reunimos los resultados de cada proceso a P0
    //// if (pid == 0) std::cout << "Gather de y_reduc a y en la diagonal hacia P0" << "\n";
    if (comm_diagonal != MPI_COMM_NULL)
        MPI_Gather(y_reduc, dimBloque, MPI_FLOAT, y, dimBloque, MPI_FLOAT, 0, comm_diagonal);
    MPI_Barrier(MPI_COMM_WORLD);

    //// if (pid == 0) {
    ////     std::cout << "y_ver: ";
    ////     for (int i=0; i < N; i++) {
    ////         if (i >= 3) {
    ////             std::cout << "..." << "\n";
    ////             break;
    ////         }
    ////         std::cout << y_ver[i] << " ";
    ////     }
    //// }
    //// MPI_Barrier(MPI_COMM_WORLD);

    //// if (pid == 0) {
    ////     std::cout << "y: ";
    ////     for (int i=0; i < N; i++) {
    ////         if (i >= 3) {
    ////             std::cout << "..." << "\n";
    ////             break;
    ////         }
    ////         std::cout << y[i] << " ";
    ////     }
    //// }
    //// MPI_Barrier(MPI_COMM_WORLD);

    // Mostramos resultados
    if (pid == 0) {
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
        // Liberamos memoria especifica de P0
        delete [] x;
        delete [] y;
        delete [] y_ver;
        delete [] buf_envio;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Liberamos buffer de reduccion de los procesos de la diagonal
    if (comm_diagonal != MPI_COMM_NULL)
        delete [] y_reduc;

    // Liberamos memoria de cada proceso y terminamos
    delete [] x_p;
    delete [] y_p;
    delete [] A_p;
    MPI_Finalize();
    return 0;
}