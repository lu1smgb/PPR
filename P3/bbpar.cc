/* ******************************************************************** */
/*               Algoritmo Branch-And-Bound Secuencial                  */
/* ******************************************************************** */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <unistd.h>
#include <mpi.h>
#include "libbb.h"

unsigned int NCIUDADES;
int rank, size;
const bool difundirCota = false;

int main(int argc, char **argv)
{
	MPI::Init(argc, argv);
	switch (argc)
	{
	case 3:
		NCIUDADES = atoi(argv[1]);
		break;
	default:
		std::cerr << "La sintaxis es: bbseq <tama�o> <archivo>" << std::endl;
		exit(1);
		break;
	}

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &comunicadorCarga);
	MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &comunicadorCota);

	MPI_Comm_rank(comunicadorCarga, &rank);
	MPI_Comm_size(comunicadorCarga, &size);

	int **tsp0 = reservarMatrizCuadrada(NCIUDADES);
	tNodo nodo,	  // nodo a explorar
		lnodo,	  // hijo izquierdo
		rnodo,	  // hijo derecho
		solucion; // mejor solucion
	bool activo,  // condicion de fin
		nueva_U;  // hay nuevo valor de c.s.
	int U;		  // valor de c.s.
	int iteraciones = 0;
	tPila pila; // pila de nodos a explorar

	U = INFINITO;	 // inicializa cota superior
	InicNodo(&nodo); // inicializa estructura nodo

	std::cout << "rank=" << rank << "; size=" << size << std::endl;

	MPI_Barrier(comunicadorCarga);
	if (rank == 0) LeerMatriz(argv[2], tsp0); // lee matriz de fichero
	//MPI_Barrier(comunicadorCarga);
	MPI_Bcast(tsp0[0], NCIUDADES*NCIUDADES, MPI_INT, 0, comunicadorCarga);
	
	activo = !Inconsistente(tsp0);

	if (rank != 0) {
		std::cout << "1er equilibrado de rank=" <<rank<< std::endl;
		EquilibradoCarga(&pila, &activo);
		if (activo) pila.pop(nodo);
	}
	
	double t;
	if (rank == 0) t = MPI_Wtime();
	while (activo)
	{ // ciclo del Branch&Bound
		Ramifica(&nodo, &lnodo, &rnodo, tsp0);
		nueva_U = false;
		if (Solucion(&rnodo))
		{
			if (rnodo.ci() < U)
			{ // se ha encontrado una solucion mejor
				U = rnodo.ci();
				nueva_U = true;
				CopiaNodo(&rnodo, &solucion);
			}
		}
		else
		{ //  no es un nodo solucion
			if (rnodo.ci() < U)
			{ //  cota inferior menor que cota superior
				if (!pila.push(rnodo))
				{
					printf("Error: pila agotada\n");
					liberarMatriz(tsp0);
					exit(1);
				}
			}
		}
		if (Solucion(&lnodo))
		{
			if (lnodo.ci() < U)
			{ // se ha encontrado una solucion mejor
				U = lnodo.ci();
				nueva_U = true;
				CopiaNodo(&lnodo, &solucion);
			}
		}
		else
		{ // no es nodo solucion
			if (lnodo.ci() < U)
			{ // cota inferior menor que cota superior
				if (!pila.push(lnodo))
				{
					printf("Error: pila agotada\n");
					liberarMatriz(tsp0);
					exit(1);
				}
			}
		}
		if (difundirCota) DifusionCotaSuperior(&U);
		if (nueva_U) pila.acotar(U);
		EquilibradoCarga(&pila, &activo);
		if (activo) pila.pop(nodo);
		printf("P%d termina ciclo %d con U=%d\n", rank, iteraciones+1, U);
		iteraciones++;

	}
	//MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		t = MPI::Wtime() - t;
		printf("Solucion: \n");
		EscribeNodo(&solucion);
		std::cout << "Tiempo gastado= " << t << std::endl;
		std::cout << "Numero de iteraciones = " << iteraciones << std::endl
			<< std::endl;
		liberarMatriz(tsp0);
	}
	MPI::Finalize();
}