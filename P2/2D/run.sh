#!/bin/bash
mpicxx -o main.out main.cpp
for p in 4 9; do
    for n in 300 600 900 1200 1500; do
        mpirun --hostfile hostfile -np $p ./main.out $n
    done;
done;