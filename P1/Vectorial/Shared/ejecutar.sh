#!/bin/bash

for bsize in 64 128 256; do
    for n in 20224 40192 80128 100096; do
        ./vectorial_shared $n $bsize
    done;
done;