# simple-cuda

This repository contains basic CUDA implementations for beginners that want to access full and runable codes, change input/block/grid sizes and see how CUDA spreads threads on SMs for debugging and better understanding of CUDA structure. Steps are explained below:

## Creating inputs

The code in `create-inputs.cpp` writes random floating point numbers as a 1D array in two separate files, `inp1.txt` and `inp2.txt`. The first file is used for simple vector addition and both files are used as inputs to matrix multiplication. Simply, build `create_inputs` and write the number of elements as the command line options. Example:
```
g++ -O3 -o create_inputs create_inputs.cpp
./create_inputs 1024
```
