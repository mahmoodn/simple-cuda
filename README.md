# simple-cuda

This repository contains simple and basic cuda implementations for beginners that what to access full and runable codes, change input/block/grid sizes and see how CUDA spreads threads on SMs. Steps are explained below:

## Creating inputs

The code in `create-inputs.cpp` writes random floating point numbers as a 1D array in two separate files, `inp1.txt` and `inp2.txt`. To change the number of elements, simply change the number in `#define N` [line](https://github.com/mahmoodn/simple-cuda/blob/main/create-inputs.cpp#L5C1-L5C16) and then build and run the code to create input files. Example:
```
sed -i 's/\#define N .*/\#define N 128/g' create_inputs.cpp
g++ -O3 -o create_inputs create_inputs.cpp
./create_inputs
```
