# simple-cuda

This repository contains basic CUDA implementations for beginners that want to access full and runable codes, change input/block/grid sizes and see how CUDA spreads threads on SMs for debugging and better understanding of CUDA structure. Steps are explained below:

## Creating inputs

The code in `create-inputs.cpp` writes random floating point numbers as a 1D array in two separate files, `inp1.txt` and `inp2.txt`. The first file is used for simple vector addition and both files are used as inputs to matrix multiplication. Simply, build `create_inputs` and write the number of elements as the command line options. Example:

```
$ g++ -O3 -o create_inputs create_inputs.cpp
$ ./create_inputs 16
```

## Vector addition

The code in `addition.cu` reads `inp1.txt` and performs a simple vector addition on the GPU. Simply, build the code and specify the block size in input command. Example:

```
$ nvcc -o addition -Xptxas -O3 addition.cu
$ ./addition 16
--------------------------------------------------
Device Number: 0
  Device name: NVIDIA GeForce RTX 3080
  SM count: 68
  Cores per SM: 128
  Max Blocks per SM: 16
  Max Threads per Block: 1024
  Max Threads per SM: 1536
--------------------------------------------------
Read 16 elements from inp1.txt

SM(0) Block(0,0,0) Thread(0,0,0) -> 9.45 + 1
SM(0) Block(0,0,0) Thread(1,0,0) -> 7.97 + 1
SM(0) Block(0,0,0) Thread(2,0,0) -> 4.75 + 1
SM(0) Block(0,0,0) Thread(3,0,0) -> 1.24 + 1
SM(0) Block(0,0,0) Thread(4,0,0) -> -4.10 + 1
SM(0) Block(0,0,0) Thread(5,0,0) -> -7.91 + 1
SM(0) Block(0,0,0) Thread(6,0,0) -> 3.03 + 1
SM(0) Block(0,0,0) Thread(7,0,0) -> 0.80 + 1
SM(0) Block(0,0,0) Thread(8,0,0) -> 5.59 + 1
SM(0) Block(0,0,0) Thread(9,0,0) -> 4.20 + 1
SM(0) Block(0,0,0) Thread(10,0,0) -> 3.82 + 1
SM(0) Block(0,0,0) Thread(11,0,0) -> -6.93 + 1
SM(0) Block(0,0,0) Thread(12,0,0) -> 6.32 + 1
SM(0) Block(0,0,0) Thread(13,0,0) -> 9.97 + 1
SM(0) Block(0,0,0) Thread(14,0,0) -> 2.16 + 1
SM(0) Block(0,0,0) Thread(15,0,0) -> -4.94 + 1

addABC elapsed time : 0.899072 ms

Result:
10.4488 8.9695 5.7455 2.2354 -3.101 -6.9092 4.0279 1.7959 6.5887 5.1981 4.8159 -5.9302 7.3159 10.9748 3.1615 -3.9358 
Done


$ ./addition 8
--------------------------------------------------
Device Number: 0
  Device name: NVIDIA GeForce RTX 3080
  SM count: 68
  Cores per SM: 128
  Max Blocks per SM: 16
  Max Threads per Block: 1024
  Max Threads per SM: 1536
--------------------------------------------------
Read 16 elements from inp1.txt

SM(2) Block(1,0,0) Thread(0,0,0) -> 5.59 + 1
SM(2) Block(1,0,0) Thread(1,0,0) -> 4.20 + 1
SM(2) Block(1,0,0) Thread(2,0,0) -> 3.82 + 1
SM(2) Block(1,0,0) Thread(3,0,0) -> -6.93 + 1
SM(2) Block(1,0,0) Thread(4,0,0) -> 6.32 + 1
SM(2) Block(1,0,0) Thread(5,0,0) -> 9.97 + 1
SM(2) Block(1,0,0) Thread(6,0,0) -> 2.16 + 1
SM(2) Block(1,0,0) Thread(7,0,0) -> -4.94 + 1
SM(0) Block(0,0,0) Thread(0,0,0) -> 9.45 + 1
SM(0) Block(0,0,0) Thread(1,0,0) -> 7.97 + 1
SM(0) Block(0,0,0) Thread(2,0,0) -> 4.75 + 1
SM(0) Block(0,0,0) Thread(3,0,0) -> 1.24 + 1
SM(0) Block(0,0,0) Thread(4,0,0) -> -4.10 + 1
SM(0) Block(0,0,0) Thread(5,0,0) -> -7.91 + 1
SM(0) Block(0,0,0) Thread(6,0,0) -> 3.03 + 1
SM(0) Block(0,0,0) Thread(7,0,0) -> 0.80 + 1

addABC elapsed time : 1.21955 ms

Result:
10.4488 8.9695 5.7455 2.2354 -3.101 -6.9092 4.0279 1.7959 6.5887 5.1981 4.8159 -5.9302 7.3159 10.9748 3.1615 -3.9358 
Done
```

The above commands read a vector with 32 elements and performs the addition with block sizes of 16 and 8. In the first command, only one block with 16 threads is created and that is offloaded to SM(0). In the second command, two blocks each with 8 threads are created and they are offloaded on SM_0 and SM_2. Note that if you want to compile for a specific architecture you can add `-arch=sm_XX`. See this [page](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) to find the corresponding SM number for an architecture.
