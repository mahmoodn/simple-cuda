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

SM(0) | Block(0,0,0) | Thread(0,0,0) | TID(0) -> -9.25 + 1
SM(0) | Block(0,0,0) | Thread(1,0,0) | TID(1) -> 8.08 + 1
SM(0) | Block(0,0,0) | Thread(2,0,0) | TID(2) -> 1.30 + 1
SM(0) | Block(0,0,0) | Thread(3,0,0) | TID(3) -> -6.62 + 1
SM(0) | Block(0,0,0) | Thread(4,0,0) | TID(4) -> -7.49 + 1
SM(0) | Block(0,0,0) | Thread(5,0,0) | TID(5) -> 2.22 + 1
SM(0) | Block(0,0,0) | Thread(6,0,0) | TID(6) -> 2.01 + 1
SM(0) | Block(0,0,0) | Thread(7,0,0) | TID(7) -> 7.18 + 1
SM(0) | Block(0,0,0) | Thread(8,0,0) | TID(8) -> -5.34 + 1
SM(0) | Block(0,0,0) | Thread(9,0,0) | TID(9) -> -2.97 + 1
SM(0) | Block(0,0,0) | Thread(10,0,0) | TID(10) -> -0.16 + 1
SM(0) | Block(0,0,0) | Thread(11,0,0) | TID(11) -> 1.12 + 1
SM(0) | Block(0,0,0) | Thread(12,0,0) | TID(12) -> 6.48 + 1
SM(0) | Block(0,0,0) | Thread(13,0,0) | TID(13) -> 9.48 + 1
SM(0) | Block(0,0,0) | Thread(14,0,0) | TID(14) -> -8.82 + 1
SM(0) | Block(0,0,0) | Thread(15,0,0) | TID(15) -> -1.27 + 1

addABC elapsed time : 0.912384 ms

SM map:
   0|     16     0     0     0     0     0     0     0
   8|      0     0     0     0     0     0     0     0
  16|      0     0     0     0     0     0     0     0
  24|      0     0     0     0     0     0     0     0
  32|      0     0     0     0     0     0     0     0
  40|      0     0     0     0     0     0     0     0
  48|      0     0     0     0     0     0     0     0
  56|      0     0     0     0     0     0     0     0
  64|      0     0     0     0
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

SM(2) | Block(1,0,0) | Thread(0,0,0) | TID(8) -> -5.34 + 1
SM(2) | Block(1,0,0) | Thread(1,0,0) | TID(9) -> -2.97 + 1
SM(2) | Block(1,0,0) | Thread(2,0,0) | TID(10) -> -0.16 + 1
SM(2) | Block(1,0,0) | Thread(3,0,0) | TID(11) -> 1.12 + 1
SM(2) | Block(1,0,0) | Thread(4,0,0) | TID(12) -> 6.48 + 1
SM(2) | Block(1,0,0) | Thread(5,0,0) | TID(13) -> 9.48 + 1
SM(2) | Block(1,0,0) | Thread(6,0,0) | TID(14) -> -8.82 + 1
SM(2) | Block(1,0,0) | Thread(7,0,0) | TID(15) -> -1.27 + 1
SM(0) | Block(0,0,0) | Thread(0,0,0) | TID(0) -> -9.25 + 1
SM(0) | Block(0,0,0) | Thread(1,0,0) | TID(1) -> 8.08 + 1
SM(0) | Block(0,0,0) | Thread(2,0,0) | TID(2) -> 1.30 + 1
SM(0) | Block(0,0,0) | Thread(3,0,0) | TID(3) -> -6.62 + 1
SM(0) | Block(0,0,0) | Thread(4,0,0) | TID(4) -> -7.49 + 1
SM(0) | Block(0,0,0) | Thread(5,0,0) | TID(5) -> 2.22 + 1
SM(0) | Block(0,0,0) | Thread(6,0,0) | TID(6) -> 2.01 + 1
SM(0) | Block(0,0,0) | Thread(7,0,0) | TID(7) -> 7.18 + 1

addABC elapsed time : 0.903168 ms

SM map:
   0|      8     0     8     0     0     0     0     0
   8|      0     0     0     0     0     0     0     0
  16|      0     0     0     0     0     0     0     0
  24|      0     0     0     0     0     0     0     0
  32|      0     0     0     0     0     0     0     0
  40|      0     0     0     0     0     0     0     0
  48|      0     0     0     0     0     0     0     0
  56|      0     0     0     0     0     0     0     0
  64|      0     0     0     0
Done

```

The above commands read a vector with 32 elements and performs the addition with block sizes of 16 and 8. In the first command, only one block with 16 threads is created and that is offloaded to SM(0). In the second command, two blocks each with 8 threads are created and they are offloaded on SM_0 and SM_2. Note that if you want to compile for a specific architecture you can add `-arch=sm_XX`. See this [page](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) to find the corresponding SM number for an architecture. As another run, the following outputs show the SM usage map for 16K elements with block sizes of 4, 128 and 1024.
```
$ ./create_inputs 16384
$ ./addition 4
...
addABC elapsed time : 0.01536 ms

SM map:
   0|    240   228   240   224   236   228   236   224
   8|    280   264   268   256   244   224   236   228
  16|    236   228   240   216   272   276   272   256
  24|    236   228   232   228   224   228   224   228
  32|    264   256   284   268   232   228   228   232
  40|    232   224   232   224   272   276   288   272
  48|    232   232   228   236   232   224   232   224
  56|    260   240   276   256   240   216   240   220
  64|    232   220   232   220
Done

$ ./addition 128
...
addABC elapsed time : 0.012288 ms

SM map:
   0|    256   256   256   256   256   256   256   256
   8|    256   256   256   256   256   256   256   256
  16|    256   256   256   256   256   256   256   256
  24|    256   256   256   256   256   256   256   256
  32|    256   256   256   256   256   256   256   256
  40|    256   256   256   256   256   256   256   256
  48|    256   256   256   256   256   128   256   128
  56|    256   128   256   128   256   128   256   128
  64|    256   128   256   128
Done


$ ./addition 1024
...

addABC elapsed time : 0.012288 ms

SM map:
   0|   1024     0  1024     0  1024     0  1024     0
   8|   1024     0  1024     0  1024     0  1024     0
  16|   1024     0  1024     0  1024     0  1024     0
  24|   1024     0  1024     0  1024     0  1024     0
  32|      0     0     0     0     0     0     0     0
  40|      0     0     0     0     0     0     0     0
  48|      0     0     0     0     0     0     0     0
  56|      0     0     0     0     0     0     0     0
  64|      0     0     0     0

```
