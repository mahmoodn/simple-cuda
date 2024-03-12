// nvcc -o matrixmul -arch=sm_86 -Xptxas -O3,-v matrixmul.cu
// cuobjdump vec_add -sass
// nvcc -o matrixmul -arch=sm_86 -g -G matrixmul.cu

#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <string>
#include "cuda_runtime_api.h"
#include "helper.h"




// This function reads an input file and copy the data to the host vector
void readInputFiles(vector<float> &v)
{
  ifstream f("inp1.txt");
  float a;
  while (f >> a)
    v.push_back(a);
}

// This helper function prints the final results on the screen
void print(vector<float> &v)
{
  for (auto x:v)
    cout << x << " ";
  cout << "\n";
}
void print(vector<int> &v)
{
  for (auto x:v)
    cout << x << " ";
  cout << "\n";
}


// This function allocates the array on device memory and copy inout data from host to device
// Note that std::vector::data() returns a pointer to the first location of vector, a.k.a hv[0]
void allocateOnDevice(float *&dv, vector<float> &hv, int n)
{
  CUDACHECK( cudaMalloc((void**)&dv, n * sizeof(float)) );
  CUDACHECK( cudaMemcpy(dv, hv.data(), n * sizeof(float), cudaMemcpyHostToDevice) );
}

// This function transfers the result from device to host
void transferResults(float *&dv, vector<float> &hv, int n)
{
  hv.resize(n);
  CUDACHECK( cudaMemcpy(hv.data(), dv, n * sizeof(float), cudaMemcpyDeviceToHost) );
  //print(hv);
}

// This function frees device memory
void freeMemory(float *&dv)
{
  CUDACHECK(cudaFree(dv));
}


// This function returns the SM number
__device__ uint get_smid(void) {
     uint ret;
     asm("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}


// Kernel inplementation
__global__ void simpleAdd(float *v, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    printf("SM(%d) Block(%d,%d,%d) Thread(%d,%d,%d) -> %.2f + 1\n", get_smid(), blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, v[i]);
    v[i] = v[i] + 1;
  }
}

    
int main()
{
  printInfo();
  vector<float> hostVector;
  readInputFiles(hostVector);
  cout << "Read " << hostVector.size() << " elements from inp1.txt\n\n";

  int n = hostVector.size();
  float *deviceVector;
  allocateOnDevice(deviceVector, hostVector, n);

  int numBlocks = (n + blockSize - 1) / blockSize;
  
  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start); cudaEventRecord(start,0);

  simpleAdd<<<numBlocks, blockSize>>>(deviceVector, n);

  cudaEventCreate(&stop);  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start,stop);

  cout << "\naddABC elapsed time : " << elapsedTime << " ms\n";
  cudaDeviceSynchronize();

  transferResults(deviceVector, hostVector, n);
  freeMemory(deviceVector);
  
  cout << "\nResult:\n";
  print(hostVector);

  cout << "Done\n";
  return 0;
}
