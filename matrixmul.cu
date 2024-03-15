// nvcc -o matrixmul -arch=sm_86 -Xptxas -O3,-v matrixmul.cu
// cuobjdump vec_add -sass
// nvcc -o matrixmul -arch=sm_86 -g -G matrixmul.cu

#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <string>
#include <iomanip>
#include "cuda_runtime_api.h"
#include "helper.h"

void readInputFiles(vector<float> &v1, vector<float> &v2)
{
  ifstream f1("inp1.txt");
  ifstream f2("inp2.txt");
  float a;
  while (f1 >> a)
    v1.push_back(a);
  while (f2 >> a)
    v2.push_back(a);
}

void allocateOnDevice(float *&dv1, 
                      float *&dv2, 
                      float *&dv3, 
                      vector<float> &hv1, 
                      vector<float> &hv2, 
                      int n)
{
  CUDACHECK(cudaMalloc( (void**)&dv1, 
              n * sizeof(float) ));
  CUDACHECK(cudaMalloc( (void**)&dv2, 
              n * sizeof(float) ));
  CUDACHECK(cudaMalloc( (void**)&dv3, 
              n * sizeof(float) ));
           
  CUDACHECK(cudaMemcpy( dv1, hv1.data(), 
              n * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy( dv2, hv2.data(), 
              n * sizeof(float), cudaMemcpyHostToDevice));
}
void transferResults(float *&dv, 
                     vector<float> &hv, 
                     int n)
{
  hv.resize(n);
  CUDACHECK(cudaMemcpy( hv.data(), dv,
              n * sizeof(float), cudaMemcpyDeviceToHost));
  //print(hv);
}

void freeMemory(float *&dv1, 
                float *&dv2, 
                float *&dv3)
{
  CUDACHECK(cudaFree(dv1));
  CUDACHECK(cudaFree(dv2));
  CUDACHECK(cudaFree(dv3));   
}



__global__ void mulABC(float *v1, float *v2, float *v3, int width)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < width && col < width) {
    float sum = 0.0;
    int s = get_smid();
    atomicAdd(&g_smid[s], 1); //Atomic add is needed
    for (int k = 0; k < width; k++) {
      printf("SM(%d) | Block(%d,%d,%d) | Thread(%d,%d,%d) -> row=%d col=%d k=%d -> %.2f * %.2f + %.2f\n", 
             s, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, 
             row, col, k, v1[row*width+k], v2[k*width+col], sum);
      sum += v1[row*width+k] * v2[k*width+col];
    }
    v3[row*width+col] = sum;
  }
}
    
int main(int argc, char** argv)
{
  if (argc != 3) {
    cout << "Usage: ./addition <matrix_width> <block_width>\n";
    return 1;
  }
  int width = stoi(argv[1]);
  int blockWidth = stoi(argv[2]);
  int smCount;
  printInfo(smCount, 1);  // Dummy 1 to pass the assention
  vector<float> hostVector1, hostVector2, hostVector3;
  
  readInputFiles(hostVector1, hostVector2);
  cout << "Read " << hostVector1.size() << " elements from inp1.txt and " << hostVector2.size() << " from inp2.txt\n";
  assert(hostVector1.size() == hostVector2.size()); // Be sure that arrays are the same size
  
  int n = hostVector1.size();
  float *deviceVector1, *deviceVector2, *deviceVector3;
  allocateOnDevice(deviceVector1, deviceVector2, deviceVector3, hostVector1, hostVector2, n);
  
  cudaEvent_t start, stop;
  float elapsedTime;

  /* Given width=4 and blockWidth=4
   * numBlocks = (4+4-1)/4 = 1
   * blocksPerGrid(1, 1, 1)
   * threadsPerBlock(4, 4, 1)
   * This means one block, 4 threads on X and 4 threads on Y
   *
   * Given width=4 and blockWidth=2
   * numBlocks = (4+2-1)/2 = 2
   * blocksPerGrid(2, 2, 1)
   * threadsPerBlock(2, 2, 1)
   * This means four blocks, 2 threads on X and 2 threads on Y in each block
   */
  int numBlocks = (width + blockWidth - 1) / blockWidth;  // Integer representation for ceil()
  dim3 blocksPerGrid(numBlocks, numBlocks, 1);           
  dim3 threadsPerBlock(blockWidth, blockWidth, 1);
  
  cudaEventCreate(&start); cudaEventRecord(start,0);
  mulABC<<<blocksPerGrid, threadsPerBlock>>>( deviceVector1, deviceVector2, deviceVector3, width );
  cudaEventCreate(&stop);  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start,stop);
  cout << "mulABC elapsed time : " << elapsedTime << " ms\n";
  cudaDeviceSynchronize();
  
  transferResults(deviceVector3, hostVector3, n);
  printSM(smCount);
  
  freeMemory(deviceVector1, deviceVector2, deviceVector3);

  /*cout << "\nResult:\n";
  print(hostVector3);*/
  
  cout << "Done\n";
  return 0;
}
