#include <iostream>
#include "cuda_runtime_api.h"

using namespace std;


#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  cout << "MapSMtoCores for SM " << major << "." << minor 
       << " is undefined. Default to use " 
       <<  nGpuArchCoresPerSM[index - 1].Cores << " Cores/SM\n";

  return nGpuArchCoresPerSM[index - 1].Cores;
}

// This function prints basic information about the device
void printInfo(int &smCount, int blockSize)
{
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  cout << string(50, '-') << endl;
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    cout << "Device Number: " << i << endl;
    cout << "  Device name: " << prop.name << endl;
    cout << "  SM count: " << prop.multiProcessorCount << endl;
    cout << "  Cores per SM: " << _ConvertSMVer2Cores(prop.major, prop.minor) << endl;
    cout << "  Max Blocks per SM: " << prop.maxBlocksPerMultiProcessor << endl;
    cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
    cout << "  Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << endl;
    assert(blockSize <= prop.maxThreadsPerBlock);
    smCount = prop.multiProcessorCount;
  }
  cout << string(50, '-') << endl;
}


// This function returns the SM number
__device__ uint get_smid(void) {
     uint ret;
     asm("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}

// This array is defined on the device memory showing SM indices.
__device__ int g_smid[68] = {
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0
};


// This function prints the SM map
void printSM(int smCount)
{
  int h_smid[smCount];
  CUDACHECK( cudaMemcpyFromSymbol(h_smid, g_smid, sizeof(h_smid)) );
  cout << "\nSM map:";
  for (int i = 0; i < smCount; i++) {
    if (i % 8 == 0) {
      cout << "\n" << setw(4) << i << "| ";
    }
    cout << setw(6) << h_smid[i];
  }
  cout << endl;
}



// This helper function prints the final results on the screen
void print(vector<float> &v)
{
  for (auto x:v)
    cout << x << " ";
  cout << "\n";
}
