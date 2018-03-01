#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA_CHECK: " << cudaGetErrorString(code) << " " << file << " "
        << line << std::endl;
    if (abort)
      exit(code);
  }
}

int main(void) {
    float *h = new float[1000];
    float *d = nullptr;


    CUDA_CHECK(cudaMalloc(&d, 1000 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h, 1000 * sizeof(float), cudaMemcpyHostToDevice));


    delete[] h;
    CUDA_CHECK(cudaFree(d));
    return 0;
}