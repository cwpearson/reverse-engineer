#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cassert>

#include <cuda_runtime.h>
#include <cuda.h>

#include <dlfcn.h>

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

#define RT_CHECK(ans)                                                   \
  { rtAssert((ans), __FILE__, __LINE__); }
inline void rtAssert(CUresult code, const char *file, int line, bool abort = true) {
  if (code != CUDA_SUCCESS) {
    const char *str;
    cuGetErrorString(code, &str);
    std::cerr << "RT_CHECK: " << str << " " << file << " "
        << line << std::endl;
    if (abort)
      exit(code);
  }
}

const size_t N = 8 * 1024 * 1024;

void call1(float *h, float *d) {
  RT_CHECK(cuMemcpyHtoD((uintptr_t)d, h, N * sizeof(float)));
  RT_CHECK(cuMemcpyHtoD((uintptr_t)d, h, N * sizeof(float)));
}

void call1p(float *h, float *d) {
  RT_CHECK(cuMemcpyHtoD((uintptr_t)d, h, N * sizeof(float)));
  RT_CHECK(cuMemcpyHtoD((uintptr_t)d, h, N * sizeof(float)));
}


void call2(float *d, float *h) {
  RT_CHECK(cuMemcpyDtoH(h, (uintptr_t)d, N * sizeof(float)));
  RT_CHECK(cuMemcpyDtoH(h, (uintptr_t)d, N * sizeof(float)));
}

void call2p(float *d, float *h) {
  RT_CHECK(cuMemcpyDtoH(h, (uintptr_t)d, N * sizeof(float)));
  RT_CHECK(cuMemcpyDtoH(h, (uintptr_t)d, N * sizeof(float)));
}

int main(void) {
    float *h = new float[N];
    float *hp;
    float *d = nullptr; 

    CUDA_CHECK(cudaMalloc(&d, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&hp, N * sizeof(float)));

    void *handle = dlopen("libcuda.so", RTLD_NOW);
    void *cudaMallocAddr = dlsym(handle, "cuMemAlloc_v2");
    Dl_info info;
    assert(dladdr(cudaMallocAddr, &info));
    fprintf(stderr, "%s @ %p\n", info.dli_fname, info.dli_fbase);

    call1(h, d);
    call1p(hp, d);
    call2(d, h);
    call2p(d, hp);

    delete[] h;
    CUDA_CHECK(cudaFree(d));
    return 0;
}
