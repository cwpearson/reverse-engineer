#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <sstream>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda.h>
#include <nvToolsExt.h>

#include <dlfcn.h>
#include <unistd.h>



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

#define DR_CHECK(ans)                                                   \
  { rtAssert((ans), __FILE__, __LINE__); }
inline void rtAssert(CUresult code, const char *file, int line, bool abort = true) {
  if (code != CUDA_SUCCESS) {
    const char *str;
    cuGetErrorString(code, &str);
    std::cerr << "DR_CHECK: " << str << " " << file << " "
        << line << std::endl;
    if (abort)
      exit(code);
  }
}

#define DH_BODY DR_CHECK(cuMemcpyDtoH(h, (uintptr_t)d, n * sizeof(float)))
#define HD_BODY DR_CHECK(cuMemcpyHtoD((uintptr_t)d, h, n * sizeof(float)))

void dh1(float *h, const float*d, const size_t n) {
  DH_BODY;
}
void dh2(float *h, const float*d, const size_t n) {
  DH_BODY;
}

void hd1(float *d, const float*h, const size_t n) {
  HD_BODY;
}
void hd2(float *d, const float*h, const size_t n) {
  HD_BODY;
}

#undef DH_BODY
#undef HD_BODY

const size_t N = 8 * 1024 * 1024;

void touch(float *f, const size_t e, const size_t n) {
  const size_t stride = std::max(1ul, e / sizeof(float));
  for (int i = 0; i < n; i += stride) {
    f[i] = rand();
  }
}


int main(void) {

  const long pageSize = sysconf(_SC_PAGESIZE);

  int numDevices;
  CUDA_CHECK(cudaGetDeviceCount(&numDevices));
  fprintf(stderr, "%d devices\n", numDevices);

  // set up host allocations
  float *hpg = new float[N];
  assert((N == 0) || (hpg != nullptr));
  float *hpn;
  CUDA_CHECK(cudaMallocHost(&hpn, N * sizeof(float)));

  // setup device allocations
  float **d = new float*[numDevices];
  for (int i = 0; i < numDevices; ++i) {
    d[i] = nullptr;
    CUDA_CHECK(cudaMalloc(&d[i], N * sizeof(float)));
  }

  void *handle = dlopen("libcuda.so", RTLD_NOW);
  void *cudaMallocAddr = dlsym(handle, "cuMemAlloc_v2");
  Dl_info info;
  assert(dladdr(cudaMallocAddr, &info));
  fprintf(stderr, "%s @ %p\n", info.dli_fname, info.dli_fbase);

  // Touch host memory
  touch(hpn, pageSize, N);
  touch(hpg, pageSize, N);

  // memcpy sequence
  for (int i = 0; i < numDevices; ++i) {
    std::stringstream buffer;
    buffer << "pn cpu->" << i;
    nvtxRangePush(buffer.str().c_str());
    touch(hpn, pageSize, N);
    DR_CHECK(cuMemcpyHtoD((uintptr_t)d[i], hpn, N * sizeof(float)));
    touch(hpn, pageSize, N);
    DR_CHECK(cuMemcpyDtoH(hpn, (uintptr_t)d[i], N * sizeof(float)));
    touch(hpn, pageSize, N);
    DR_CHECK(cuMemcpyHtoD((uintptr_t)d[i], hpn, N * sizeof(float)));
    touch(hpn, pageSize, N);
    DR_CHECK(cuMemcpyDtoH(hpn, (uintptr_t)d[i], N * sizeof(float)));
    
    touch(hpn, pageSize, N);
    DR_CHECK(cuMemcpyHtoD((uintptr_t)d[i], hpn, N * sizeof(float)));
    touch(hpn, pageSize, N);
    DR_CHECK(cuMemcpyHtoD((uintptr_t)d[i], hpn, N * sizeof(float)));
    touch(hpn, pageSize, N);
    DR_CHECK(cuMemcpyDtoH(hpn, (uintptr_t)d[i], N * sizeof(float)));
    touch(hpn, pageSize, N);
    DR_CHECK(cuMemcpyDtoH(hpn, (uintptr_t)d[i], N * sizeof(float)));
    touch(hpn, pageSize, N);
    DR_CHECK(cuMemcpyHtoD((uintptr_t)d[i], hpn, N * sizeof(float)));
    touch(hpn, pageSize, N);
    DR_CHECK(cuMemcpyHtoD((uintptr_t)d[i], hpn, N * sizeof(float)));
    touch(hpn, pageSize, N);
    DR_CHECK(cuMemcpyDtoH(hpn, (uintptr_t)d[i], N * sizeof(float)));
    touch(hpn, pageSize, N);
    DR_CHECK(cuMemcpyDtoH(hpn, (uintptr_t)d[i], N * sizeof(float)));
    nvtxRangePop();
  }

  for (int i = 0; i < numDevices; ++i) {
    std::stringstream buffer;
    buffer << "pg cpu->" << i;
    nvtxRangePush(buffer.str().c_str());
    // DR_CHECK(cuMemcpyHtoD((uintptr_t)d[i], h, N * sizeof(float)));
    // DR_CHECK(cuMemcpyDtoH(h, (uintptr_t)d[i], N * sizeof(float)));
    // DR_CHECK(cuMemcpyHtoD((uintptr_t)d[i], h, N * sizeof(float)));
    // DR_CHECK(cuMemcpyDtoH(h, (uintptr_t)d[i], N * sizeof(float)));
    
    touch(hpg, pageSize, N);
    DR_CHECK(cuMemcpyHtoD((uintptr_t)d[i], hpg, N * sizeof(float)));
    touch(hpg, pageSize, N);
    DR_CHECK(cuMemcpyDtoH(hpg, (uintptr_t)d[i], N * sizeof(float)));
    touch(hpg, pageSize, N);
    DR_CHECK(cuMemcpyHtoD((uintptr_t)d[i], hpg, N * sizeof(float)));
    touch(hpg, pageSize, N);
    DR_CHECK(cuMemcpyDtoH(hpg, (uintptr_t)d[i], N * sizeof(float)));
    
    touch(hpg, pageSize, N);
    DR_CHECK(cuMemcpyHtoD((uintptr_t)d[i], hpg, N * sizeof(float)));
    touch(hpg, pageSize, N);
    DR_CHECK(cuMemcpyHtoD((uintptr_t)d[i], hpg, N * sizeof(float)));
    touch(hpg, pageSize, N);
    DR_CHECK(cuMemcpyDtoH(hpg, (uintptr_t)d[i], N * sizeof(float)));
    touch(hpg, pageSize, N);
    DR_CHECK(cuMemcpyDtoH(hpg, (uintptr_t)d[i], N * sizeof(float)));
    touch(hpg, pageSize, N);
    hd1(d[i], hpg, N);
    touch(hpg, pageSize, N);
    hd2(d[i], hpg, N);
    touch(hpg, pageSize, N);
    DR_CHECK(cuMemcpyDtoH(hpg, (uintptr_t)d[i], N * sizeof(float)));
    touch(hpg, pageSize, N);
    DR_CHECK(cuMemcpyDtoH(hpg, (uintptr_t)d[i], N * sizeof(float)));

    nvtxRangePop();
  }

    // delete[] h;
    // CUDA_CHECK(cudaFree(d));
    // CUDA_CHECK(cudaFreeHost(hp));
    return 0;
}
