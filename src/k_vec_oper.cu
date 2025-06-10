#include <__clang_cuda_builtin_vars.h>
#ifndef SERIAL



#include "config.h"
#include "timer.h"
#include "vec_oper.h"
#include "k_vec_oper.h"

#include <stdint.h>
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime.h>



time_s Operation(arr_t* arr, arrO_t& out, const uint32_t& size) {
  time_s time;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  arr_t* d_arr;
  arrO_t* d_out;
  uint32_t* d_size;
  cudaMalloc((void**)&d_arr, size*sizeof(arr_t));
  cudaMalloc((void**)&d_out, sizeof(arrO_t));
  cudaMalloc((void**)&d_size, sizeof(uint32_t));

  cudaStream_t stream;
  cudaStreamCreate(&stream);


  CUDATIME(({
    cudaHostRegister(arr, size, cudaHostRegisterDefault);

    cudaMemcpyAsync(d_arr, arr, size*sizeof(arr_t), cudaMemcpyHostToDevice, stream);
    cudaMemset(d_out, 0, sizeof(arrO_t));
    cudaMemcpy(d_size, &size, sizeof(arr_t), cudaMemcpyHostToDevice);

    cudaHostUnregister(arr);
  }), time.memcpy, start, end);

  dim3 blocks(KBLOCKS, 1, 1);
  dim3 threads(KTHREADS, 1, 1);

  CUDATIME(({
    KSum<<<blocks, threads>>>(d_arr, size, *d_out);
  }), time.run, start, end);

  CUDATIME(({ 
    cudaMemcpy(&out, &d_out, sizeof(arrO_t), cudaMemcpyDeviceToHost);
  }), time.memret, start, end); 

  time.total = time.memcpy + time.run + time.memret;

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  cudaFree(d_arr); 
  cudaFree(d_out);
  cudaFree(d_size);

  return time;
}



__global__ void KSum(arr_t* arr, const uint32_t& size, arrO_t* out) {
  const uint32_t proc_size = size / gridDim.x;
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x * proc_size;

  for(uint32_t i=idx; i<idx+proc_size && i<size; ++i)
    out[threadIdx.x] += arr[i];
}

#endif
