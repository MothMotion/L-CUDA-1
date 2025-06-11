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
  arrO_t *d_out, *d_result;
  uint32_t *d_size, *d_blocks;
  cudaMalloc((void**)&d_arr, size*sizeof(arr_t));
  cudaMalloc((void**)&d_out, KBLOCKS*sizeof(arrO_t));
  cudaMalloc((void**)&d_result, sizeof(arrO_t));
  cudaMalloc((void**)&d_size, sizeof(uint32_t));
  cudaMalloc((void**)&d_blocks, sizeof(uint32_t));

  dim3 blocks(KBLOCKS, 1, 1);
  dim3 threads(KTHREADS, 1, 1);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaHostRegister(arr, size*sizeof(arr_t), cudaHostRegisterDefault);
  CUDATIME(({
    cudaMemcpyAsync(d_arr, arr, size*sizeof(arr_t), cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_out, 0, blocks.x*sizeof(arrO_t));
    cudaMemsetAsync(d_result, 0, sizeof(arrO_t));
    cudaMemcpyAsync(d_size, &size, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_blocks, &blocks.x, sizeof(uint32_t), cudaMemcpyHostToDevice); 
  }), time.memcpy, start, end);
  cudaHostUnregister(arr);

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  CUDATIME(({
    KSum<<<blocks, threads, threads.x * sizeof(arrO_t)>>>(d_arr, *d_size, d_out);
    KSum<<<1, threads, threads.x * sizeof(arrO_t)>>>((arrO_t*)d_out, *d_blocks, d_result);
  }), time.run, start, end);

  CUDATIME(({ 
    cudaMemcpy(&out, d_result, sizeof(arrO_t), cudaMemcpyDeviceToHost);
  }), time.memret, start, end); 

  time.total = time.memcpy + time.run + time.memret;

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  cudaFree(d_arr); 
  cudaFree(d_out);
  cudaFree(d_size);
  cudaFree(d_blocks);
  cudaFree(d_result);

  return time;
}



template<typename T, typename M>
__global__ void KSum(T* arr, const uint32_t& size, M* out) {
  extern __shared__ arrO_t sdata[];
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  arrO_t temp_out = 0;

  for(uint32_t i=idx; i<size; i += blockDim.x * gridDim.x)
    temp_out += arr[i];

  sdata[threadIdx.x] = temp_out;
  __syncthreads();

  for(uint32_t i = blockDim.x/2; i>0; i >>= 1) {
    if(threadIdx.x < i)
      sdata[threadIdx.x] += sdata[threadIdx.x + i];
    __syncthreads();
  }

  if(threadIdx.x == 0)
    out[blockIdx.x] += sdata[0];
}

#endif
