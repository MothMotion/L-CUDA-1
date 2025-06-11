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
  // {size, options}
  uint32_t *d_size, *d_counter;
  cudaMalloc((void**)&d_arr, size*sizeof(arr_t));
  cudaMalloc((void**)&d_out, sizeof(arrO_t));
  cudaMalloc((void**)&d_size, sizeof(uint32_t));
  cudaMalloc((void**)&d_counter, sizeof(uint32_t));

  dim3 blocks(KBLOCKS, 1, 1);
  dim3 threads(KTHREADS, 1, 1);

  cudaStream_t stream;
  cudaStreamCreate(&stream);


  CUDATIME(({
    cudaHostRegister(arr, size*sizeof(arr_t), cudaHostRegisterDefault);

    cudaMemcpyAsync(d_arr, arr, size*sizeof(arr_t), cudaMemcpyHostToDevice, stream);
    cudaMemset(d_out, 0, blocks.x*sizeof(arrO_t));
    cudaMemcpy(d_size, &size, sizeof(uint32_t), cudaMemcpyHostToDevice); 

    cudaHostUnregister(arr);
  }), time.memcpy, start, end); 

  CUDATIME(({
    while(blocks.x > threads.x * threads.x) {
      if(blocks.x == KBLOCKS)
        KSum<<<blocks, threads>>>(d_arr, *d_size, d_out);
      else
        KSum<<<blocks, threads>>>((arrO_t*)d_arr, *d_size, d_out);

      cudaMemcpy(d_arr, d_out, blocks.x * sizeof(arrO_t), cudaMemcpyDeviceToDevice);
      blocks.x = (blocks.x + threads.x - 1)/threads.x;
      cudaMemset(d_out, 0, blocks.x * sizeof(arrO_t)); 
    }
    KSum<<<1, threads>>>((arrO_t*)d_arr, *d_size, d_out);
  }), time.run, start, end);

  CUDATIME(({ 
    cudaMemcpy(&out, d_out, sizeof(arrO_t), cudaMemcpyDeviceToHost);
  }), time.memret, start, end); 

  time.total = time.memcpy + time.run + time.memret;

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  cudaFree(d_arr); 
  cudaFree(d_out);
  cudaFree(d_size);
  cudaFree(d_counter);

  return time;
}



template<typename T, typename M>
__global__ void KSum(T* arr, const uint32_t& size, M* out) {
  const uint32_t proc_size = size / gridDim.x;
  const uint32_t thread_size = proc_size / blockDim.x;
  uint32_t idx = blockIdx.x * proc_size + threadIdx.x * thread_size;

  arrO_t temp_out = 0;

  for(uint32_t i=idx; i<idx+thread_size && i<size; ++i)
    temp_out += arr[i];

  __shared__ arrO_t sdata[MAX_THREADS];
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
