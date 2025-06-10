#pragma once
#if !defined(K_VEC_OPER_H) && !defined(SERIAL)
#define K_VEC_OPER_H



#include "config.h"
#include "timer.h"
#include "vec_oper.h"

#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>



__global__ void KSum(arr_t* arr1, const uint32_t& size, arrO_t* out);
__global__ void KGetSum(arrO_t* arr, const uint32_t& size);

#endif
