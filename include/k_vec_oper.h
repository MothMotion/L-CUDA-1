#pragma once
#if !defined(K_VEC_OPER_H) && !defined(SERIAL)
#define K_VEC_OPER_H



#include "config.h"
#include "timer.h"
#include "vec_oper.h"

#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>


template<typename T, typename M>
__global__ void KSum(T* arr1, const uint32_t& size, M* out);

#endif
