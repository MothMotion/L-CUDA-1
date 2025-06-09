#pragma once
#ifndef VEC_OPER_H
#define VEC_OPER_H



#include "config.h"
#include "random.h"
#include "timer.h"

#include <stdint.h>



time_s Operation(arr_t* arr, arrO_t& out, const uint32_t& size);

#ifdef SERIAL

arrO_t Sum(arr_t* arr, const uint32_t& size);

#endif



#endif // !VEC_OPER_H
