#pragma once
#ifndef CONFIG_H
#define CONFIG_H



#include <stdint.h>
#include <stdlib.h>



const uint32_t ARRAY_SIZE = (uint32_t)atoi(getenv("ARRAY_SIZE"));
const uint32_t CYCLES = (uint32_t)atoi(getenv("CYCLES"));
const uint32_t KBLOCKS = (uint32_t)atoi(getenv("KBLOCKS"));
const uint32_t KTHREADS = (uint32_t)atoi(getenv("KTHREADS"));

//#define SERIAL

#define arr_t float
#define arrO_t double

const arr_t MIN_RAND = -1;
const arr_t MAX_RAND = 1;

#endif // !CONFIG_H
