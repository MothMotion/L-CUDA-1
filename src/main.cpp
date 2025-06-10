#include "config.h"
#include "random.h"
#include "timer.h"
#include "vec_oper.h"

#ifndef SERIAL
#include "k_vec_oper.h"
#endif

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>



int main() {
  srand(10000);
  time_s time_temp;
  uint32_t arr_size = ARRAY_SIZE,
           cycles   = CYCLES;

  arr_t *arr = (arr_t*)malloc(arr_size * sizeof(arr_t));
  arrO_t out = 0;

  float avg_rand = 0;

  time_s avg_time; 

  #ifdef SERIAL
  printf("Serial ");
  #else
  printf("Parallel ");
  #endif
  printf("execution.\nSettings:\n\tArray size:\t%d\n\tCycles:\t\t%d\n\tBlocks:\t\t%d\n\tThreads:\t%d\n\tElement size:\t%d\n\n",
         arr_size, cycles, ( (KBLOCKS) ? (KBLOCKS) : ((arr_size%KTHREADS) ? (arr_size/KTHREADS + 1) : (arr_size/KTHREADS)) ), KTHREADS, (uint32_t)sizeof(arr_t));



  for(uint32_t i=0; i<cycles; ++i) {
    GETTIME(({
      Randomize(arr, arr_size, MIN_RAND, MAX_RAND);
    }), time_temp.total);
    avg_rand += time_temp.total/cycles * 1000;

    
    avg_time += Operation(arr, out, arr_size) / cycles;
    printf("Cycle:\t%du\tOut:\t%f", i, out);
  }



  printf("Average time spent per cycle.\nRandomizing:\t%fms\n", avg_rand); 

  #ifdef SERIAL
  avg_time.print("Total:\t");

  #else
  avg_time.print("Total:\t", "ms\n\tCopying:\t", "ms\n\tRunning:\t", "ms\n\tReturning:\t");

  #endif
  printf("ms\n");

  return 0;
}
