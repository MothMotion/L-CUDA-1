#pragma once
#ifndef TIMER_H
#define TIMER_H



#include "config.h"
#include <time.h>



#define GETTIME(code_block, time_var) {\
  struct timespec start, end;\
  clock_gettime(CLOCK_REALTIME, &start);\
  code_block;\
  clock_gettime(CLOCK_REALTIME, &end);\
  time_var = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/1e9;\
}

#define CUDATIME(code_block, time_var, start, end)\
  cudaEventRecord(start);\
  code_block;\
  cudaEventRecord(end);\
  cudaEventSynchronize(end);\
  cudaEventElapsedTime(&time_var, start, end);

struct time_s {
  float total   = 0;

  #ifndef SERIAL
  float memcpy  = 0;
  float run     = 0;
  float memret  = 0;
  #endif

  void print(const char* t_tot
    #ifndef SERIAL
             , const char* t_cpy,
             const char* t_run,
             const char* t_ret
    #endif
  );

  time_s& operator+=(const time_s& t);
  time_s& operator-=(const time_s& t);
  time_s& operator*=(const float& f);
  time_s& operator/=(const float& f);
  time_s operator*(const float& f);
  time_s operator/(const float& f);
};


#endif // !TIMER_H
