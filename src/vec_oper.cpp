#ifdef SERIAL


#include "config.h"
#include "timer.h"
#include "vec_oper.h"

#include <stdint.h>
#include <stddef.h>



time_s Operation(arr_t* arr, arrO_t& out, const uint32_t& size) {
  time_s result;
  GETTIME(({
    Sum(arr, size);
  }), result.total); 
  return result * 1000;
}



arrO_t Sum(arr_t* arr, const uint32_t& size) {
  arrO_t result = 0;

  for(uint32_t i=0; i<size; ++i)
    result += arr[i];

  return result;
}

#endif
