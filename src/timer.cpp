#include "timer.h"
#include "stdio.h"



void time_s::print(const char* t_tot
  #ifndef SERIAL
                   , const char* t_cpy,
                   const char* t_run,
                   const char* t_ret
  #endif
  ) {
  printf("%s%f", t_tot, this->total);
  #ifndef SERIAL
  printf("%s%f", t_cpy, this->memcpy);
  printf("%s%f", t_run, this->run);
  printf("%s%f", t_ret, this->memret);
  #endif
}

time_s& time_s::operator+=(const time_s& t) {
  this->total += t.total;
  #ifndef SERIAL
  this->memcpy += t.memcpy;
  this->run += t.run;
  this->memret += t.memret;
  #endif

  return *this;
}

time_s& time_s::operator-=(const time_s& t) {
  this->total -= t.total;
  #ifndef SERIAL
  this->memcpy -= t.memcpy;
  this->run -= t.run;
  this->memret -= t.memret;
  #endif

  return *this;
}

time_s& time_s::operator*=(const float& f) {
  this->total *= f;
  #ifndef SERIAL
  this->memcpy *= f;
  this->run *= f;
  this->memret *= f;
  #endif

  return *this;
}

time_s& time_s::operator/=(const float& f) {
  this->total /= f;
  #ifndef SERIAL
  this->memcpy /= f;
  this->run /= f;
  this->memret /= f;
  #endif

  return *this;
}

time_s time_s::operator*(const float& f) {
  return {this->total * f
    #ifndef SERIAL
    , this->memcpy * f,
    this->run * f,
    this->memret * f
    #endif
  };
}

time_s time_s::operator/(const float& f) {
  return {this->total / f
    #ifndef SERIAL
    , this->memcpy / f,
    this->run / f,
    this->memret / f
    #endif
  };
}
