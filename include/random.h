#pragma once
#ifndef RANDOM_H
#define RANDOM_H

#include "config.h"

#include <stdint.h>



void Randomize(arr_t* arr, const uint32_t& size, const arr_t& min_v, const arr_t& max_v);

#endif // !RANDOM_H
