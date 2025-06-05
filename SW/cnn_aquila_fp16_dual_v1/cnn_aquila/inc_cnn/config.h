#pragma once

// typedef float my_float_t;
#ifndef USING_GEM5
typedef std::float16_t my_float_t;
typedef float float_t;
// typedef float my_float_t;
#else
typedef _Float16 my_float_t;
typedef float float_t;
#endif
// typedef float _Float16;
// typedef _Float16 my_float_t;