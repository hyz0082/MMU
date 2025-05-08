#pragma once

#include "util.h"
#include "config.h"
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "hw_cmd.h"

// static inline my_float_t identity(my_float_t *arr, uint64_t index, uint64_t out_dim)
// {
//     // return arr[index];
//     return read_dram_value_cmd(&arr[index]);
//     // return 0;
// }

// static inline my_float_t relu(my_float_t *arr, uint64_t index, uint64_t out_dim)
// {
//     // return max((my_float_t)0, arr[index]);
//     return max((my_float_t)0, read_dram_value_cmd(&arr[index]));
//     // return 1;
// }

static inline my_float_t identity(my_float_t *arr, uint64_t index, uint64_t out_dim)
{
    // return arr[index];
    // return read_dram_value_cmd(&arr[index]);
    return 0;
}

static inline my_float_t relu(my_float_t *arr, uint64_t index, uint64_t out_dim)
{
    // return max((my_float_t)0, arr[index]);
    // return max((my_float_t)0, read_dram_value_cmd(&arr[index]));
    return 1;
}


static inline my_float_t bounded_relu(my_float_t *arr, uint64_t index, uint64_t out_dim)
{
    return max((my_float_t)0, min((my_float_t)1, arr[index]));
}


float my_exp_fp32(float x) {
    // Polynomial approximation (example, adjust coefficients for better accuracy)
    float result = 1.0f + x * (1.0f + x * (0.5f + x * (0.166666667f + x * 0.041666667f)));

    //Basic Range reduction.
    if(x > 10.0f){
        result = x;//std::exp(x); //if out of range, use standard lib.
    }
    else if (x < -10.0f){
        result = x;//std::exp(x);
    }

    return result;
}



// old softmax
//
// static inline my_float_t softmax(my_float_t *arr, uint64_t index, uint64_t out_dim)
// {
//     my_float_t max = 0;
//     for (uint64_t i = 0; i < out_dim; i++)
//         if (arr[i] > max)
//             max = arr[i];
//     // my_float_t numer = exp(arr[index] - max);
//     my_float_t numer = my_exp_fp32(arr[index] - max);
//     my_float_t denom = 0;
//     for (uint64_t i = 0; i < out_dim; i++)
//         // denom += exp(arr[i] - max);
//         denom += my_exp_fp32(arr[i] - max);
//     return numer / denom;
// }


float taylor_exp(my_float_t x, int terms) {
    float result = 1.0;
    float term = 1.0;
    float in_ = x;

    for (int i = 1; i < terms; i++) {
        term *= in_ / i;
        result += term;
    }
    return result;
}

// static inline my_float_t softmax(my_float_t *arr, uint64_t index, uint64_t out_dim)
// {

//     float numer = taylor_exp(arr[index], 20);//exp(arr[index]);
//     float denom = 0;
//     for (uint64_t i = 0; i < out_dim; i++)
//         denom += taylor_exp(arr[i], 20);//exp(arr[i]);
//     return (my_float_t)(numer / denom);
// }

// static inline my_float_t softmax(my_float_t *arr, uint64_t index, uint64_t out_dim)
// {
//     float_t max = 0;
//     for (uint64_t i = 0; i < out_dim; i++)
//         if (arr[i] > max)
//             max = arr[i];
//     // float_t numer = exp(arr[index] - max);
//     float_t numer = exp(arr[index] - max);
//     float_t denom = 0;
//     for (uint64_t i = 0; i < out_dim; i++)
//         // denom += exp(arr[i] - max);
//         denom += exp(arr[i] - max);
//     return numer / denom;
// }

static inline my_float_t softmax(my_float_t *arr, uint64_t index, uint64_t out_dim)
{
    my_float_t max = 0;
    for (uint64_t i = 0; i < out_dim; i++)
        if (arr[i] > max)
            max = arr[i];
    // float_t numer = exp(arr[index] - max);
    my_float_t numer = exp(arr[index] - max);
    my_float_t denom = 0;
    for (uint64_t i = 0; i < out_dim; i++) {
        my_float_t tmp = exp(arr[i] - max);
        denom += tmp;//exp(arr[i]);
    }
    // printf("numer: %f, denom: %f\n", (float_t)numer, (float_t)denom);
    return numer / denom;
}