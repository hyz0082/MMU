#pragma once

#include "util.h"
#include "config.h"
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

static inline float_t identity(float_t *arr, uint64_t index, uint64_t out_dim)
{
    return arr[index];
}

static inline float_t relu(float_t *arr, uint64_t index, uint64_t out_dim)
{
    return max((float_t)0, arr[index]);
}

static inline float_t bounded_relu(float_t *arr, uint64_t index, uint64_t out_dim)
{
    return max((float_t)0, min((float_t)1, arr[index]));
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
// static inline float_t softmax(float_t *arr, uint64_t index, uint64_t out_dim)
// {
//     float_t max = 0;
//     for (uint64_t i = 0; i < out_dim; i++)
//         if (arr[i] > max)
//             max = arr[i];
//     // float_t numer = exp(arr[index] - max);
//     float_t numer = my_exp_fp32(arr[index] - max);
//     float_t denom = 0;
//     for (uint64_t i = 0; i < out_dim; i++)
//         // denom += exp(arr[i] - max);
//         denom += my_exp_fp32(arr[i] - max);
//     return numer / denom;
// }


float_t taylor_exp(float_t x, int terms) {
    float_t result = 1.0;
    float_t term = 1.0;
    for (int i = 1; i < terms; i++) {
        term *= x / i;
        result += term;
    }
    return result;
}

static inline float_t softmax(float_t *arr, uint64_t index, uint64_t out_dim)
{
    // float_t max = 0;
    // for (uint64_t i = 0; i < out_dim; i++)
    //     if (arr[i] > max)
    //         max = arr[i];
    // printf("in sf\n");
    static int flag = 0;
    static float_t denom_s = 0;

    float_t numer = taylor_exp(arr[index], 35);//exp(arr[index]);
    if(flag) return numer / denom_s;
    float_t denom = 0;
    for (uint64_t i = 0; i < out_dim; i++)
        denom += taylor_exp(arr[i], 35);//exp(arr[i]);
    flag = 1;
    denom_s = denom;
    return numer / denom;
}