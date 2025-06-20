#include <math.h>
#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "config.h"
#include "layer.h"
#include "list.h"
#include "util.h"
#include "activation_function.h"
#include "hw_cmd.h"
#include <time.h>
#include <float.h>
#ifndef USING_GEM5
#include "loader.h"
#endif

typedef struct _batchnorm_layer
{
    layer_base base;
    index3d in_;
} batchnorm_layer;

batchnorm_layer * get_batchnorm_layer_entry(struct list_node *ptr)
{
    return list_entry(ptr, batchnorm_layer, base.list);
}

my_float_t _beta(batchnorm_layer *entry, int ind) {
    return entry->base._W[(entry->in_.depth_ * 1) + ind];
}

my_float_t _gamma(batchnorm_layer *entry, int ind) {
    return entry->base._W[(entry->in_.depth_ * 0) + ind];
}

my_float_t _mean(batchnorm_layer *entry, int ind) {
    return entry->base._W[(entry->in_.depth_ * 2) + ind];
}



float root(float n){
    // Max and min are used to take into account numbers less than 1
    float lo = min(1, n), hi = max(1, n), mid;
  
    // Update the bounds to be off the target by a factor of 10
    while(100 * lo * lo < n) lo *= 10;
    while(0.01 * hi * hi > n) hi *= 0.1;
  
    for(int i = 0 ; i < 100 ; i++){
      mid = (lo+hi)/2;
      if(mid*mid == n) return mid;
      if(mid*mid > n) hi = mid;
      else lo = mid;
    }
    return mid;
}

my_float_t _invstd(batchnorm_layer *entry, int ind) {
    // return 1/root(entry->base._W[(entry->in_.depth_ * 3) + ind] + 0.00001);
    return 1/root(entry->base._W[(entry->in_.depth_ * 3) + ind] + 0.00001);
    // return 1/(my_float_t)sqrt(entry->base._W[(entry->in_.depth_ * 3) + ind] + 0.00001);
}




void batchnorm_layer_forward_propagation(struct list_node *ptr, unsigned int hart_id, input_struct *input)
{
#ifdef USING_GEM5
    clock_t  tick, ticks_per_msec = CLOCKS_PER_SEC/1000;
    clock_t hardware_compute_time = 0;
    tick = clock();
    clock_t ticks_estimate = 2;
    clock_t start_clock = clock();
    clock_t send_data_time = 0;
    clock_t hw_compute_time = 0;
    clock_t store_data_time = 0;
#endif

    batchnorm_layer *entry = get_batchnorm_layer_entry(ptr);
    static int bn_cnt = 0;
    if (input->in_size_ != entry->base.in_size_)
    {
        if (hart_id == 0) 
            printf("Error input size not match %d/%d\n", input->in_size_, entry->base.in_size_);
        exit(-1);
    }

    entry->base.a_ptr_ = input->in_ptr_;
    entry->base.out_ptr_ = entry->base.a_ptr_;

    my_float_t *in = input->in_ptr_;
    my_float_t *a = entry->base.a_ptr_;
    my_float_t *W = entry->base._W;
    my_float_t *b = entry->base._b;
    my_float_t *out = entry->base.out_ptr_;

    input->in_ptr_ = out;
    input->in_size_ = entry->base.out_size_;

    index3d in_ = entry->in_;
    int dim = in_.height_*in_.width_;

    int total_size = dim;
    int blocksize = compute_block_size(total_size);
    int start = (blocksize) * hart_id;
    int end = min((blocksize) * (hart_id+1), total_size);

    reset_cmd();

    //###########################################################
    // print output
    // if(bn_cnt == 0)
    // {
    //     printf("output:\n");
    //     printf("\n\n[%s]\n", entry->base.layer_name_);
    //     printf("shape depth:%d,  height:%d width:%d\n", (int)in_.depth_, (int)in_.height_, (int)in_.width_);
    //     printf("    ");
    //     for(int i = 0; i < in_.width_; i++) {
    //         printf("%6d ", (i));
    //     }
    //     printf("\n");
    //     // const my_float_t *pi = &out[0];
    //     int p_cnt = 0;
    //     for (int inc = 0; inc < in_.depth_; inc++) {
    //         printf("\n[depth%d]\n", inc);
    //         // const my_float_t *pi = &in[get_index(&in_padded_, 0, 0, inc)];
    //         for (int h = 0; h < in_.height_; h++) {
    //             printf("%3d ", (h));
    //             printf("[ ");
    //             for (int w = 0; w < in_.width_; w++) {
    //                 // printf("%2.6f ", ((float)out[p_cnt]));
    //                 printf("%2.6f ", ((float_t)read_dram_value_cmd(&out[p_cnt])));
                    
    //                 p_cnt++;
    //             }
    //             printf("]\n");
    //         }
    //     }
    // }
    //###########################################################
    bn_cnt++;
}


layer_base * new_batchnorm_layer(
                                cnn_controller *ctrl,
                                my_float_t(*activate) (my_float_t *, int, int),
                                int channels,
                                int in_width,
                                int in_height
                                )
{

// #ifndef USING_GEM5
    batchnorm_layer *ret = (batchnorm_layer *)malloc(sizeof(batchnorm_layer));
// #else
//     batchnorm_layer *ret = (batchnorm_layer *) ctrl->nwk_cur_ptr;
//     ctrl->nwk_cur_ptr += sizeof(batchnorm_layer);
// #endif

    init_layer(&ret->base,
               ctrl,
               in_width*in_height*channels,
               in_width*in_height*channels,
               4*channels,
               0,
               activate==softmax);
#ifdef PRINT_LAYER
    static int call_time = 0;
    // sprintf(ret->base.layer_name_, "norm%d", call_time++);
    my_sprintf(ret->base.layer_name_, "norm%d", call_time++);
#endif
    ret->base.activate = activate;
    ret->in_ = new_index3d(in_width, in_height, channels);
    // printf("insize of BN layer %d\n", ret->base.in_size_);
    printf("BN: W  [%f, %f, ... , %f, %f]\n", ret->base._W[0], ret->base._W[1], ret->base._W[4*channels-2], ret->base._W[4*channels-1]);
    // printf("BN: in [%f, %f, ... , %f, %f]\n", ret->base.in_ptr_[0], ret->base.in_ptr_[1], ret->base.in_ptr_[ret->base.in_size_-2], ret->base.in_ptr_[ret->base.in_size_-1]);
    // printf("BN: b  [%f, %f, ... , %f, %f]\n", ret->base._b[0], ret->base._b[1], ret->base._b[out_dim-2], ret->base._b[out_dim-1]);
    ret->base.forward_propagation = batchnorm_layer_forward_propagation;
    
    // pre calc BatchNorm value
    for(int ch = 0; ch < channels; ch++) {
        my_float_t mul = _gamma(ret, ch) * _invstd(ret, ch);
        my_float_t add = _gamma(ret, ch) * (- _mean(ret, ch)) * _invstd(ret, ch) + _beta(ret, ch);
        ret->base._W[(ret->in_.depth_ * 0) + ch] = mul;
        ret->base._W[(ret->in_.depth_ * 1) + ch] = add;

        // write_dram_value_cmd(&ret->base._W[(ret->in_.depth_ * 0) + ch], mul);
        // write_dram_value_cmd(&ret->base._W[(ret->in_.depth_ * 1) + ch], add);
    }

    for(int ch = 0; ch < channels; ch++) {
        my_float_t mul = _gamma(ret, ch) * _invstd(ret, ch);
        my_float_t add = _gamma(ret, ch) * (- _mean(ret, ch)) * _invstd(ret, ch) + _beta(ret, ch);
        if(mul * 7 + add == 100) {
            printf(" ");
        }
    } 

    return &ret->base;
}