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

float_t _beta(batchnorm_layer *entry, uint64_t ind) {
    return entry->base._W[(entry->in_.depth_ * 1) + ind];
}

float_t _gamma(batchnorm_layer *entry, uint64_t ind) {
    return entry->base._W[(entry->in_.depth_ * 0) + ind];
}

float_t _mean(batchnorm_layer *entry, uint64_t ind) {
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

float_t _invstd(batchnorm_layer *entry, uint64_t ind) {
    // return 1/root(entry->base._W[(entry->in_.depth_ * 3) + ind] + 0.00001);
    return 1/root(entry->base._W[(entry->in_.depth_ * 3) + ind] + 0.00001);
    // return 1/(float_t)sqrt(entry->base._W[(entry->in_.depth_ * 3) + ind] + 0.00001);
}




void batchnorm_layer_forward_propagation(struct list_node *ptr, unsigned int hart_id, input_struct *input)
{
    batchnorm_layer *entry = get_batchnorm_layer_entry(ptr);
    if (input->in_size_ != entry->base.in_size_)
    {
        if (hart_id == 0) 
            printf("Error input size not match %d/%d\n", input->in_size_, entry->base.in_size_);
        exit(-1);
    }

    // entry->base.padded_ptr = (float_t *)malloc(entry->base.padding_size * sizeof(float_t));
    // if (entry->base.padded_ptr != NULL) { // Check if memory was allocated
    //     memset((void*)entry->base.padded_ptr, 0, entry->base.padding_size * sizeof(float_t));
    // }
    // else {
    //     printf("Error: Unable to allocate memory for layer->padded_ptr\n");
    //     // exit(1);
    // }

    // malloc a_ptr
    entry->base.a_ptr_ = (float_t *)malloc(entry->base.out_size_ * sizeof(float_t));
    if (entry->base.a_ptr_ != NULL) { // Check if memory was allocated
        memset((void*)entry->base.a_ptr_, 0, entry->base.out_size_ * sizeof(float_t));
    }
    else {
        printf("Error: Unable to allocate memory for entry->base.a_ptr_\n");
        // exit(1);
    }

    // if (entry->base.need_space_for_a){
    //     entry->base.out_ptr_ = (float_t *)malloc(entry->base.out_size_ * sizeof(float_t));
    //     if (entry->base.out_ptr_ != NULL) { // Check if memory was allocated
    //         memset((void*)entry->base.out_ptr_, 0, entry->base.out_size_ * sizeof(float_t));
    //     }
    //     else {
    //         printf("Error: Unable to allocate memory for entry->base.out_ptr_\n");
    //         // exit(1);
    //     }
    // }
    // else {
        entry->base.out_ptr_ = entry->base.a_ptr_;
    // }

    float_t *in = input->in_ptr_;

    float_t *a = entry->base.a_ptr_;
    // float_t *a = input->in_ptr_;
    
    float_t *W = entry->base._W;
    float_t *b = entry->base._b;

    float_t *out = entry->base.out_ptr_;
    // float_t *out = input->in_ptr_;

    input->in_ptr_ = out;
    input->in_size_ = entry->base.out_size_;

    index3d in_ = entry->in_;
    uint64_t dim = in_.height_*in_.width_;

    uint64_t total_size = dim;
    uint64_t blocksize = compute_block_size(total_size);
    uint64_t start = (blocksize) * hart_id;
    uint64_t end = min((blocksize) * (hart_id+1), total_size);

    // printf("start batchNorm:\n");
    int b_size = end / 100, cnt = 0;
    for (uint64_t j = start; j < end; j++)
    {
        uint32_t tmp_3, tmp_4;
        // tmp_3 = j;
        // tmp_3 = o + 1;
        // tmp_4 = end;
        if(j % 100 == 0) {
            printf("[%s]: %d/%d\n", entry->base.layer_name_, cnt++, b_size);
        }
        // printf("start layer: %d\n", tmp_3);

        for (uint64_t ch = 0; ch < in_.depth_; ch++){
            uint64_t pos = ch * dim + j;

            // if (pos == 0) printf("%f %f %f %f %f\n", _gamma(entry, ch) , in[pos] , _mean(entry, ch), _invstd(entry, ch) , _beta(entry, ch));
            // a[pos] = _gamma(entry, ch) * (in[pos] - _mean(entry, ch)) * _invstd(entry, ch) + _beta(entry, ch);
            
            out[pos] = _gamma(entry, ch) * (in[pos] - _mean(entry, ch)) * _invstd(entry, ch) + _beta(entry, ch);
            
            // out[pos] = max((float_t)0, out[pos]);
            // out
            // if(out[pos] < (float_t)0) {
            //     out[pos] = (float_t)0;
            // }
        }
    }

    // uint64_t total_size = in_.depth_;
    // uint64_t blocksize = compute_block_size(total_size);
    // uint64_t start = (blocksize) * hart_id;
    // uint64_t end = min((blocksize) * (hart_id+1), total_size);

    // for (uint64_t ch = start; ch < end; ch++)
    // {
    //     for (uint64_t j = 0; j < dim; j++){
    //         uint64_t pos = ch * dim + j;
    //         a[pos] = _gamma(entry, ch) * (in[pos] - _mean(entry, ch)) * _invstd(entry, ch) + _beta(entry, ch);
    //     }
    // }
    // wait for other process done
    // atomic_or(&entry->base.a_done_flag, 1LL << hart_id);
    // while (entry->base.a_done_flag != entry->base.mask);

    total_size = entry->base.out_size_;
    blocksize = compute_block_size(total_size);
    start = (blocksize) * hart_id;
    end = min((blocksize) * (hart_id+1), total_size);
    for (uint64_t i = start; i < end; i++)
        out[i] = entry->base.activate(out, i, entry->base.out_size_);
    
    // wait for other process done
    // atomic_or(&entry->base.done_flag, 1LL << hart_id);
    // while (entry->base.done_flag != entry->base.mask);

    // free(entry->base.padded_ptr);
    free(in);

#ifdef PRINT_LAYER
    if (hart_id == 0) 
    {
        printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, out[0], out[1], out[entry->base.out_size_-2], out[entry->base.out_size_-1]);
    }
#endif
}


layer_base * new_batchnorm_layer(
                                cnn_controller *ctrl,
                                float_t(*activate) (float_t *, uint64_t, uint64_t),
                                uint64_t channels,
                                uint64_t in_width,
                                uint64_t in_height
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
    static uint64_t call_time = 0;
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
    return &ret->base;
}