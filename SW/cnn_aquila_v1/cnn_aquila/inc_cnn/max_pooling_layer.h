#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
// #include <math.h>
#include <float.h>
#include "config.h"
#include "layer.h"
#include "list.h"
#include "util.h"
#include "activation_function.h"
#include "average_pooling_layer.h"

#ifndef USING_GEM5
#include "loader.h"
#endif

typedef struct _max_pooling_layer
{
    layer_base base;
    uint64_t stride_;
    uint64_t pooling_size_;
    uint64_t padding_size_;
    index3d in_;
    index3d in_padded_;
    index3d out_;

    uint64_t padding_done_flag;
    uint64_t padding_mask;
} max_pooling_layer;

max_pooling_layer * get_max_pooling_layer_entry(struct list_node *ptr)
{
    return list_entry(ptr, max_pooling_layer, base.list);
}

void pool_copy_and_pad_input(max_pooling_layer *entry, unsigned int hart_id, input_struct *input)
{
    if (entry->padding_size_)
    {
        index3d in_ = entry->in_;
        index3d in_padded_ = entry->in_padded_;
        uint64_t padding_size = entry->padding_size_;

        float_t *in = input->in_ptr_;
        float_t *dst = entry->base.padded_ptr;


        
        uint64_t total_size = in_.depth_ * in_.height_;
        uint64_t blocksize = compute_block_size(total_size);
        uint64_t start = (blocksize) * hart_id;
        uint64_t end = min((blocksize) * (hart_id+1), total_size);
    
        for (uint64_t i = start; i < end; i++)
        {

            uint64_t c = i / in_.height_;
            uint64_t y = i % in_.height_;
            // printf("[%d, %d]\n", c, y);
            float_t *pimg = &dst[get_index(&in_padded_, padding_size, padding_size + y, c)];
            const float_t *pin = &in[get_index(&in_, 0, y, c)];

            for (uint64_t x = 0; x < in_.width_; x++)
            {
                pimg[x] = pin[x];
            }
        }
        
        // wait for other process done
        // atomic_or(&entry->padding_done_flag, 1LL << hart_id);
        // while (entry->padding_done_flag != entry->padding_mask);
    }
    else
        entry->base.padded_ptr = input->in_ptr_;
}

void max_pooling_layer_forward_propagation(struct list_node *ptr, unsigned int hart_id, input_struct *input)
{
#ifdef USING_GEM5
    clock_t  tick, ticks_per_msec = CLOCKS_PER_SEC;
    tick = clock();
#endif
    max_pooling_layer *entry = get_max_pooling_layer_entry(ptr);
    if (input->in_size_ != entry->base.in_size_)
    {
        if (hart_id == 0) 
            printf("Error input size not match %d/%d\n", input->in_size_, entry->base.in_size_);
        exit(-1);
    }

    entry->base.padded_ptr = (float_t *)malloc(entry->base.padding_size * sizeof(float_t));
    if (entry->base.padded_ptr != NULL) { // Check if memory was allocated
        memset((void*)entry->base.padded_ptr, 0, entry->base.padding_size * sizeof(float_t));
    }
    else {
        printf("Error: Unable to allocate memory for layer->padded_ptr\n");
        // exit(1);
    }

    // malloc a_ptr
    entry->base.a_ptr_ = (float_t *)malloc(entry->base.out_size_ * sizeof(float_t));
    if (entry->base.a_ptr_ != NULL) { // Check if memory was allocated
        memset((void*)entry->base.a_ptr_, 0, entry->base.out_size_ * sizeof(float_t));
    }
    else {
        printf("Error: Unable to allocate memory for entry->base.a_ptr_\n");
        // exit(1);
    }

    if (entry->base.need_space_for_a){
        entry->base.out_ptr_ = (float_t *)malloc(entry->base.out_size_ * sizeof(float_t));
        if (entry->base.out_ptr_ != NULL) { // Check if memory was allocated
            memset((void*)entry->base.out_ptr_, 0, entry->base.out_size_ * sizeof(float_t));
        }
        else {
            printf("Error: Unable to allocate memory for entry->base.out_ptr_\n");
            // exit(1);
        }
    }
    else {
        entry->base.out_ptr_ = entry->base.a_ptr_;
    }

    pool_copy_and_pad_input(entry, hart_id, input);
    free(input->in_ptr_);

    float_t *in = entry->base.padded_ptr;
    float_t *a = entry->base.a_ptr_;
    float_t *out = entry->base.out_ptr_;
    input->in_ptr_ = out;
    input->in_size_ = entry->base.out_size_;
    uint64_t stride_ = entry->stride_;

    index3d in_padded_ = entry->in_padded_;
    index3d out_ = entry->out_;

    uint64_t total_size = entry->base.out_size_;
    uint64_t blocksize = compute_block_size(total_size);
    uint64_t start = (blocksize) * hart_id;
    uint64_t end = min((blocksize) * (hart_id+1), total_size);

    uint64_t dim = out_.height_*out_.width_;

    printf("start max pooling:\n");
    for (uint64_t o = start; o < end; o++)
    {
        uint32_t tmp_3, tmp_4;
        tmp_3 = o;
        // printf("start layer: %d\n", tmp_3);

        uint64_t c = o / dim;
        a[o] = (float_t)-DBL_MAX;
        uint64_t xy = o % dim;
        uint64_t dsty = xy / out_.width_;
        uint64_t dstx = xy % out_.width_;
        uint64_t y = dsty*stride_;
        uint64_t x = dstx*stride_;
        uint64_t dymax = min(entry->pooling_size_, in_padded_.height_ - y);
        uint64_t dxmax = min(entry->pooling_size_, in_padded_.width_ - x);

        for (uint64_t dy = 0; dy < dymax; dy++)
            for (uint64_t dx = 0; dx < dxmax; dx++)
            {
                a[o] = max(a[o], in[get_index(&in_padded_, x + dx, y + dy, c)]);
            }
    }
    // wait for other process done
    // atomic_or(&entry->base.a_done_flag, 1LL << hart_id);
    // while (entry->base.a_done_flag != entry->base.mask);
    for (uint64_t o = start; o < end; o++)
        out[o] = entry->base.activate(a, o, entry->base.out_size_);
    
    // wait for other process done
    // atomic_or(&entry->base.done_flag, 1LL << hart_id);
    // while (entry->base.done_flag != entry->base.mask);

    free(entry->base.padded_ptr);

#ifdef PRINT_LAYER
    if (hart_id == 0) 
    {
        printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, out[0], out[1], out[entry->base.out_size_-2], out[entry->base.out_size_-1]);
    }
#endif

#ifdef USING_GEM5
    tick = (clock() - tick)/ticks_per_msec;
    printf("It took %ld msec to perform max pooling.\n\n", tick);
#endif
}

layer_base * new_max_pooling_layer(
                                cnn_controller *ctrl,
                                float_t(*activate) (float_t *, uint64_t, uint64_t),
                                uint64_t in_width,
                                uint64_t in_height,
                                uint64_t in_channels,
                                uint64_t pooling_size,
                                uint64_t stride,
                                uint64_t padding_size
                                )
{

// #ifndef USING_GEM5
    max_pooling_layer *ret = (max_pooling_layer *)malloc(sizeof(max_pooling_layer));
// #else
//     max_pooling_layer *ret = (max_pooling_layer *) ctrl->nwk_cur_ptr;
//     ctrl->nwk_cur_ptr += sizeof(max_pooling_layer);
// #endif
    if (padding_size)
        ctrl->padding_size = (in_width + 2*padding_size) * (in_height + 2*padding_size) * in_channels;
    else 
        ctrl->padding_size = 0;
    
    init_layer(&ret->base,
               ctrl,
               in_width*in_height*in_channels,
               pool_out_dim(in_width + 2*padding_size, pooling_size, stride) * pool_out_dim(in_height + 2*padding_size, pooling_size, stride) * in_channels, 
               0,
               0,
               activate==softmax);
#ifdef PRINT_LAYER
    static uint64_t call_time = 0;
    // sprintf(ret->base.layer_name_, "max_pool%d", call_time++);
    my_sprintf(ret->base.layer_name_, "max_pool%d", call_time++);
#endif
    ret->stride_ = stride;
    ret->pooling_size_ = pooling_size;
    ret->padding_size_ = padding_size;
    ret->in_ = new_index3d(in_width, in_height, in_channels);
    ret->in_padded_ = new_index3d(in_width + 2*padding_size, in_height + 2*padding_size, in_channels);
    ret->out_ = new_index3d(pool_out_dim(in_width + 2*padding_size, pooling_size, stride), pool_out_dim(in_height + 2*padding_size, pooling_size, stride), in_channels);
    if (padding_size)
    {
        ret->padding_done_flag = 0;
        ret->padding_mask = (ctrl->total_CPUs < 64) ? (1LL << ctrl->total_CPUs) - 1 : 0LL - 1;
    }

    ret->base.activate = activate;
    ret->base.forward_propagation = max_pooling_layer_forward_propagation;
    // printf("insize of max pooling layer %d\n", ret->base.in_size_);
    // printf("max pool: in [%f, %f, ... , %f, %f]\n", ret->base.in_ptr_[0], ret->base.in_ptr_[1], ret->base.in_ptr_[ret->base.in_size_-2], ret->base.in_ptr_[ret->base.in_size_-1]);
    // printf("max pool: b  [%f, %f, ... , %f, %f]\n", ret->base._b[0], ret->base._b[1], ret->base._b[in_channels-2], ret->base._b[in_channels-1]);
    return &ret->base;
}