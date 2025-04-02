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
#include <math.h>
#ifndef USING_GEM5
#include "loader.h"
#endif

//
#include "syscalls.h"
//

typedef struct _fully_connected_layer
{
    layer_base base;

    uint8_t has_bias_;
} fully_connected_layer;

fully_connected_layer * get_fully_connected_layer_entry(struct list_node *ptr)
{
    return list_entry(ptr, fully_connected_layer, base.list);
}

void fully_connected_layer_forward_propagation(struct list_node *ptr, unsigned int hart_id, input_struct *input)
{
#ifdef USING_GEM5
    clock_t  tick, ticks_per_msec = CLOCKS_PER_SEC/1000;
    tick = clock();
#endif

    fully_connected_layer *entry = get_fully_connected_layer_entry(ptr);
    
    if (input->in_size_ != entry->base.in_size_)
    {
        if (hart_id == 0) 
            printf("Error input size not match %d/%d\n", input->in_size_, entry->base.in_size_);
        exit(-1);
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

    float_t *in = input->in_ptr_;
    float_t *a = entry->base.a_ptr_;
    float_t *W = entry->base._W;
    float_t *b = entry->base._b;
    float_t *out = entry->base.out_ptr_;
    input->in_ptr_ = out;
    input->in_size_ = entry->base.out_size_;

    uint64_t total_size = entry->base.out_size_;
    uint64_t blocksize = compute_block_size(total_size);
    uint64_t start = (blocksize) * hart_id;
    uint64_t end = min((blocksize) * (hart_id+1), total_size);
    printf("start fc\n");
    reset_cmd();
    int kernel_len = entry->base.in_size_;
    int weight_gbuff_size = 32768;//32000; //32768
    int weight_num = weight_gbuff_size / kernel_len;

    set_KMN_cmd(kernel_len , 4, weight_num);
    set_conv_cmd(kernel_len);
    set_idx_cmd(0, kernel_len, kernel_len*2, kernel_len*3);
    reset_preload_cmd();
    for(int i = 0; i < kernel_len*4; i++) {
        send_idx_cmd(i, i);
    }
    // send data
    for (uint64_t c = 0; c < entry->base.in_size_; c++) {
        send_data_cmd(in[c], c);
    }
    // for (uint64_t c = 0; c < entry->base.in_size_; c++) {
    //     send_data_cmd(in[c], c+kernel_len);
    // }
    // for (uint64_t c = 0; c < entry->base.in_size_; c++) {
    //     send_data_cmd(in[c], c+kernel_len*2);
    // }
    // for (uint64_t c = 0; c < entry->base.in_size_; c++) {
    //     send_data_cmd(in[c], c+kernel_len*3);
    // }
    
    // printf("start fc bbbbbbb\n");
    for (uint64_t i = start; i < end; i += weight_num)
    {
        // send weight_num weight
        int remain_len = min(weight_num, end - i);
        int send_weight_cnt = 0;
        for(int oc = 0; oc < remain_len; oc++) {
            for (uint64_t c = 0; c < entry->base.in_size_; c++) {
                send_weight_cmd(W[(i+oc)*entry->base.in_size_ + c], send_weight_cnt++);
            }
        }
        // for (uint64_t c = 0; c < entry->base.in_size_; c++) {
        //     send_weight_cmd(W[(i)*entry->base.in_size_ + c], send_weight_cnt++);
        // }
        __asm__ volatile ("nop");
        __asm__ volatile ("nop");
        trigger_conv_cmd();
        wait_idle_cmd();
        int var[4] = {0, 4, 8, 12};
        for(int m = 0; m < remain_len; m++) {
            a[i+m] = read_data_cmd(var[m%4], m/4);
        // a[i+1] = read_data_cmd(4, 0);
        // a[i+2] = read_data_cmd(8, 0);
        // a[i+3] = read_data_cmd(12, 0);
        }
        

        if (entry->has_bias_) {
            for(int m = 0; m < remain_len; m++) {
            a[i+m] += b[i+m];
            // a[i+1] += b[i+1];
            // a[i+2] += b[i+2];
            // a[i+3] += b[i+3];
            }
        } 
    }

#ifdef USING_GEM5
    // tick = (clock() - tick)/ticks_per_msec;
    printf("It took %ld msec to perform fc without softmax.\n\n", (clock() - tick)/ticks_per_msec);
#endif
    // tick = (clock() - tick)/ticks_per_msec;
    // printf("It took %ld msec to perform fc without softmax.\n\n", tick);
    // for (uint64_t i = start; i < end; i++)
    // {
    //     uint32_t tmp_3, tmp_4;
    //     tmp_3 = i;

    //     a[i] = (float_t)0;
    //     for (uint64_t c = 0; c < entry->base.in_size_; c++)
    //         a[i] += W[i*entry->base.in_size_ + c] * in[c];

    //     if (entry->has_bias_)
    //         a[i] += b[i];
    // }

     float_t tmp = exp(a[0]);
    out[0] += tmp;
    // for (uint64_t i = start; i < end; i++)
    //     out[i] = entry->base.activate(a, i, entry->base.out_size_);
    

    // float_t max = 0;
    // float_t denom = 0;
    // for (uint64_t i = start; i < end; i++) {
    //     if (a[i] > max) {
    //         max = a[i];
    //     }
    // }
    // for (uint64_t i = start; i < end; i++) {
    //     denom += exp(a[i] - max);
    // }
    // for (uint64_t i = start; i < end; i++) {
    //     float_t numer = exp(a[i] - max);
    //     out[i] = numer / denom;
    // }


    // for (uint64_t i = start; i < end; i++) {
    //     send_exp_acc_cmd(a[i]);
    //     wait_idle_cmd();
    // }
    // for (uint64_t i = start; i < end; i++) {
    //     sf_calc_cmd(a[i]);
    //     wait_idle_cmd();
    //     out[i] = read_sf_cmd();
    // }
    // wait for other process done
    // atomic_or(&entry->base.done_flag, 1LL << hart_id);
    // while (entry->base.done_flag != entry->base.mask);
    free(in);

#ifdef PRINT_LAYER
    if (hart_id == 0) 
    {
        printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, out[0], out[1], out[entry->base.out_size_-2], out[entry->base.out_size_-1]);
    }
#endif

#ifdef USING_GEM5
    tick = (clock() - tick)/ticks_per_msec;
    printf("It took %ld msec to perform fc.\n\n", tick);
#endif
}


layer_base * new_fully_connected_layer(
                                       cnn_controller *ctrl,
                                       float_t(*activate) (float_t *, uint64_t, uint64_t),
                                       uint64_t in_dim,
                                       uint64_t out_dim,
                                       uint8_t has_bias
                                       )
{

// #ifndef USING_GEM5
    fully_connected_layer *ret = (fully_connected_layer *)malloc(sizeof(fully_connected_layer));
// #else
//     fully_connected_layer *ret = (fully_connected_layer *) ctrl->nwk_cur_ptr;
//     ctrl->nwk_cur_ptr += sizeof(fully_connected_layer);
// #endif
    ctrl->padding_size = 0;
    init_layer(&ret->base,
               ctrl,
               in_dim,
               out_dim,
               in_dim * out_dim,
               has_bias ? out_dim : 0,
               activate==softmax);
#ifdef PRINT_LAYER
    static uint64_t call_time = 0;
    // sprintf(ret->base.layer_name_, "fc%d", call_time++);
    my_sprintf(ret->base.layer_name_, "fc%d", call_time++);
#endif
    ret->has_bias_ = has_bias;
    ret->base.activate = activate;
    // printf("insize of FC layer %d\n", ret->base.in_size_);
    // // printf("FC: in [%f, %f, ... , %f, %f]\n", ret->base.in_ptr_[0], ret->base.in_ptr_[1], ret->base.in_ptr_[ret->base.in_size_-2], ret->base.in_ptr_[ret->base.in_size_-1]);
    printf("FC: W  [%f, %f, ... , %f, %f]\n", ret->base._W[0], ret->base._W[1], ret->base._W[in_dim * out_dim-2], ret->base._W[in_dim * out_dim-1]);
    // printf("FC: b  [%f, %f, ... , %f, %f]\n", ret->base._b[0], ret->base._b[1], ret->base._b[out_dim-2], ret->base._b[out_dim-1]);
    ret->base.forward_propagation = fully_connected_layer_forward_propagation;
    return &ret->base;
}
