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
    entry->base.a_ptr_ = (my_float_t *)malloc(entry->base.out_size_ * sizeof(my_float_t));
    if (entry->base.a_ptr_ != NULL) {
    }
    else {
        printf("Error: Unable to allocate memory for entry->base.a_ptr_\n");
        // exit(1);
    }

    if (entry->base.need_space_for_a){
        entry->base.out_ptr_ = (my_float_t *)malloc(entry->base.out_size_ * sizeof(my_float_t));
        if (entry->base.out_ptr_ != NULL) {
        }
        else {
            printf("Error: Unable to allocate memory for entry->base.out_ptr_\n");
            // exit(1);
        }
    }
    else {
        entry->base.out_ptr_ = entry->base.a_ptr_;
    }

    my_float_t *in = input->in_ptr_;
    my_float_t *a = entry->base.a_ptr_;
    my_float_t *W = entry->base._W;
    my_float_t *b = entry->base._b;
    my_float_t *out = entry->base.out_ptr_;
    input->in_ptr_ = out;
    input->in_size_ = entry->base.out_size_;

    int total_size = entry->base.out_size_;
    int blocksize = compute_block_size(total_size);
    int start = (blocksize) * hart_id;
    int end = min((blocksize) * (hart_id+1), total_size);
    printf("start fc\n");
    reset_cmd();
    int kernel_len = entry->base.in_size_;
    int weight_gbuff_size = 8192;
    int weight_num = weight_gbuff_size / kernel_len;

    set_KMN_cmd(kernel_len , 4, weight_num);
    set_conv_cmd(kernel_len);
    // set_idx_cmd(0, kernel_len, kernel_len*2, kernel_len*3);
    set_idx_cmd(0, 0, 0, 0);
    set_col_idx_cmd(0, kernel_len, kernel_len*2, kernel_len*3);

    reset_preload_cmd();

    set_gemm_core_sel_cmd(1);

    for(int i = 0; i < kernel_len*4; i++) {
        send_idx_cmd(i, i);
    }

    for(int i = 0; i < 1024; i++) {
        send_offset_1_cmd(0, i);
        send_offset_2_cmd(0, i);
        send_offset_3_cmd(0, i);
        send_offset_4_cmd(0, i);
    }

    reset_relu_cmd();

    my_float_t *pi = in;
    reset_sram_offset_cmd();
    set_dram_read_input_cmd();
    for(int c = 0; c < entry->base.in_size_; c += 100) {
        int remain_len = min(100, entry->base.in_size_ - c);
        set_length_cmd(remain_len);
        uint32_t tmp_s;
        memcpy(&tmp_s, &pi, sizeof(tmp_s));
        set_addr_cmd(tmp_s);
        trigger_dram_read_cmd();
        wait_idle_quick_cmd();
        pi += 100;
    }
    
    my_float_t cur_max = -9999;

    for (int i = start; i < end; i += weight_num)
    {
        // send weight_num weight
        int remain_len = min(weight_num, end - i);
        int send_weight_cnt = 0;
        
        // read weight
        set_dram_read_weight_cmd();
        reset_sram_offset_cmd();
        set_length_cmd(entry->base.in_size_);
        for(int oc = 0; oc < remain_len; oc++) {
            const my_float_t * ppw = &W[(i+oc)*entry->base.in_size_];
            uint32_t tmp_s;
            memcpy(&tmp_s, &ppw, sizeof(tmp_s));
            set_addr_cmd(tmp_s);
            trigger_dram_read_cmd();
            wait_idle_quick_cmd();
        }

        int batchNormIndex = 0;
        for(int oc = 0; oc < remain_len; oc+= 4) {
            set_mul_sram_0_cmd(1, batchNormIndex);
            set_mul_sram_1_cmd(1, batchNormIndex);
            set_mul_sram_2_cmd(1, batchNormIndex);
            set_mul_sram_3_cmd(1, batchNormIndex);

            set_add_sram_0_cmd(0, batchNormIndex);
            set_add_sram_1_cmd(0, batchNormIndex);
            set_add_sram_2_cmd(0, batchNormIndex);
            set_add_sram_3_cmd(0, batchNormIndex);
            batchNormIndex ++;
            __asm__ volatile ("nop");
        }

        __asm__ volatile ("nop");
        __asm__ volatile ("nop");
        trigger_conv_cmd();
        wait_idle_quick_cmd();
        int var[4] = {0, 4, 8, 12};
        for(int m = 0; m < remain_len; m++) {
            static int fc_num = 0;
            printf("got fc %d: %f\n", fc_num++, (float)read_data_cmd(var[m%4], m/4));
            if (entry->has_bias_) {
                    my_float_t tmp_a = read_data_cmd(var[m%4], m/4) + read_dram_value_cmd(&b[i+m]);
                    cur_max = max(cur_max, tmp_a);
                    write_dram_value_cmd(&a[i+m], tmp_a);
            }
            else {
                write_dram_value_cmd(&a[i+m], read_data_cmd(var[m%4], m/4));
            }
        }
    }

    for (int i = start; i < end; i++) {
        send_exp_acc_cmd(read_dram_value_cmd(&a[i]) - cur_max);
        wait_idle_quick_cmd();
    }
    for (int i = start; i < end; i++) {
        sf_calc_cmd(read_dram_value_cmd(&a[i]) - cur_max);
        wait_idle_quick_cmd();
        write_dram_value_cmd(&out[i], read_sf_cmd());
    }

    free(in);


#ifdef USING_GEM5
    tick = (clock() - tick)/ticks_per_msec;
    printf("It took %ld msec to perform fc.\n\n", tick);
#endif
}


layer_base * new_fully_connected_layer(
                                       cnn_controller *ctrl,
                                       my_float_t(*activate) (my_float_t *, int, int),
                                       int in_dim,
                                       int out_dim,
                                       uint8_t has_bias
                                       )
{

    fully_connected_layer *ret = (fully_connected_layer *)malloc(sizeof(fully_connected_layer));
    ctrl->padding_size = 0;
    init_layer(&ret->base,
               ctrl,
               in_dim,
               out_dim,
               in_dim * out_dim,
               has_bias ? out_dim : 0,
               activate==softmax);
#ifdef PRINT_LAYER
    static int call_time = 0;
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