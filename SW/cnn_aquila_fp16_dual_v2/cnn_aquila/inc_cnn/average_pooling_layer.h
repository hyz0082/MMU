#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
// #include <math.h>
#include "config.h"
#include "layer.h"
#include "list.h"
#include "util.h"
#include "activation_function.h"

#include "hw_cmd.h"

#include <time.h>

#ifndef USING_GEM5
#include "loader.h"
#endif

typedef struct _average_pooling_layer
{
    layer_base base;
    int stride_;
    my_float_t scale_factor_;
    int pooling_size_;
    index3d in_;
    index3d out_;
} average_pooling_layer;

average_pooling_layer * get_average_pooling_layer_entry(struct list_node *ptr)
{
    return list_entry(ptr, average_pooling_layer, base.list);
}

void average_pooling_layer_forward_propagation(struct list_node *ptr, unsigned int hart_id, input_struct *input)
{
#ifdef USING_GEM5
    clock_t  tick, ticks_per_msec = CLOCKS_PER_SEC/1000;
    tick = clock();
#endif

    average_pooling_layer *entry = get_average_pooling_layer_entry(ptr);
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
        if (entry->base.out_ptr_ != NULL) { // Check if memory was allocated
            memset((void*)entry->base.out_ptr_, 0, entry->base.out_size_ * sizeof(my_float_t));
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
    my_float_t *out = in;//entry->base.out_ptr_;
    input->in_ptr_ = out;
    input->in_size_ = entry->base.out_size_;
    int stride_ = entry->stride_;

    index3d in_ = entry->in_;
    index3d out_ = entry->out_;

    int total_size = entry->base.out_size_;
    int blocksize = compute_block_size(total_size);
    int start = (blocksize) * hart_id;
    int end = min((blocksize) * (hart_id+1), total_size);
    
    int dim = out_.height_*out_.width_;
    printf("start avg pooling:\n");

    int source = 1; // 0: sw, 1: hw

    // if(source) {
    //     my_float_t *dst = a;
    //     my_float_t *src = in;
    //     int max_len = 112;
    //     my_float_t *pimg = dst;
    //     my_float_t *pin  = src;
    //     for(int i = 0; i < entry->base.out_size_; i++) {
    //         write_dram_value_cmd(&a[i], read_dram_value_cmd(&in[i]));
            
    //     }
    //     // for(int i = 0; i < entry->base.out_size_; i+=max_len) {
    //     //     send_bn_mul_data(1, 0);
    //     //     send_bn_add_data(0, 0);
            
    //     //     int remain_len = min(max_len, entry->base.out_size_ - i);
    //     //     /*
    //     //     * send data
    //     //     */
    //     //     reset_sram_offset_cmd();
    //     //     set_length_cmd(remain_len);
    //     //     set_dram_read_input_cmd();
    //     //     uint32_t tmp_s;
    //     //     memcpy(&tmp_s, &pin, sizeof(tmp_s));
    //     //     set_addr_cmd(tmp_s);
    //     //     trigger_dram_read_cmd();
    //     //     wait_idle_cmd();
    //     //     /*
    //     //     * start BatchNorm
    //     //     */
    //     //     set_mode_cmd(1, remain_len);
    //     //     reset_relu_cmd();
    //     //     trigger_add_cmd();
    //     //     wait_idle_cmd();
    //     //     set_mode_cmd(0, 0);

    //     //     /*
    //     //     * write data
    //     //     */
    //     //     set_dram_write_lens_cmd(remain_len);
    //     //     set_num_lans_cmd(0);
    //     //     set_output_recv_cnt_cmd(0);
    //     //     memcpy(&tmp_s, &pimg, sizeof(tmp_s));
    //     //     set_dram_write_addr_cmd(0, tmp_s);
    //     //     set_dram_w_tr_cmd();
    //     //     wait_idle_cmd();

    //     //     pin += remain_len;
    //     //     pimg += remain_len; 
    //     //     __asm__ volatile ("nop");
    //     // }
    // }
    
    // for (int o = start; o < end; o++)
    //     out[o] = entry->base.activate(a, o, entry->base.out_size_);
    
    /*
     * free input here
     */
    // free(in);

#ifdef PRINT_LAYER
    if (hart_id == 0) 
    {
        // printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, (float)out[0], (float)out[1], (float)out[entry->base.out_size_-2], (float)out[entry->base.out_size_-1]);
        printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, (float_t)read_dram_value_cmd(&out[0]), (float_t)read_dram_value_cmd(&out[1]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-2]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-1]));
    }
#endif

#ifdef USING_GEM5
    tick = (clock() - tick)/ticks_per_msec;
    printf("It took %ld msec to perform avg pooling.\n\n", tick);
#endif

}

static int pool_out_dim(int in_size, int pooling_size, int stride)
{
    return (int)(((my_float_t)in_size - pooling_size) / stride) + 1;
}

layer_base * new_average_pooling_layer(
                                       cnn_controller *ctrl,
                                       my_float_t(*activate) (my_float_t *, int, int),
                                       int in_width,
                                       int in_height,
                                       int in_channels,
                                       int pooling_size,
                                       int stride
                                       )
{

// #ifndef USING_GEM5
    average_pooling_layer *ret = (average_pooling_layer *)malloc(sizeof(average_pooling_layer));
// #else
    // average_pooling_layer *ret = (average_pooling_layer *) ctrl->nwk_cur_ptr;
    // ctrl->nwk_cur_ptr += sizeof(average_pooling_layer);
// #endif
    ctrl->padding_size = 0;
    init_layer(&ret->base,
               ctrl,
               in_width*in_height*in_channels,
               pool_out_dim(in_width, pooling_size, stride) * pool_out_dim(in_height, pooling_size, stride) * in_channels, 
               0,
               0,
               activate==softmax);
#ifdef PRINT_LAYER
    static int call_time = 0;
    // sprintf(ret->base.layer_name_, "avg_pool%d", call_time++);
    my_sprintf(ret->base.layer_name_, "avg_pool%d", call_time++);
#endif
    ret->scale_factor_ = (my_float_t)1 / (pooling_size*pooling_size);
    ret->stride_ = stride;
    ret->pooling_size_ = pooling_size;
    ret->in_ = new_index3d(in_width, in_height, in_channels);
    ret->out_ = new_index3d(pool_out_dim(in_width, pooling_size, stride), pool_out_dim(in_height, pooling_size, stride), in_channels);

    ret->base.activate = activate;
    ret->base.forward_propagation = average_pooling_layer_forward_propagation;
    // printf("insize of average pooling layer %d\n", ret->base.in_size_);
    // printf("avg pool: in [%f, %f, ... , %f, %f]\n", ret->base.in_ptr_[0], ret->base.in_ptr_[1], ret->base.in_ptr_[ret->base.in_size_-2], ret->base.in_ptr_[ret->base.in_size_-1]);
    // printf("avg pool: b  [%f, %f, ... , %f, %f]\n", ret->base._b[0], ret->base._b[1], ret->base._b[in_channels-2], ret->base._b[in_channels-1]);
    return &ret->base;
}