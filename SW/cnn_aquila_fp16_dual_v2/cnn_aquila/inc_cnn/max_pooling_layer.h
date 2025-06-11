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

#include "hw_cmd.h"

#ifndef USING_GEM5
#include "loader.h"
#endif

typedef struct _max_pooling_layer
{
    layer_base base;
    int stride_;
    int pooling_size_;
    int padding_size_;
    index3d in_;
    index3d in_padded_;
    index3d out_;

    int padding_done_flag;
    int padding_mask;
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
        int padding_size = entry->padding_size_;

        my_float_t *in = input->in_ptr_;
        my_float_t *dst = entry->base.padded_ptr;

        int total_size = in_.depth_ * in_.height_;
        int blocksize = compute_block_size(total_size);
        int start = (blocksize) * hart_id;
        int end = min((blocksize) * (hart_id+1), total_size);
    
        for (int i = start; i < end; i++)
        {

            int c = i / in_.height_;
            int y = i % in_.height_;
            // printf("[%d, %d]\n", c, y);
            my_float_t *pimg = &dst[get_index(&in_padded_, padding_size, padding_size + y, c)];
            const my_float_t *pin = &in[get_index(&in_, 0, y, c)];

            for (int x = 0; x < in_.width_; x++)
            {
                // pimg[x] = pin[x];
                my_float_t tmp = read_dram_value_cmd(&pin[x]);
                write_dram_value_cmd(&pimg[x], tmp);
            }
            // int max_len = 100;
            // for (int j = 0; j < in_.width_; j += max_len) {
            //     int remain_len = min(max_len, in_.width_ - j);
            //     /*
            //     * send data
            //     */
            //     reset_sram_offset_cmd();
            //     set_length_cmd(remain_len);
            //     // set_dram_read_input_cmd();
            //     uint32_t tmp_s;
            //     memcpy(&tmp_s, &pin, sizeof(tmp_s));
            //     set_addr_cmd(tmp_s);
            //     trigger_dram_read_cmd();
            //     wait_idle_cmd();
            //     /*
            //     * start BatchNorm
            //     */
            //     set_mode_cmd(1, remain_len);
            //     // reset_relu_cmd();
            //     trigger_add_cmd();
            //     wait_idle_cmd();
            //     set_mode_cmd(0, 0);
 
            //     /*
            //     * write data
            //     */
            //     set_dram_write_lens_cmd(remain_len);
            //     // set_num_lans_cmd(0);
            //     set_output_recv_cnt_cmd(0);
            //     memcpy(&tmp_s, &pimg, sizeof(tmp_s));
            //     set_dram_write_addr_cmd(0, tmp_s);
            //     set_dram_w_tr_cmd();
            //     wait_idle_cmd();

            //     pin += remain_len;
            //     pimg += remain_len; 
            // }
        }
    }
    else
        entry->base.padded_ptr = input->in_ptr_;
}

void max_pooling_layer_forward_propagation(struct list_node *ptr, unsigned int hart_id, input_struct *input)
{
#ifdef USING_GEM5
    clock_t  tick, ticks_per_msec = CLOCKS_PER_SEC/1000;
    clock_t hardware_compute_time = 0;
    tick = clock();
#endif
    max_pooling_layer *entry = get_max_pooling_layer_entry(ptr);
    if (input->in_size_ != entry->base.in_size_)
    {
        if (hart_id == 0) 
            printf("Error input size not match %d/%d\n", input->in_size_, entry->base.in_size_);
        exit(-1);
    }


    // comment start

//     entry->base.padded_ptr = (my_float_t *)malloc(entry->base.padding_size * sizeof(my_float_t));
//     if (entry->base.padded_ptr != NULL) { // Check if memory was allocated
//         // memset((void*)entry->base.padded_ptr, 0, entry->base.padding_size * sizeof(my_float_t));
//         // for(int i = 0; i < entry->base.padding_size; i++) {
//         //     write_dram_value_cmd(&entry->base.padded_ptr[i], 0);
//         // }
//         reset_dram_value_cmd(entry->base.padded_ptr, entry->base.padding_size);
//     }
//     else {
//         printf("Error: Unable to allocate memory for layer->padded_ptr\n");
//         // exit(1);
//     }

//     // malloc a_ptr
//     entry->base.a_ptr_ = (my_float_t *)malloc(entry->base.out_size_ * sizeof(my_float_t));
//     if (entry->base.a_ptr_ != NULL) { // Check if memory was allocated
//         // memset((void*)entry->base.a_ptr_, 0, entry->base.out_size_ * sizeof(my_float_t));
//         // for(int i = 0; i < entry->base.padding_size; i++) {
//         //     write_dram_value_cmd(&entry->base.a_ptr_[i], 0);
//         // }
//         reset_dram_value_cmd(entry->base.a_ptr_, entry->base.out_size_);
//     }
//     else {
//         printf("Error: Unable to allocate memory for entry->base.a_ptr_\n");
//         // exit(1);
//     }

//     if (entry->base.need_space_for_a){
//         entry->base.out_ptr_ = (my_float_t *)malloc(entry->base.out_size_ * sizeof(my_float_t));
//         if (entry->base.out_ptr_ != NULL) { // Check if memory was allocated
//             memset((void*)entry->base.out_ptr_, 0, entry->base.out_size_ * sizeof(my_float_t));
//         }
//         else {
//             printf("Error: Unable to allocate memory for entry->base.out_ptr_\n");
//             // exit(1);
//         }
//     }
//     else {
//         entry->base.out_ptr_ = entry->base.a_ptr_;
//     }

//     int skip_en = 0;
//     /*
//      * max pooling
//      */
//     if(!skip_en) {
//         pool_copy_and_pad_input(entry, hart_id, input);
//     }
    
//     // remove
//     if(skip_en) {
//     my_float_t *dst = entry->base.out_ptr_;
//     my_float_t *src = input->in_ptr_;
//     int max_len = 112;
//     my_float_t *pimg = dst;
//     my_float_t *pin  = src;
//     for(int i = 0; i < entry->base.out_size_; i+=max_len) {
//         // entry->base.out_ptr_[i] = input->in_ptr_[i];
//         // write_dram_value_cmd(dst + i, read_dram_value_cmd(src + i));
//         // write_dram_value_cmd(&entry->base.out_ptr_[i], read_dram_value_cmd(&input->in_ptr_[i]));
//         send_bn_mul_data(1, 0);
//         send_bn_add_data(0, 0);
        
//         int remain_len = min(max_len, entry->base.out_size_ - i);
//         /*
//         * send data
//         */
//         reset_sram_offset_cmd();
//         set_length_cmd(remain_len);
//         set_dram_read_input_cmd();
//         uint32_t tmp_s;
//         memcpy(&tmp_s, &pin, sizeof(tmp_s));
//         set_addr_cmd(tmp_s);
//         trigger_dram_read_cmd();
//         wait_idle_cmd();
//         /*
//         * start BatchNorm
//         */
//         set_mode_cmd(1, remain_len);
//         reset_relu_cmd();
//         trigger_add_cmd();
//         wait_idle_cmd();
//         set_mode_cmd(0, 0);

//         /*
//         * write data
//         */
//         set_dram_write_lens_cmd(remain_len);
//         set_num_lans_cmd(0);
//         set_output_recv_cnt_cmd(0);
//         memcpy(&tmp_s, &pimg, sizeof(tmp_s));
//         set_dram_write_addr_cmd(0, tmp_s);
//         set_dram_w_tr_cmd();
//         wait_idle_cmd();

//         pin += remain_len;
//         pimg += remain_len; 
//         __asm__ volatile ("nop");
        
// #ifdef USING_GEM5
//         clock_t  tmp_tick = clock();
//         hardware_compute_time += (clock() - tmp_tick)/(ticks_per_msec/1000);
//         tmp_tick = clock();
//         hardware_compute_time += (clock() - tmp_tick)/(ticks_per_msec/100);
// #endif
//     }
//     }

//     free(input->in_ptr_);
    // comment end

    // my_float_t *in = entry->base.padded_ptr;
    // my_float_t *a = entry->base.a_ptr_;
    // my_float_t *out = entry->base.out_ptr_;
    my_float_t *out = input->in_ptr_;
    input->in_ptr_ = out;
    input->in_size_ = entry->base.out_size_;
    int stride_ = entry->stride_;

    index3d in_padded_ = entry->in_padded_;
    index3d out_ = entry->out_;

    int total_size = entry->base.out_size_;
    int blocksize = compute_block_size(total_size);
    int start = (blocksize) * hart_id;
    int end = min((blocksize) * (hart_id+1), total_size);

    int dim = out_.height_*out_.width_;

    printf("start max pooling:\n");
    // for(int i = 0; i < total_size; i++) {
    //     out[i] = input->in_ptr_[i];
    // }
    // free(input->in_ptr_);
    /*
     * max pooling
     */
    // if(!skip_en)
    // for (int o = start; o < end; o++)
    // {
    //     uint32_t tmp_3, tmp_4;
    //     tmp_3 = o;

    //     int c = o / dim;
    //     // a[o] = (my_float_t)-DBL_MAX;
    //     // write_dram_value_cmd(&a[o], (my_float_t)-DBL_MAX);
    //     int xy = o % dim;
    //     int dsty = xy / out_.width_;
    //     int dstx = xy % out_.width_;
    //     int y = dsty*stride_;
    //     int x = dstx*stride_;
    //     int dymax = min(entry->pooling_size_, in_padded_.height_ - y);
    //     int dxmax = min(entry->pooling_size_, in_padded_.width_ - x);

    //     // for (int dy = 0; dy < dymax; dy++)
    //     //     for (int dx = 0; dx < dxmax; dx++)
    //     //     {
    //     //         my_float_t tmp_1, tmp_2;
    //     //         tmp_1 = read_dram_value_cmd(&a[o]);
    //     //         tmp_2 = read_dram_value_cmd(&in[get_index(&in_padded_, x + dx, y + dy, c)]);
    //     //         write_dram_value_cmd(&a[o], max(tmp_1, tmp_2));
    //     //         // a[o] = max(a[o], in[get_index(&in_padded_, x + dx, y + dy, c)]);
    //     //     }
    //     my_float_t max_fp16 = (my_float_t)-DBL_MAX;
    //     for (int dy = 0; dy < dymax; dy++)
    //         for (int dx = 0; dx < dxmax; dx++)
    //         {
    //             my_float_t tmp_1;
    //             tmp_1 = read_dram_value_cmd(&in[get_index(&in_padded_, x + dx, y + dy, c)]);
    //             // write_dram_value_cmd(&a[o], max(tmp_1, tmp_2));
    //             max_fp16 = max(max_fp16, tmp_1);
    //             // a[o] = max(a[o], in[get_index(&in_padded_, x + dx, y + dy, c)]);
    //         }
    //     write_dram_value_cmd(&a[o], max_fp16);
    // }


    // for (int o = start; o < end; o++)
    //     out[o] = entry->base.activate(a, o, entry->base.out_size_);

    // free(entry->base.padded_ptr);

    //###########################################################
    // print output tensor
    // if(1)
    // {
    //     printf("output:\n");
    //     printf("\n\n[%s]\n", entry->base.layer_name_);
    //     printf("shape depth:%d,  height:%d width:%d\n", (int)out_.depth_, (int)out_.height_, (int)out_.width_);
    //     printf("    ");
    //     for(int i = 0; i < out_.width_; i++) {
    //         printf("%6d ", (i));
    //     }
    //     printf("\n");
    //     // const my_float_t *pi = &out[0];
    //     int p_cnt = 0;
    //     for (int inc = 0; inc < out_.depth_; inc++) {
    //         printf("\n[depth%d]\n", inc);
    //         // const my_float_t *pi = &in[get_index(&in_padded_, 0, 0, inc)];
    //         for (int h = 0; h < out_.height_; h++) {
    //             printf("%3d ", (h));
    //             printf("[ ");
    //             for (int w = 0; w < out_.width_; w++) {
    //                 // printf("%2.6f ", ((float)out[p_cnt]));
    //                 printf("%2.6f ", ((float_t)read_dram_value_cmd(&out[p_cnt])));
                    
    //                 p_cnt++;
    //             }
    //             printf("]\n");
    //         }
    //     }
    // }
    //###########################################################


    // comment end

#ifdef PRINT_LAYER
    if (hart_id == 0) 
    {
        // printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, out[0], out[1], out[entry->base.out_size_-2], out[entry->base.out_size_-1]);
        printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, (float_t)read_dram_value_cmd(&out[0]), (float_t)read_dram_value_cmd(&out[1]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-2]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-1]));
    }
#endif

#ifdef USING_GEM5
    tick = (clock() - tick)/ticks_per_msec;
    printf("It took %ld msec to perform max pooling.\n\n", tick);
#endif
}

layer_base * new_max_pooling_layer(
                                cnn_controller *ctrl,
                                my_float_t(*activate) (my_float_t *, int, int),
                                int in_width,
                                int in_height,
                                int in_channels,
                                int pooling_size,
                                int stride,
                                int padding_size
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
    static int call_time = 0;
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