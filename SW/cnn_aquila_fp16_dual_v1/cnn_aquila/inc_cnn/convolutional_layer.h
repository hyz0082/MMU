#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
// #include <math.h>
#include <string.h>
#include "config.h"
#include "layer.h"
#include "list.h"
#include "util.h"
#include "activation_function.h"
#include "../file_read.h"
#include "hw_cmd.h"
// #include "res"
#include "global_var.h"
// #include "conv30_w.h"

#include <time.h>

#ifndef USING_GEM5
#include "loader.h"
#endif

enum padding{
    valid,
    same
};

typedef struct _convolutional_layer
{
    layer_base base;
    index3d in_;
    index3d in_padded_;
    index3d out_;
    index3d weight_;
    index3d padding_;
    enum padding pad_type_;
    uint64_t w_stride_;
    uint64_t h_stride_;
    uint8_t has_bias_;

    uint64_t padding_done_flag;
    uint64_t padding_mask;

    uint8_t delete_input;
} convolutional_layer;

convolutional_layer * get_convolutional_layer_entry(struct list_node *ptr)
{
    return list_entry(ptr, convolutional_layer, base.list);
}

static uint64_t in_length(uint64_t in_length, uint64_t padding_size, enum padding pad_type)
{
    return pad_type == same ? in_length + 2 * padding_size : in_length;
}

static uint64_t conv_out_length(uint64_t in_length, uint64_t window_size, uint64_t padding_size, uint64_t stride, enum padding pad_type)
{
    if(same) {
        return (int)(((my_float_t)in_length + 2 * padding_size - window_size) / stride) + 1;
    }
    else {
        uint64_t tmp = (in_length - window_size + 1);
        if(tmp % stride == 0) {
            return tmp / stride;
        }
        else {
            return tmp / stride + 1;
        }
        // return (uint64_t)ceil((my_float_t)(in_length - window_size + 1) / stride);
    }
    // return pad_type == same ? (int)(((my_float_t)in_length + 2 * padding_size - window_size) / stride) + 1 : (uint64_t)ceil((my_float_t)(in_length - window_size + 1) / stride);
}

static uint64_t conv_out_dim(uint64_t in_width, uint64_t in_height, uint64_t window_width,  uint64_t window_height, uint64_t w_padding, uint64_t h_padding, uint64_t w_stride, uint64_t h_stride, enum padding pad_type)
{
    return conv_out_length(in_width, window_width, w_padding, w_stride, pad_type) * conv_out_length(in_height, window_height, h_padding, h_stride, pad_type);
}

void conv_copy_and_pad_input(convolutional_layer *entry, unsigned int hart_id, input_struct *input)
{   
    // // set_gemm_core_sel_cmd(1);

    if (entry->pad_type_ == same)
    {
        send_bn_mul_data(1, 0);
        send_bn_add_data(0, 0);
        // set_length_cmd(remain_len);
        set_dram_read_input_cmd();
        reset_relu_cmd();
        set_num_lans_cmd(0);
        index3d in_ = entry->in_;
        index3d in_padded_ = entry->in_padded_;
        index3d padding_ = entry->padding_;
        my_float_t *in = input->in_ptr_;
        my_float_t *dst = entry->base.padded_ptr;
        
        uint64_t total_size = in_.depth_ * in_.height_;
        uint64_t blocksize = compute_block_size(total_size);
        uint64_t start = (blocksize) * hart_id;
        uint64_t end = min((blocksize) * (hart_id+1), total_size);
        
        // my_float_t *pimg = &dst[get_index(&in_padded_, padding_.width_, padding_.height_ + 0, 0)];
        //     // const 
        // my_float_t *pin = &in[get_index(&in_, 0, 0, 0)];
        for (uint64_t i = start; i < end; i++)
        {
            uint64_t c = i / in_.height_;
            uint64_t y = i % in_.height_;
            my_float_t *pimg = &dst[get_index(&in_padded_, padding_.width_, padding_.height_ + y, c)];
            // const 
            my_float_t *pin = &in[get_index(&in_, 0, y, c)];

            // for (uint64_t x = 0; x < in_.width_; x++)
            // {
            //     // pimg[x] = pin[x];
            //     my_float_t tmp = read_dram_value_cmd(&pin[x]);
            //     write_dram_value_cmd(&pimg[x], tmp);
            // }
            // send_bn_mul_data(1, 0);
            // send_bn_add_data(0, 0);
            int max_len = 100;
            for (uint64_t j = 0; j < in_.width_; j += max_len) {
                int remain_len = min(max_len, in_.width_ - j);
                /*
                * send data
                */
                reset_sram_offset_cmd();
                set_length_cmd(remain_len);
                // set_dram_read_input_cmd();
                uint32_t tmp_s;
                memcpy(&tmp_s, &pin, sizeof(tmp_s));
                set_addr_cmd(tmp_s);
                trigger_dram_read_cmd();
                // wait_idle_cmd();
                wait_idle_quick_cmd();
                /*
                * start BatchNorm
                */
                set_mode_cmd(1, remain_len);
                // reset_relu_cmd();
                trigger_add_cmd();
                // wait_idle_cmd();
                wait_idle_quick_cmd();
                set_mode_cmd(0, 0);
 
                /*
                * write data
                */
                set_dram_write_lens_cmd(remain_len);
                // set_num_lans_cmd(0);
                set_output_recv_cnt_cmd(0);
                memcpy(&tmp_s, &pimg, sizeof(tmp_s));
                set_dram_write_addr_cmd(0, tmp_s);
                set_dram_w_tr_cmd();
                wait_idle_cmd();

                pin += remain_len;
                pimg += remain_len; 
            }
            
        }
    }
}

void conv_copy_and_pad_input_old(convolutional_layer *entry, unsigned int hart_id, input_struct *input)
{
    if (entry->pad_type_ == same)
    {
        index3d in_ = entry->in_;
        index3d in_padded_ = entry->in_padded_;
        index3d padding_ = entry->padding_;
        my_float_t *in = input->in_ptr_;
        my_float_t *dst = entry->base.padded_ptr;
        
        uint64_t total_size = in_.depth_ * in_.height_;
        uint64_t blocksize = compute_block_size(total_size);
        uint64_t start = (blocksize) * hart_id;
        uint64_t end = min((blocksize) * (hart_id+1), total_size);
    
        for (uint64_t i = start; i < end; i++)
        {
            uint64_t c = i / in_.height_;
            uint64_t y = i % in_.height_;
            my_float_t *pimg = &dst[get_index(&in_padded_, padding_.width_, padding_.height_ + y, c)];
            const my_float_t *pin = &in[get_index(&in_, 0, y, c)];

            for (uint64_t x = 0; x < in_.width_; x++)
            {
                // pimg[x] = pin[x];
                my_float_t tmp = read_dram_value_cmd(&pin[x]);
                write_dram_value_cmd(&pimg[x], tmp);
            }
        }
    }
}

void conv_copy_and_pad_input_sw(convolutional_layer *entry, unsigned int hart_id, input_struct *input)
{
    if (entry->pad_type_ == same)
    {
        index3d in_ = entry->in_;
        index3d in_padded_ = entry->in_padded_;
        index3d padding_ = entry->padding_;
        my_float_t *in = input->in_ptr_;
        my_float_t *dst = entry->base.padded_ptr;
        
        uint64_t total_size = in_.depth_ * in_.height_;
        uint64_t blocksize = compute_block_size(total_size);
        uint64_t start = (blocksize) * hart_id;
        uint64_t end = min((blocksize) * (hart_id+1), total_size);
    
        for (uint64_t i = start; i < end; i++)
        {
            uint64_t c = i / in_.height_;
            uint64_t y = i % in_.height_;
            my_float_t *pimg = &dst[get_index(&in_padded_, padding_.width_, padding_.height_ + y, c)];
            const my_float_t *pin = &in[get_index(&in_, 0, y, c)];

            for (uint64_t x = 0; x < in_.width_; x++)
            {
                pimg[x] = pin[x];
                // my_float_t tmp = read_dram_value_cmd(&pin[x]);
                // write_dram_value_cmd(&pimg[x], tmp);
            }
        }
    }
}

void convolutional_layer_forward_propagation(struct list_node *ptr, unsigned int hart_id, input_struct *input)
{

#ifdef USING_GEM5
    clock_t  tick, ticks_per_msec = 2;
    clock_t  preprocess_time = 0;
    clock_t  send_data_time = 0, hardware_compute_time = 0;
    clock_t  send_weight_time = 0, send_idx_time = 0;
    clock_t  padding_time = 0, store_data_time = 0;
    clock_t  tmp_tick = clock();
    tick = clock();
#endif

    convolutional_layer *entry = get_convolutional_layer_entry(ptr);

    static int conv_cnt = 0;
    set_gemm_core_sel_cmd(1);
    
    if (input->in_size_ != entry->base.in_size_)
    {
        printf("Error input size not match %d/%d\n", input->in_size_, entry->base.in_size_);
        exit(-1);
    }

    // malloc input space
    entry->base.padded_ptr = (my_float_t *)malloc(entry->base.padding_size * sizeof(my_float_t));
    if (entry->base.padded_ptr != NULL) {
        if(entry->weight_.width_ != 1) {
            reset_dram_value_cmd(entry->base.padded_ptr, entry->base.padding_size);
        }
    }
    else {
        printf("Error: Unable to allocate memory for layer->padded_ptr\n");
        exit(1);
    }

    // // malloc output space
    entry->base.a_ptr_ = (my_float_t *)malloc(entry->base.out_size_ * sizeof(my_float_t));
    if (entry->base.a_ptr_ != NULL) {
    }
    else {
        printf("Error: Unable to allocate memory for entry->base.a_ptr_\n");
        exit(1);
    }

    entry->base.out_ptr_ = entry->base.a_ptr_;

    tmp_tick = clock();
    if(entry->weight_.width_ != 1) {
        conv_copy_and_pad_input(entry, hart_id, input);
    }
    padding_time += (clock() - tmp_tick)/ticks_per_msec;

    my_float_t *in_ptr_prev = input->in_ptr_;

    my_float_t *in;
    if(entry->weight_.width_ != 1) {
        in = entry->base.padded_ptr;
    }
    else {
        in = input->in_ptr_;
    }

    my_float_t *a = entry->base.a_ptr_;
    my_float_t *W = weights + entry->base.weight_offset;
    my_float_t *b = entry->base._b;
    my_float_t *out = entry->base.out_ptr_;
    input->in_ptr_ = out;
    input->in_size_ = entry->base.out_size_;

    index3d in_ = entry->in_;
    index3d in_padded_ = entry->in_padded_;
    index3d padding_ = entry->padding_;
    index3d out_ = entry->out_;
    index3d weight_ = entry->weight_;
    uint64_t h_stride_ = entry->h_stride_;
    uint64_t w_stride_ = entry->w_stride_;

    uint64_t start = 0;
    uint64_t end = out_.depth_;

    uint32_t tmp[10];

    my_float_t *batchNorm_W = W + ( (out_.depth_) * (in_.depth_) * (weight_.height_) * (weight_.width_) );

    reset_cmd();

    int data_gbuff_size   = 32768;
    int weight_gbuff_size = 32768;
    int idx_gbuff_size    = 327680;
    /*
     * modify height_per_operation on each convolutional layer
     */
    /*
     * weight
     */
    int kernel_len = (in_.depth_) * (weight_.height_) * (weight_.width_);
    int weight_num;
    /*
     *    I = Input Size
     *    K = Kernel Size
     *    P = Padding
     *    S = Stride
     *    O = Output Size
     * Then:
     *    O = (I - K + 2P) / S + 1
     */
    int h_per_op_hw;
    int input_num, out_h_per_op_hw, out_w_per_op_hw;

    // static int h_per_op_hw_arr[53] = {7, 1, 3, 1, //0~3
    //                                   1, 1, 3, 1, //4~7
    //                                   1, 3, 1, 1, //8~11
    //                                   4, 1, 2, 2, //12~15
    //                                   7, 1, 2, 7, //16~19
    //                                   1, 2, 7, 1, //20~23
    //                                   2, 4, 2, 2, //24~27
    //                                   2, 8, 2, 2, //28~31
    //                                   8, 2, 2, 8, //32~35
    //                                   2, 2, 8, 2, //36~39
    //                                   2, 8, 2, 2, //40~43
    //                                   4, 7, 2, 2, //44~47
    //                                   7, 7, 2, 7, //48~51
    //                                   7};         //52


    // static int input_num_arr[53] = {112, 56,  56,  56, //0~3
    //                                  56, 56,  56,  56, //4~7
    //                                  56, 56,  56,  56, //8~11
    //                                  28, 28,  28,  56, //12~15
    //                                 140, 28,  56, 140, //16~19
    //                                  28, 56, 140,  28, //20~23
    //                                  56, 14,  28,  14, //24~27
    //                                  28, 84,  28,  28, //28~31
    //                                  84, 28,  28,  84, //32~35
    //                                  28, 28,  84,  28, //36~39
    //                                  28, 84,  28,  28, //40~43
    //                                   7, 49,   7,  14, //44~47
    //                                  35, 49,  14,  35, //48~51
    //                                  49};              //52

    // static int weight_num_arr[53] = {  4,  64,  56,  73, //0~3
    //                                   73,  64,  56,  73, //4~7
    //                                   64,  56,  73,  73, //8~11
    //                                   28, 146, 128,  64, //12~15
    //                                   28, 146,  64,  28, //16~19
    //                                  146,  64,  28, 146, //20~23
    //                                   64,  14, 128,  64, //24~27
    //                                   32,  14, 128,  32, //28~31
    //                                   14, 128,  32,  14, //32~35
    //                                  128,  32,  14, 128, //36~39
    //                                   32,  14, 128,  32, //40~43
    //                                    7,  64,  32,  16, //44~47
    //                                    7,  64,  16,   7, //48~51
    //                                   64};               //52

    static int h_per_op_hw_arr[53] = {7, 1, 3, 1, //0~3
                                      1, 1, 3, 1, //4~7
                                      1, 3, 1, 1, //8~11
                                      4, 1, 2, 2, //12~15
                                      7, 1, 2, 7, //16~19
                                      1, 2, 7, 1, //20~23
                                      2, 4, 2, 2, //24~27
                                      2, 8, 2, 2, //28~31
                                      8, 2, 2, 8, //32~35
                                      2, 2, 8, 2, //36~39
                                      2, 7, 2, 2, //40~43
                                      4, 7, 2, 2, //44~47
                                      7, 7, 2, 6, //48~51
                                      7};         //52


    static int input_num_arr[53] = {112, 56,  56,  56, //0~3
                                     56, 56,  56,  56, //4~7
                                     56, 56,  56,  56, //8~11
                                     28, 28,  28,  56, //12~15
                                    140, 28,  56, 140, //16~19
                                     28, 56, 140,  28, //20~23
                                     56, 14,  28,  14, //24~27
                                     28, 84,  28,  28, //28~31
                                     84, 28,  28,  84, //32~35
                                     28, 28,  84,  28, //36~39
                                     28, 84,  28,  28, //40~43
                                      7, 49,   7,  14, //44~47
                                     35, 49,  14,  35, //48~51
                                     49};              //52

    static int weight_num_arr[53] = {  4,  64,  56,  73, //0~3
                                      73,  64,  56,  73, //4~7
                                      64,  56,  73,  73, //8~11
                                      28, 146, 128,  64, //12~15
                                      28, 146,  64,  28, //16~19
                                     146,  64,  28, 146, //20~23
                                      64,  14, 128,  64, //24~27
                                      32,  14, 128,  32, //28~31
                                      14, 128,  32,  14, //32~35
                                     128,  32,  14, 128, //36~39
                                      32,  14, 128,  32, //40~43
                                       7,  64,  32,  16, //44~47
                                       7,  64,  16,   7, //48~51
                                      64};               //52

    h_per_op_hw = h_per_op_hw_arr[conv_cnt];
    input_num = input_num = out_.width_ * ((h_per_op_hw - weight_.height_) / h_stride_ + 1);//input_num_arr[conv_cnt];
    weight_num = weight_num_arr[conv_cnt];

    out_w_per_op_hw = out_.width_;
    out_h_per_op_hw = ((h_per_op_hw - weight_.height_) / h_stride_ + 1);
    int size_per_channel_hw = h_per_op_hw * in_padded_.width_;

    printf("input  shape: (%d, %d, %d)\n", (int)in_.width_, (int)in_.height_, (int)in_.depth_);
    printf("output shape: (%d, %d, %d)\n", (int)out_.width_, (int)out_.height_, (int)out_.depth_);
    // printf("weight shape: (%d, %d, %d)\n", (int)weight_.width_, (int)weight_.height_, (int)in_.depth_);
    printf("input  sram usage: %d\n", (int)size_per_channel_hw * (int)in_.depth_);
    printf("weight sram usage: %d\n", (int)weight_.height_ * (int)weight_.width_ * (int)in_.depth_ * (int)weight_num);
    printf("each HW conv shape: %d %d %d\n", input_num, weight_num, kernel_len);
    printf("result sram %d\n", (int)input_num*(int)weight_num);

    set_KMN_cmd(kernel_len, input_num, weight_num);
    set_conv_cmd(kernel_len);
    set_idx_cmd(0, 0, 0, 0);
    set_col_idx_cmd(0, kernel_len, kernel_len*2, kernel_len*3);

    reset_preload_cmd();

    // set_gemm_core_sel_cmd(1);

    /*
     * idx ram setting
     */
    tmp_tick = clock();
    set_gemm_core_sel_cmd(3);
    int index_offset = 0;

    for(uint64_t compute_ch = 0; compute_ch < in_.depth_; compute_ch++) {
        int ch_offset = (h_per_op_hw * compute_ch) * in_padded_.width_;
        for(uint64_t wy = 0; wy < weight_.height_; wy++) {
            for(uint64_t wx = 0; wx < weight_.width_; wx++) {
                int index_value = ch_offset + (wy * in_padded_.width_ + wx);
                send_idx_cmd(index_value, index_offset++);
            }
        }
    }
    // set_gemm_core_sel_cmd(1);
    // set offset
    int offset_sel = 1;
    int offset_index = 0;
    for(uint64_t y = 0; y < out_h_per_op_hw; y++) {
        for(uint64_t x = 0; x < out_w_per_op_hw; x++) {
            int input_offset = (y * h_stride_) * in_padded_.width_ + x * w_stride_;
            if(offset_sel == 1) {
                send_offset_1_cmd(input_offset, offset_index);
            }
            else if(offset_sel == 2) {
                send_offset_2_cmd(input_offset, offset_index);
            }
            else if(offset_sel == 3) {
                send_offset_3_cmd(input_offset, offset_index);
            }
            else if(offset_sel == 4) {
                send_offset_4_cmd(input_offset, offset_index);
            }
            offset_sel++;
            if(offset_sel == 5) {
                offset_sel = 1;
                offset_index++;
            }
        }
    }
    set_gemm_core_sel_cmd(1);
    send_idx_time += (clock() - tmp_tick)/ticks_per_msec;
    /*
     * end
     */
    // set_gemm_core_sel_cmd(1);

    if(conv_cnt == 3  || conv_cnt == 4  || conv_cnt == 7  || 
        conv_cnt == 10 || conv_cnt == 13 || conv_cnt == 14 || 
        conv_cnt == 17 || conv_cnt == 20 || conv_cnt == 23 || 
        conv_cnt == 26 || conv_cnt == 27 || conv_cnt == 30 || 
        conv_cnt == 33 || conv_cnt == 36 || conv_cnt == 39 || 
        conv_cnt == 42 || conv_cnt == 45 || conv_cnt == 46 || 
        conv_cnt == 49 || conv_cnt == 52) {
        reset_relu_cmd();
    }
    else {
        set_relu_cmd();
    }

    preprocess_time = (clock() - tick)/ticks_per_msec;

    if(conv_cnt == 0)
    for (uint64_t o = start; o < end; o += weight_num)
    {
        printf("[%s]: %d/%d\n", entry->base.layer_name_, (int)(o + 1), (int)end);
        
        int remain_oc = min(weight_num, end - o);

        /*
         * GeMM HW READ WEIGHT FROM DRAM
         * READ (weight_.height_ * weight_.width_ * in_.depth_) 
         * PER OP
         */
        tmp_tick = clock();
        // set_gemm_core_sel_cmd(1);
        set_dram_read_weight_cmd();
        reset_sram_offset_cmd();
        set_length_cmd(weight_.height_ * weight_.width_ * in_.depth_);

        for(int curr_oc = o; curr_oc < o + remain_oc; curr_oc++) {
            int offset_w = (weight_.height_ * (in_.depth_ * curr_oc)) * weight_.width_;
            const my_float_t * ppw = W + offset_w;
            uint32_t tmp_s;
            memcpy(&tmp_s, &ppw, sizeof(tmp_s));
            set_addr_cmd(tmp_s);
            trigger_dram_read_cmd();
            wait_idle_quick_cmd();
        }
        
        // send batchNorm weight
        int batchNormIndex = 0;
        for(int curr_oc = o; curr_oc < o + remain_oc; curr_oc += 4) {
            set_mul_sram_0_cmd(batchNorm_W[curr_oc    ], batchNormIndex);
            set_mul_sram_1_cmd(batchNorm_W[curr_oc + 1], batchNormIndex);
            set_mul_sram_2_cmd(batchNorm_W[curr_oc + 2], batchNormIndex);
            set_mul_sram_3_cmd(batchNorm_W[curr_oc + 3], batchNormIndex);

            set_add_sram_0_cmd(batchNorm_W[out_.depth_ + curr_oc    ], batchNormIndex);
            set_add_sram_1_cmd(batchNorm_W[out_.depth_ + curr_oc + 1], batchNormIndex);
            set_add_sram_2_cmd(batchNorm_W[out_.depth_ + curr_oc + 2], batchNormIndex);
            set_add_sram_3_cmd(batchNorm_W[out_.depth_ + curr_oc + 3], batchNormIndex);
            batchNormIndex ++;
            __asm__ volatile ("nop");
        }

        send_weight_time += (clock() - tmp_tick)/ticks_per_msec;
        
        
        // 3 4 7 10 13 14 17 20 23 26 27 30 33 36 39 42 45 46
        // 49 52
        __asm__ volatile ("nop");
        // set_gemm_core_sel_cmd(1);
        int output_offset = 0;
        /*
         * configuration for read input
         */
        // set_gemm_core_sel_cmd(1);
        set_length_cmd(size_per_channel_hw);
        set_dram_read_input_cmd();

        // set_gemm_core_sel_cmd(1);

        int num_lans_get = 1;
        int conv_0_offset = 0;

        wait_idle_quick_cmd();
        reset_lans_cmd();
        set_lans_idx_cmd();
        
        for(uint64_t h = 0; h < out_.height_; h += out_h_per_op_hw) {
            int send_data_offset = 0;
            int remain_num = out_.width_ * min(out_h_per_op_hw, out_.height_ - h);

            int gemm_core_2_en = 0;
            int remain_num_2;
            int h_2;

            // if(h + out_h_per_op_hw < out_.height_ && conv_cnt != 0) {
            //     gemm_core_2_en = 1;
            //     // gemm_core_2_en = 0;
            //     h_2 = h + out_h_per_op_hw;
            //     remain_num_2 = out_.width_ * min(out_h_per_op_hw, out_.height_ - h_2);
                
            // }
            
            /*
             * HW SEND INPUT 1
             */
            tmp_tick = clock();
            // set_gemm_core_sel_cmd(1);
            uint64_t curr_h = h * h_stride_;
            reset_sram_offset_cmd();
            for(uint64_t inc = 0; inc < in_.depth_; inc++) {
                int offset_i = (in_padded_.width_ * in_padded_.height_ * inc) +
                                (in_padded_.width_ * (curr_h));
                my_float_t *pi = in + offset_i;
                uint32_t tmp_s;
                memcpy(&tmp_s, &pi, sizeof(tmp_s));
                set_addr_cmd(tmp_s);
                trigger_dram_read_cmd();
                wait_idle_quick_cmd();
            }
            // }
            // printf("send input: %d\n", (clock() - tmp_tick)/ticks_per_msec);
            send_data_time += (clock() - tmp_tick)/ticks_per_msec;

            tmp_tick = clock();
            trigger_conv_cmd();

            /*
             * READ INPUT 2
             */
            // if(gemm_core_2_en) {
            //     h += out_h_per_op_hw;
            //     // set_gemm_core_sel_cmd(2);
            //     uint64_t curr_h = h_2 * h_stride_;
            //     reset_sram_offset_cmd();
            //     for(uint64_t inc = 0; inc < in_.depth_; inc++) {
            //         int offset_i = (in_padded_.width_ * in_padded_.height_ * inc) +
            //                         (in_padded_.width_ * (curr_h));
            //         my_float_t *pi = in + offset_i;
            //         uint32_t tmp_s;
            //         memcpy(&tmp_s, &pi, sizeof(tmp_s));
            //         set_addr_cmd(tmp_s);
            //         trigger_dram_read_cmd();
            //         wait_idle_2_quick_cmd();
            //     }
            //     trigger_conv_cmd();
            // }

            wait_idle_quick_cmd();

            num_lans_get++;
            
            // set_gemm_core_sel_cmd(1);
            set_sram_next_cmd();
            set_lans_idx_cmd();
            
            if(num_lans_get == 3) {
                // set_gemm_core_sel_cmd(1);
                set_mode_cmd(3, 0);
                reset_pooling_idx_cmd();
                __asm__ volatile ("nop");
                set_pooling_start_cmd();
                wait_idle_quick_cmd();
                num_lans_get = 1;
                set_mode_cmd(0, 0);
                set_lans_idx_cmd();
                // printf("%f %f %f %f  ", (float_t)read_data_cmd(0, 0), 
                //                         (float_t)read_data_cmd(1, 0), 
                //                         (float_t)read_data_cmd(2, 0), 
                //                         (float_t)read_data_cmd(3, 0));

                // printf("%f %f %f %f  ", (float_t)read_data_cmd(4, 0), 
                //                         (float_t)read_data_cmd(5, 0), 
                //                         (float_t)read_data_cmd(6, 0), 
                //                         (float_t)read_data_cmd(7, 0));

                // printf("%f %f %f %f  ", (float_t)read_data_cmd(8, 0), 
                //                         (float_t)read_data_cmd(9, 0), 
                //                         (float_t)read_data_cmd(10, 0), 
                //                         (float_t)read_data_cmd(11, 0));

                // printf("%f %f %f %f\n", (float_t)read_data_cmd(12, 0), 
                //                         (float_t)read_data_cmd(13, 0), 
                //                         (float_t)read_data_cmd(14, 0), 
                //                         (float_t)read_data_cmd(15, 0));
                set_dram_write_lens_cmd(56);
                for(int s = 0; s < 4; s++) {
                    my_float_t *pa = &a[(o + s) * 3136];

                    set_num_lans_cmd(s%4);
                    
                    set_output_recv_cnt_cmd(0);
                    uint32_t tmp_s;
                    pa = pa + conv_0_offset;
                    memcpy(&tmp_s, &pa, sizeof(tmp_s));
                    set_dram_write_addr_cmd(s%4, tmp_s);
                    set_dram_w_tr_cmd();
                    wait_idle_quick_cmd();
                }
                conv_0_offset += 56;
            }

            /*
             * HW WRITE RESULT 1
             */
            tmp_tick = clock();
            // if(conv_cnt != 0)
            // {
            //     // set_gemm_core_sel_cmd(1);
            //     set_dram_write_lens_cmd(remain_num);
            //     for(int s = 0; s < remain_oc; s++) {
            //         my_float_t *pa = &a[get_index(&out_, 0, 0, o + s)];
            //         set_num_lans_cmd(s%4);
                    
            //         int base_idx = (s/4) * (input_num/4 + ((input_num%4) != 0));
            //         set_output_recv_cnt_cmd(base_idx);
            //         uint32_t tmp_s;
            //         pa = pa + output_offset;
            //         memcpy(&tmp_s, &pa, sizeof(tmp_s));
            //         set_dram_write_addr_cmd(s%4, tmp_s);

            //         set_dram_w_tr_cmd();
            //         wait_idle_quick_cmd();
            //     }
            // }
            // else {
            // }
            wait_idle_cmd();
            output_offset += input_num;

            /*
             * HW WRITE RESULT 2
             */
            tmp_tick = clock();
            // if(conv_cnt != 0 && gemm_core_2_en)
            // {
            //     wait_idle_2_quick_cmd();
            //     // set_gemm_core_sel_cmd(2);
            //     set_dram_write_lens_cmd(remain_num_2);
            //     for(int s = 0; s < remain_oc; s++) {
            //         my_float_t *pa = &a[get_index(&out_, 0, 0, o + s)];
            //         set_num_lans_cmd(s%4);
                    
            //         int base_idx = (s/4) * (input_num/4 + ((input_num%4) != 0));
            //         set_output_recv_cnt_cmd(base_idx);
            //         uint32_t tmp_s;
            //         pa = pa + output_offset;
            //         memcpy(&tmp_s, &pa, sizeof(tmp_s));
            //         set_dram_write_addr_cmd(s%4, tmp_s);

            //         set_dram_w_tr_cmd();
            //         wait_idle_2_quick_cmd();
            //     }
            //     output_offset += input_num;
            //     // set_gemm_core_sel_cmd(1);
            // }
            // wait_idle_cmd();
        }
    }
    else
    for (uint64_t o = start; o < end; o += weight_num)
    {
        printf("[%s]: %d/%d\n", entry->base.layer_name_, (int)(o + 1), (int)end);
        
        int remain_oc = min(weight_num, end - o);
        int remain_oc_2;
        int o_2;
        
        int second_core_en = 0;
        if(o + weight_num < end) {
            second_core_en = 1;
            // second_core_en = 0;
            o_2 = o + weight_num;
            remain_oc_2 = min(weight_num, end - o_2);
        }

        /*
         * HW READ WEIGHT FROM DRAM
         */
        tmp_tick = clock();
        set_gemm_core_sel_cmd(1);
        set_dram_read_weight_cmd();
        reset_sram_offset_cmd();
        set_length_cmd(weight_.height_ * weight_.width_ * in_.depth_);

        set_read_rounds_cmd(remain_oc);
        set_read_offset_cmd(weight_.height_ * in_.depth_ * weight_.width_* 2);
        int offset_w = (weight_.height_ * (in_.depth_ * o)) * weight_.width_;
        const my_float_t * ppw = W + offset_w;
        uint32_t tmp_s;
        memcpy(&tmp_s, &ppw, sizeof(tmp_s));
        set_addr_cmd(tmp_s);
        trigger_dram_read_cmd();
        wait_idle_quick_cmd();
        
        // send batchNorm weight
        int batchNormIndex = 0;
        for(int curr_oc = o; curr_oc < o + remain_oc; curr_oc += 4) {
            set_mul_sram_0_cmd(batchNorm_W[curr_oc    ], batchNormIndex);
            set_mul_sram_1_cmd(batchNorm_W[curr_oc + 1], batchNormIndex);
            set_mul_sram_2_cmd(batchNorm_W[curr_oc + 2], batchNormIndex);
            set_mul_sram_3_cmd(batchNorm_W[curr_oc + 3], batchNormIndex);

            set_add_sram_0_cmd(batchNorm_W[out_.depth_ + curr_oc    ], batchNormIndex);
            set_add_sram_1_cmd(batchNorm_W[out_.depth_ + curr_oc + 1], batchNormIndex);
            set_add_sram_2_cmd(batchNorm_W[out_.depth_ + curr_oc + 2], batchNormIndex);
            set_add_sram_3_cmd(batchNorm_W[out_.depth_ + curr_oc + 3], batchNormIndex);
            batchNormIndex ++;
            __asm__ volatile ("nop");
        }
        /*
         * SEND WEIGHT 2
         */
        if(second_core_en)
        {
            set_gemm_core_sel_cmd(2);
            set_dram_read_weight_cmd();
            reset_sram_offset_cmd();
            set_length_cmd(weight_.height_ * weight_.width_ * in_.depth_);

            set_read_rounds_cmd(remain_oc_2);
            set_read_offset_cmd(weight_.height_ * in_.depth_ * weight_.width_* 2);
            int offset_w = (weight_.height_ * (in_.depth_ * o_2)) * weight_.width_;
            const my_float_t * ppw = W + offset_w;
            uint32_t tmp_s;
            memcpy(&tmp_s, &ppw, sizeof(tmp_s));
            set_addr_cmd(tmp_s);
            trigger_dram_read_cmd();
            wait_idle_2_quick_cmd();
            
            // send batchNorm weight
            batchNormIndex = 0;
            for(int curr_oc = o_2; curr_oc < o_2 + remain_oc_2; curr_oc += 4) {
                set_mul_sram_0_cmd(batchNorm_W[curr_oc    ], batchNormIndex);
                set_mul_sram_1_cmd(batchNorm_W[curr_oc + 1], batchNormIndex);
                set_mul_sram_2_cmd(batchNorm_W[curr_oc + 2], batchNormIndex);
                set_mul_sram_3_cmd(batchNorm_W[curr_oc + 3], batchNormIndex);

                set_add_sram_0_cmd(batchNorm_W[out_.depth_ + curr_oc    ], batchNormIndex);
                set_add_sram_1_cmd(batchNorm_W[out_.depth_ + curr_oc + 1], batchNormIndex);
                set_add_sram_2_cmd(batchNorm_W[out_.depth_ + curr_oc + 2], batchNormIndex);
                set_add_sram_3_cmd(batchNorm_W[out_.depth_ + curr_oc + 3], batchNormIndex);
                batchNormIndex ++;
                __asm__ volatile ("nop");
            }
        }
        send_weight_time += (clock() - tmp_tick)/ticks_per_msec;
        
        
        // 3 4 7 10 13 14 17 20 23 26 27 30 33 36 39 42 45 46
        // 49 52
        __asm__ volatile ("nop");
        int output_offset = 0;
        /*
         * configuration for read input
         */
        set_gemm_core_sel_cmd(3);
        set_length_cmd(size_per_channel_hw);
        set_dram_read_input_cmd();
        set_gemm_core_sel_cmd(1);

        for(uint64_t h = 0; h < out_.height_; h += out_h_per_op_hw) {
            int send_data_offset = 0;
            int remain_num = out_.width_ * min(out_h_per_op_hw, out_.height_ - h);
            
            /*
             * HW SEND INPUT 1
             */
            tmp_tick = clock();
            set_gemm_core_sel_cmd(3);
            uint64_t curr_h = h * h_stride_;
            reset_sram_offset_cmd();
            set_read_rounds_cmd(in_.depth_);
            set_read_offset_cmd(in_padded_.width_ * in_padded_.height_*2);
            int offset_i = (in_padded_.width_ * in_padded_.height_ * 0) +
                            (in_padded_.width_ * (curr_h));
            my_float_t *pi = in + offset_i;
            uint32_t tmp_s;
            memcpy(&tmp_s, &pi, sizeof(tmp_s));
            set_addr_cmd(tmp_s);
            trigger_dram_read_cmd();
            wait_idle_quick_cmd();
            wait_idle_2_quick_cmd();
            set_read_rounds_cmd(1);
            send_data_time += (clock() - tmp_tick)/ticks_per_msec;

            trigger_conv_cmd();

            wait_idle_quick_cmd();

            /*
             * HW WRITE RESULT 1
             */
            tmp_tick = clock();
            set_gemm_core_sel_cmd(1);
            set_dram_write_lens_cmd(remain_num);
            for(int s = 0; s < remain_oc; s++) {
                my_float_t *pa = &a[get_index(&out_, 0, 0, o + s)];
                set_num_lans_cmd(s%4);
                
                int base_idx = (s/4) * (input_num/4 + ((input_num%4) != 0));
                set_output_recv_cnt_cmd(base_idx);
                uint32_t tmp_s;
                pa = pa + output_offset;
                memcpy(&tmp_s, &pa, sizeof(tmp_s));
                set_dram_write_addr_cmd(s%4, tmp_s);

                set_dram_w_tr_cmd();
                wait_idle_quick_cmd();
            }
            
            wait_idle_cmd();
            // output_offset += input_num;

            /*
             * HW WRITE RESULT 2
             */
            tmp_tick = clock();
            if(/*gemm_core_2_en*/second_core_en)
            {
                wait_idle_2_quick_cmd();
                set_gemm_core_sel_cmd(2);
                set_dram_write_lens_cmd(remain_num);
                for(int s = 0; s < remain_oc_2; s++) {
                    my_float_t *pa = &a[get_index(&out_, 0, 0, o_2 + s)];
                    set_num_lans_cmd(s%4);
                    
                    int base_idx = (s/4) * (input_num/4 + ((input_num%4) != 0));
                    set_output_recv_cnt_cmd(base_idx);
                    uint32_t tmp_s;
                    pa = pa + output_offset;
                    memcpy(&tmp_s, &pa, sizeof(tmp_s));
                    set_dram_write_addr_cmd(s%4, tmp_s);

                    set_dram_w_tr_cmd();
                    wait_idle_2_quick_cmd();
                }
                
                set_gemm_core_sel_cmd(1);
            }
            wait_idle_cmd();
            output_offset += input_num;
        }
        if(second_core_en) {
            o += weight_num;
        }
    }
    //###########################################################
    // print output tensor
    // if(conv_cnt == 0)
    // {
    //     printf("output:\n");
    //     printf("\n\n[%s]\n", entry->base.layer_name_);
    //     printf("shape depth:%d,  height:%d width:%d\n", (int)64, (int)56, (int)56);
    //     printf("    ");
    //     for(int i = 0; i < 56; i++) {
    //         printf("%6d ", (i));
    //     }
    //     printf("\n");
    //     int p_cnt = 0;
    //     for (int inc = 0; inc < 64; inc++) {
    //         printf("\n[depth%d]\n", inc);
    //         for (int h = 0; h < 56; h++) {
    //             printf("%3d ", (h));
    //             printf("[ ");
    //             for (uint64_t w = 0; w < 56; w++) {
    //                 // printf("%2.6f ", ((float)out[p_cnt]));
    //                 printf("%2.6f ", ((float_t)read_dram_value_cmd(&out[p_cnt])));
                    
    //                 p_cnt++;
    //             }
    //             printf("]\n");
    //         }
    //     }
    // }
    // if(conv_cnt == 0)
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
    //             for (uint64_t w = 0; w < out_.width_; w++) {
    //                 // printf("%2.6f ", ((float)out[p_cnt]));
    //                 printf("%2.6f ", ((float_t)read_dram_value_cmd(&out[p_cnt])));
                    
    //                 p_cnt++;
    //             }
    //             printf("]\n");
    //         }
    //     }
    // }
    //###########################################################

    if(entry->delete_input) {
        free(in_ptr_prev);
    }
    free(entry->base.padded_ptr);
    conv_cnt++;
    
    set_gemm_core_sel_cmd(1);
    set_read_rounds_cmd(1);

#ifdef PRINT_LAYER
    // printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, (float_t)out[0], (float_t)out[1], (float_t)out[entry->base.out_size_-2], (float_t)out[entry->base.out_size_-1]);
    printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, (float_t)read_dram_value_cmd(&out[0]), (float_t)read_dram_value_cmd(&out[1]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-2]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-1]));
#endif

#ifdef USING_GEM5
    tick = (clock() - tick)/ticks_per_msec;
    printf("It took %7ld msec to perform conv.\n\n", tick);
    printf("It took %7ld msec to perform on preprocess_time.\n", preprocess_time);
    printf("It took %7ld msec to perform on hardware_compute_time.\n", hardware_compute_time);
    printf("It took %7ld msec to perform on send_data_time.\n", send_data_time);
    printf("It took %7ld msec to perform on send_weight_time.\n", send_weight_time);
    printf("It took %7ld msec to perform on send_idx_time.\n", send_idx_time);
    printf("It took %7ld msec to perform on padding_time.\n", padding_time);
    printf("It took %7ld msec to perform on store_data_time.\n\n", store_data_time);
#endif
    // exit(1);
}

layer_base * new_convolutional_layer(
                                     cnn_controller *ctrl,
                                     my_float_t(*activate) (my_float_t *, uint64_t, uint64_t),
                                     uint64_t in_width,
                                     uint64_t in_height,
                                     uint64_t window_width,
                                     uint64_t window_height,
                                     uint64_t in_channels,
                                     uint64_t out_channels,
                                     enum padding  pad_type,
                                     uint8_t  has_bias,
                                     uint64_t w_stride,
                                     uint64_t h_stride,
                                     uint64_t w_padding,
                                     uint64_t h_padding,
                                     uint8_t delete_input
                                    )
{
    // printf("get conv parameter\n");
    // uint32_t tmp = in_width;
    // printf("in_width: %d\n", tmp);
    // printf("in_height: %u\n", in_height);
    convolutional_layer *ret = (convolutional_layer *)malloc(sizeof(convolutional_layer));

    if (pad_type == same)
        ctrl->padding_size = in_length(in_width, w_padding, pad_type) * in_length(in_height, h_padding, pad_type) * in_channels;
    else 
        ctrl->padding_size = 0;
        
    init_layer(&ret->base,
               ctrl,
               in_width*in_height*in_channels,
               conv_out_dim(in_width, in_height, window_width, window_height, w_padding, h_padding, w_stride, h_stride, pad_type) * out_channels, 
               window_width * window_height * in_channels * out_channels,
               has_bias ? out_channels : 0,
               activate==softmax
              );
#ifdef PRINT_LAYER
    static uint64_t call_time = 0;
    my_sprintf(ret->base.layer_name_, "conv%d", call_time++);
#endif
    ret->in_ = new_index3d(in_width, in_height, in_channels);
    ret->in_padded_ = new_index3d(in_length(in_width, w_padding, pad_type), in_length(in_height, h_padding, pad_type), in_channels);
    ret->out_ = new_index3d(conv_out_length(in_width, window_width, w_padding, w_stride, pad_type), conv_out_length(in_height, window_height, h_padding, h_stride, pad_type), out_channels);
    ret->weight_ = new_index3d(window_width, window_height, in_channels*out_channels);
    ret->padding_ = new_index3d(w_padding, h_padding, 0);
    ret->pad_type_ = pad_type;
    ret->w_stride_ = w_stride;
    ret->h_stride_ = h_stride;
    ret->has_bias_ = has_bias;
    
    if (pad_type == same)
    {
        ret->padding_done_flag = 0;
        ret->padding_mask = (ctrl->total_CPUs < 64) ? (1LL << ctrl->total_CPUs) - 1 : 0LL - 1;
    }

    ret->delete_input = delete_input;

    ret->base.activate = activate;
    ret->base.forward_propagation = convolutional_layer_forward_propagation;
    // printf("insize of average pooling layer %d\n", ret->base.in_size_);
    // printf("conv parameter:\n");
    // printf("out_.depth_: %d\n in_.depth_: %d\n out_.height_: %d\n out_.width_: %d\n weight_.height_: %d\n weight_.width_: %d\n", ret->out_.depth_, ret->in_.depth_, ret->out_.height_, ret->out_.width_,  ret->weight_.height_, ret->weight_.width_);

    printf("conv: W [%f, %f, ... , %f, %f]\n", ret->base._W[0], ret->base._W[1], ret->base._W[window_width * window_height * in_channels * out_channels-2], ret->base._W[window_width * window_height * in_channels * out_channels-1]);
    // printf("conv: in [%f, %f, ... , %f, %f]\n", ret->base.in_ptr_[0], ret->base.in_ptr_[1], ret->base.in_ptr_[ret->base.in_size_-2], ret->base.in_ptr_[ret->base.in_size_-1]);
    // printf("%d\n", window_width * window_height * in_channels * out_channels);
    // printf("avg pool: b  [%f, %f, ... , %f, %f]\n", ret->base._b[0], ret->base._b[1], ret->base._b[in_channels-2], ret->base._b[in_channels-1]);
    return &ret->base;
}

