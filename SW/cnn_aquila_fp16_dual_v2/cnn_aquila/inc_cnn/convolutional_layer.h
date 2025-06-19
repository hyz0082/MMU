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
    int w_stride_;
    int h_stride_;
    uint8_t has_bias_;

    int padding_done_flag;
    int padding_mask;

    uint8_t delete_input;
} convolutional_layer;

convolutional_layer * get_convolutional_layer_entry(struct list_node *ptr)
{
    return list_entry(ptr, convolutional_layer, base.list);
}

static int in_length(int in_length, int padding_size, enum padding pad_type)
{
    return pad_type == same ? in_length + 2 * padding_size : in_length;
}

static int conv_out_length(int in_length, int window_size, int padding_size, int stride, enum padding pad_type)
{
    if(same) {
        return (int)(((my_float_t)in_length + 2 * padding_size - window_size) / stride) + 1;
    }
    else {
        int tmp = (in_length - window_size + 1);
        if(tmp % stride == 0) {
            return tmp / stride;
        }
        else {
            return tmp / stride + 1;
        }
        // return (int)ceil((my_float_t)(in_length - window_size + 1) / stride);
    }
    // return pad_type == same ? (int)(((my_float_t)in_length + 2 * padding_size - window_size) / stride) + 1 : (int)ceil((my_float_t)(in_length - window_size + 1) / stride);
}

static int conv_out_dim(int in_width, int in_height, int window_width,  int window_height, int w_padding, int h_padding, int w_stride, int h_stride, enum padding pad_type)
{
    return conv_out_length(in_width, window_width, w_padding, w_stride, pad_type) * conv_out_length(in_height, window_height, h_padding, h_stride, pad_type);
}

void conv_copy_and_pad_input(convolutional_layer *entry, unsigned int hart_id, input_struct *input)
{   
    static int conv_copy_cnt = 0;

    if (entry->pad_type_ == same && conv_copy_cnt == 0)
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
        
        int total_size = in_.depth_ * in_.height_;
        int start = 0;
        int end = total_size;
        
        for (int i = start; i < end; i++)
        {
            int c = i / in_.height_;
            int y = i % in_.height_;
            my_float_t *pimg = &dst[get_index(&in_padded_, padding_.width_, padding_.height_ + y, c)]; 
            my_float_t *pin = &in[get_index(&in_, 0, y, c)];

            int max_len = 112;
            for (int j = 0; j < in_.width_; j += max_len) {
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
                 * move data from input sram to output sram
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
                // wait_idle_cmd();
                // 0.1s
                wait_idle_quick_cmd();

                pin += remain_len;
                pimg += remain_len; 
            }
            
        }
    }
    else {
        send_bn_mul_data(1, 0);
        send_bn_add_data(0, 0);
        set_dram_read_input_cmd();
        reset_relu_cmd();
        set_num_lans_cmd(0);
        index3d in_ = entry->in_;
        index3d in_padded_ = entry->in_padded_;
        index3d padding_ = entry->padding_;
        my_float_t *in = input->in_ptr_;
        my_float_t *dst = entry->base.padded_ptr;
        
        int total_size = in_.depth_ * in_.height_;
        int start = 0;
        int end = total_size;
        
        for (int i = start; i < end; i++)
        {
            int c = i / in_.height_;
            int y = i % in_.height_;
            my_float_t *pimg = &dst[get_index(&in_padded_, padding_.width_, padding_.height_ + y, c)]; 
            // my_float_t *pin = &in[get_index(&in_, 0, y, c)];
            my_float_t *pin = in + ((in_.height_ * c + y) * in_.width_);

            int remain_len = in_.width_;
            /*
             * send data
             */
            reset_sram_offset_cmd();
            set_length_cmd(remain_len);
            uint32_t tmp_s;
            memcpy(&tmp_s, &pin, sizeof(tmp_s));
            set_addr_cmd(tmp_s);
            trigger_dram_read_cmd();
            wait_idle_quick_cmd();
            /*
             * move data from input sram to output sram
             */
            set_mode_cmd(1, remain_len);
            trigger_add_cmd();
            wait_idle_quick_cmd();
            set_mode_cmd(0, 0);
            /*
            * write data
            */
            set_dram_write_lens_cmd(remain_len);
            set_output_recv_cnt_cmd(0);
            memcpy(&tmp_s, &pimg, sizeof(tmp_s));
            set_dram_write_addr_cmd(0, tmp_s);
            set_dram_w_tr_cmd();
            wait_idle_cmd();
            // 0.1s
            // wait_idle_quick_cmd();
        }
    }

    conv_copy_cnt++;
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
    clock_t  malloc_time = 0;
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
    // entry->base.padded_ptr = (my_float_t *)malloc(entry->base.padding_size * sizeof(my_float_t));
    // if (entry->base.padded_ptr != NULL) {
    //     if(entry->weight_.width_ != 1) {
    //         reset_dram_value_cmd(entry->base.padded_ptr, entry->base.padding_size);
    //     }
    // }
    // else {
    //     printf("Error: Unable to allocate memory for layer->padded_ptr\n");
    //     exit(1);
    // }

    // // malloc output space
    entry->base.a_ptr_ = (my_float_t *)malloc(entry->base.out_size_ * sizeof(my_float_t));
    if (entry->base.a_ptr_ != NULL) {
    }
    else {
        printf("Error: Unable to allocate memory for entry->base.a_ptr_\n");
        exit(1);
    }

    entry->base.out_ptr_ = entry->base.a_ptr_;
    malloc_time += (clock() - tmp_tick)/ticks_per_msec;

    tmp_tick = clock();
    // if(entry->weight_.width_ != 1) {
    //     conv_copy_and_pad_input(entry, hart_id, input);
    // }
    padding_time += (clock() - tmp_tick)/ticks_per_msec;

    my_float_t *in_ptr_prev = input->in_ptr_;

    my_float_t *in;
    // if(entry->weight_.width_ != 1) {
    //     in = entry->base.padded_ptr;
    // }
    // else {
        in = input->in_ptr_;
    // }

    my_float_t *a = entry->base.a_ptr_;
    my_float_t *W = weights + entry->base.weight_offset;
    my_float_t *b = entry->base._b;
    my_float_t *out = entry->base.out_ptr_;
    input->in_ptr_ = out;
    input->in_size_ = entry->base.out_size_;

    index3d in_ = entry->in_;
    index3d in_padded_ = entry->in_padded_;
    // index3d padding_ = entry->padding_;
    index3d out_ = entry->out_;
    index3d weight_ = entry->weight_;
    int h_stride_ = entry->h_stride_;
    int w_stride_ = entry->w_stride_;

    int start = 0;
    int end = out_.depth_;

    uint32_t tmp[10];

    my_float_t *batchNorm_W = W + ( (out_.depth_) * (in_.depth_) * (weight_.height_) * (weight_.width_) );

    reset_cmd();

    int data_gbuff_size   = 32768;
    int weight_gbuff_size = 8192;
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

    static int h_per_op_hw_arr[53] = {7, 2, 3, 1, //0~3
                                      1, 1, 3, 1, //4~7
                                      1, 3, 1, 1, //8~11
                                      3, 1, 2, 2, //12~15
                                      7, 1, 2, 7, //16~19
                                      1, 2, 7, 1, //20~23
                                      2, 3, 2, 2, //24~27
                                      2, 8, 2, 2, //28~31
                                      8, 2, 2, 8, //32~35
                                      2, 2, 8, 2, //36~39
                                      2, 7, 2, 2, //40~43
                                      3, 7, 2, 2, //44~47
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

    // static int weight_num_arr[53] = {  4,  64,  32,  64, //0~3
    //                                   64,  64,  32,  64, //4~7
    //                                   64,  32,  64,  64, //8~11
    //                                   16, 128, 128,  64, //12~15
    //                                   16, 128,  64,  16, //16~19
    //                                  128,  64,  16, 128, //20~23
    //                                   64,   8, 128,  64, //24~27
    //                                   32,   8, 128,  32, //28~31
    //                                    8, 128,  32,   8, //32~35
    //                                  128,  32,   8, 128, //36~39
    //                                   32,   8, 128,  32, //40~43
    //                                    2,  32,   4,   8, //44~47
    //                                    2,  32,   4,   2, //48~51
    //                                   64};               //52

    // static int weight_num_arr[53] = {  4,  32,  16, 128, //0~3
    //                                  128,  32,  16,  64, //4~7
    //                                   32,  16,  64,  64, //8~11
    //                                    8, 128,  64,  32, //12~15
    //                                    8, 128,  32,   8, //16~19
    //                                  128,  32,   8, 128, //20~23
    //                                   32,   4,  64,  32, //24~27
    //                                   16,   4,  64,  16, //28~31
    //                                    4,  64,  16,   4, //32~35
    //                                   64,  16,   4,  64, //36~39
    //                                   16,   4,  64,  16, //40~43
    //                                    4,  32,   16,   8, //44~47
    //                                    4,  32,   8,   4, //48~51
    //                                   32};               //52
    static int weight_num_arr[53] = {  4,  32,  16, 128, //0~3
                                     128,  32,  16,  64, //4~7
                                      32,  16,  64,  64, //8~11
                                       8, 128,  64,  32, //12~15
                                       8, 128,  32,   8, //16~19
                                     128,  32,   8, 128, //20~23
                                      32,   4,  64,  32, //24~27
                                      16,   4, 64,  16, //28~31
                                       4,  64,  16,   4, //32~35
                                      64,  16,   4,  64, //36~39
                                      16,   4,  64,  16, //40~43
                                       2,  32,   16,   8, //44~47
                                       2,  32,   8,   2, //48~51
                                      32};               //52
    
    
    h_per_op_hw = h_per_op_hw_arr[conv_cnt];

    input_num = input_num = out_.width_ * ((h_per_op_hw - weight_.height_) / h_stride_ + 1);
    weight_num = weight_num_arr[conv_cnt] / 2;

    out_w_per_op_hw = out_.width_;
    out_h_per_op_hw = ((h_per_op_hw - weight_.height_) / h_stride_ + 1);
    int size_per_channel_hw = h_per_op_hw * in_padded_.width_;

    printf("input  shape: (%d, %d, %d)\n", (int)in_.width_, (int)in_.height_, (int)in_.depth_);
    printf("output shape: (%d, %d, %d)\n", (int)out_.width_, (int)out_.height_, (int)out_.depth_);
    printf("weight shape: (%d, %d, %d)\n", (int)weight_.width_, (int)weight_.height_, (int)in_.depth_);
    printf("sizePerChannel   : %d\n", (int)size_per_channel_hw);
    printf("input  sram usage: %d\n", (int)size_per_channel_hw * (int)in_.depth_);
    printf("weight sram usage: %d\n", (int)weight_.height_ * (int)weight_.width_ * (int)in_.depth_ * (int)weight_num);
    printf("each HW conv shape: %d %d %d\n", input_num, weight_num, kernel_len);
    printf("result sram %d\n", (int)input_num*(int)weight_num);

    set_KMN_cmd(kernel_len, input_num, weight_num);
    set_conv_cmd(kernel_len);
    set_idx_cmd(0, 0, 0, 0);
    set_col_idx_cmd(0, kernel_len, kernel_len*2, kernel_len*3);

    reset_preload_cmd();

    // set reset sram
    int paddingSize = (in_padded_.width_ - in_.width_) / 2;
    set_gemm_core_sel_cmd(15);
    set_inputlength_cmd(in_.width_);
    set_paddingsize_cmd(paddingSize);
    set_gemm_core_sel_cmd(1);

    printf("padding size: %d\n", paddingSize);

    /*
     * Index Ram Setting
     */
    tmp_tick = clock();
    set_gemm_core_sel_cmd(15);
    int index_offset = 0;

    for(int compute_ch = 0; compute_ch < in_.depth_; compute_ch++) {
        int ch_offset = (h_per_op_hw * compute_ch) * in_padded_.width_;
        for(int wy = 0; wy < weight_.height_; wy++) {
            for(int wx = 0; wx < weight_.width_; wx++) {
                int index_value = ch_offset + (wy * in_padded_.width_ + wx);
                send_idx_cmd(index_value, index_offset++);
            }
        }
    }
    // set offset
    int offset_sel = 1;
    int offset_index = 0;
    for(int y = 0; y < out_h_per_op_hw; y++) {
        for(int x = 0; x < out_w_per_op_hw; x++) {
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

    if( conv_cnt == 3  || conv_cnt == 4  || conv_cnt == 7  || 
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
    for (int o = start; o < end; o += weight_num)
    {
        // printf("[%s]: %d/%d\n", entry->base.layer_name_, (int)(o + 1), (int)end);
        
        int remain_oc = min(weight_num, end - o);

        // read weight from dram
        tmp_tick = clock();
        
        int remain_oc_2;
        int o_2;
        
        int second_core_en = 0;
        if(o + weight_num < end) {
            second_core_en = 1;
            // second_core_en = 0;
            o_2 = o + weight_num;
            remain_oc_2 = min(weight_num, end - o_2);
        }

        // read weight from dram
        tmp_tick = clock();
        set_gemm_core_sel_cmd(1);
        int offset_w = (weight_.height_ * (in_.depth_ * o)) * weight_.width_;
        const my_float_t * ppw = W + offset_w;
        uint32_t tmp_s, tmp_s_2;
        memcpy(&tmp_s, &ppw, sizeof(tmp_s));
        read_conv_weight_cmd(weight_.height_ * weight_.width_ * in_.depth_, 
                             remain_oc, tmp_s);
        
        // read batchNorm weight from dram
        int batchNormIndex = 0;
        ppw = batchNorm_W + o;
        memcpy(&tmp_s, &ppw, sizeof(tmp_s));
        ppw = batchNorm_W + out_.depth_ + o;
        memcpy(&tmp_s_2, &ppw, sizeof(tmp_s));
        read_batchNorm_weight_cmd(remain_oc, tmp_s, tmp_s_2);
        set_dram_read_weight_cmd();
        
        // second GeMM read data
        if(second_core_en)
        {
            // read weight from dram
            set_gemm_core_sel_cmd(2);
            int offset_w = (weight_.height_ * (in_.depth_ * o_2)) * weight_.width_;
            const my_float_t * ppw = W + offset_w;
            uint32_t tmp_s, tmp_s_2;
            memcpy(&tmp_s, &ppw, sizeof(tmp_s));
            read_conv_weight_cmd(weight_.height_ * weight_.width_ * in_.depth_, 
                                remain_oc_2, tmp_s);
            
            // read batchNorm weight
            ppw = batchNorm_W + o_2;
            memcpy(&tmp_s, &ppw, sizeof(tmp_s));
            ppw = batchNorm_W + out_.depth_ + o_2;
            memcpy(&tmp_s_2, &ppw, sizeof(tmp_s));
            read_batchNorm_weight_cmd(remain_oc, tmp_s, tmp_s_2);
            set_dram_read_weight_cmd();
        }
        set_gemm_core_sel_cmd(1);
        send_weight_time += (clock() - tmp_tick)/ticks_per_msec;
        
        __asm__ volatile ("nop");
        int output_offset = 0;
        /*
         * configuration for read input
         */
        set_gemm_core_sel_cmd(15);
        // set_length_cmd(896);
        // set_input_height_cmd(4);
        // set_boundary_padding_size_cmd(696);
        // set_dram_read_input_cmd();
        set_gemm_core_sel_cmd(1);

        int num_lans_get = 1;
        int conv_0_offset = 0;

        wait_idle_quick_cmd();
        set_gemm_core_sel_cmd(15);
        reset_lans_cmd();
        set_lans_idx_cmd();
        wait_idle_quick_cmd();
        wait_idle_2_quick_cmd();

        int currHeight = -3;
        
        for(int h = 0; h < out_.height_; h += 1) {

            // int remain_num = 1;
            
            // read input
            tmp_tick = clock();
            set_gemm_core_sel_cmd(15);
            int curr_h;
            if(currHeight < 0) {
                curr_h = 0;
            }
            else {
                curr_h = currHeight;
            }

            int offset_i = (in_.width_ * (curr_h));
            my_float_t *pi = in + offset_i;
            uint32_t tmp_s;
            memcpy(&tmp_s, &pi, sizeof(tmp_s));

            if(h == 0) {
                set_currinputtype_cmd(0);
                reset_sram_offset_cmd();
                set_reset_sram_cmd();
                set_sram_offset_cmd(693); 

                set_length_cmd(4 * 224);
                set_input_height_cmd(4);
                set_boundary_padding_size_cmd(230*3 + 6);

            }
            else if(h == 1) {
                set_currinputtype_cmd(0);
                reset_sram_offset_cmd();
                set_reset_sram_cmd();
                set_sram_offset_cmd(233); 

                set_length_cmd(6 * 224);
                set_input_height_cmd(6);
                set_boundary_padding_size_cmd(230 + 6);

            }
            else if(h == 111) {
                set_currinputtype_cmd(2);
                reset_sram_offset_cmd();
                set_reset_sram_cmd();
                set_sram_offset_cmd(paddingSize); 

                set_length_cmd(5 * 224);
                set_input_height_cmd(5);
                set_boundary_padding_size_cmd(466);

            }
            else {
                set_currinputtype_cmd(1);
                set_sram_offset_cmd(paddingSize);

                set_length_cmd(7 * 224);
                set_input_height_cmd(7);
                set_boundary_padding_size_cmd(6);

            }
            currHeight += 2;
            read_conv_input_cmd(in_.depth_, in_.width_ * in_.height_*2,
                                tmp_s);

            send_data_time += (clock() - tmp_tick)/ticks_per_msec;

            tmp_tick = clock();
            trigger_conv_cmd();
            wait_idle_quick_cmd();

            num_lans_get++;
            
            set_sram_next_cmd();
            set_lans_idx_cmd();
            
            if(num_lans_get == 3) {
                set_gemm_core_sel_cmd(15);
                set_mode_cmd(3, 0);
                reset_pooling_idx_cmd();
                __asm__ volatile ("nop");
                set_pooling_start_cmd();
                wait_idle_quick_cmd();
                num_lans_get = 1;
                set_mode_cmd(0, 0);
                set_lans_idx_cmd();
    
                set_dram_write_lens_cmd(56);
                set_gemm_core_sel_cmd(1);
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
                if(second_core_en) {
                    set_gemm_core_sel_cmd(2);
                    for(int s = 0; s < 4; s++) {
                        my_float_t *pa = &a[(o_2 + s) * 3136];

                        set_num_lans_cmd(s%4);
                        
                        set_output_recv_cnt_cmd(0);
                        uint32_t tmp_s;
                        pa = pa + conv_0_offset;
                        memcpy(&tmp_s, &pa, sizeof(tmp_s));
                        set_dram_write_addr_cmd(s%4, tmp_s);
                        set_dram_w_tr_cmd();
                        wait_idle_quick_cmd();
                    }
                    set_gemm_core_sel_cmd(1);
                }
                conv_0_offset += 56;
            }

            wait_idle_cmd();
        }
        if(second_core_en) {
            o += weight_num;
        }
    }
    // for 3x3 kernel, paddingSize = 1
    else if (weight_.height_ == 3)
    for (int o = start; o < end; o += weight_num * 4)
    {
        // printf("[%s]: %d/%d\n", entry->base.layer_name_, (int)(o + 1), (int)end);
        
        int remain_oc = min(weight_num, end - o);
        int remain_oc_2, remain_oc_3, remain_oc_4;
        int o_2, o_3, o_4;
        
        int second_core_en, third_core_en, fourth_core_en;

        second_core_en = 1;
        // second_core_en = 0;
        o_2 = o + weight_num;
        remain_oc_2 = min(weight_num, end - o_2);

        third_core_en = 1;
        // third_core_en = 0;
        o_3 = o_2 + weight_num;
        remain_oc_3 = min(weight_num, end - o_3);

        fourth_core_en = 1;
        // fourth_core_en = 0;
        o_4 = o_3 + weight_num;
        remain_oc_4 = min(weight_num, end - o_4);

        // read first conv weight
        tmp_tick = clock();
        set_gemm_core_sel_cmd(1);
        int offset_w = (weight_.height_ * (in_.depth_ * o)) * weight_.width_;
        const my_float_t * ppw = W + offset_w;
        uint32_t tmp_s, tmp_s_2;
        memcpy(&tmp_s, &ppw, sizeof(tmp_s));
        read_conv_weight_cmd(weight_.height_ * weight_.width_ * in_.depth_, 
                             remain_oc, tmp_s);

        // read first batchNorm weight
        int batchNormIndex = 0;
        ppw = batchNorm_W + o;
        memcpy(&tmp_s, &ppw, sizeof(tmp_s));
        ppw = batchNorm_W + out_.depth_ + o;
        memcpy(&tmp_s_2, &ppw, sizeof(tmp_s));
        read_batchNorm_weight_cmd(remain_oc, tmp_s, tmp_s_2);
        set_dram_read_weight_cmd();
        
        // read second conv weight
        if(second_core_en)
        {
            set_gemm_core_sel_cmd(2);
            int offset_w = (weight_.height_ * (in_.depth_ * o_2)) * weight_.width_;
            const my_float_t * ppw = W + offset_w;
            uint32_t tmp_s, tmp_s_2;
            memcpy(&tmp_s, &ppw, sizeof(tmp_s));
            read_conv_weight_cmd(weight_.height_ * weight_.width_ * in_.depth_, 
                                remain_oc_2, tmp_s);
            wait_idle_2_quick_cmd();
            
            // read second batchNorm weight
            batchNormIndex = 0;
            ppw = batchNorm_W + o_2;
            memcpy(&tmp_s, &ppw, sizeof(tmp_s));
            ppw = batchNorm_W + out_.depth_ + o_2;
            memcpy(&tmp_s_2, &ppw, sizeof(tmp_s));
            read_batchNorm_weight_cmd(remain_oc, tmp_s, tmp_s_2);
            set_dram_read_weight_cmd();
        }

        if(third_core_en)
        {
            set_gemm_core_sel_cmd(4);
            int offset_w = (weight_.height_ * (in_.depth_ * o_3)) * weight_.width_;
            const my_float_t * ppw = W + offset_w;
            uint32_t tmp_s, tmp_s_2;
            memcpy(&tmp_s, &ppw, sizeof(tmp_s));
            read_conv_weight_cmd(weight_.height_ * weight_.width_ * in_.depth_, 
                                remain_oc_3, tmp_s);
            wait_idle_3_quick_cmd();
            
            // read third batchNorm weight
            // tmp_tick_2 = clock();
            batchNormIndex = 0;
            ppw = batchNorm_W + o_3;
            memcpy(&tmp_s, &ppw, sizeof(tmp_s));
            ppw = batchNorm_W + out_.depth_ + o_3;
            memcpy(&tmp_s_2, &ppw, sizeof(tmp_s));
            // ???
            read_batchNorm_weight_cmd(remain_oc, tmp_s, tmp_s_2);
            wait_idle_3_quick_cmd();
            set_dram_read_weight_cmd();
        }

        if(fourth_core_en)
        {
            set_gemm_core_sel_cmd(8);
            int offset_w = (weight_.height_ * (in_.depth_ * o_4)) * weight_.width_;
            const my_float_t * ppw = W + offset_w;
            uint32_t tmp_s, tmp_s_2;
            memcpy(&tmp_s, &ppw, sizeof(tmp_s));
            read_conv_weight_cmd(weight_.height_ * weight_.width_ * in_.depth_, 
                                remain_oc_4, tmp_s);
            wait_idle_4_quick_cmd();
            
            // read fourth batchNorm weight
            // tmp_tick_2 = clock();
            batchNormIndex = 0;
            ppw = batchNorm_W + o_4;
            memcpy(&tmp_s, &ppw, sizeof(tmp_s));
            ppw = batchNorm_W + out_.depth_ + o_4;
            memcpy(&tmp_s_2, &ppw, sizeof(tmp_s));
            // ???
            read_batchNorm_weight_cmd(remain_oc, tmp_s, tmp_s_2);
            wait_idle_4_quick_cmd();
            set_dram_read_weight_cmd();
        }
        send_weight_time += (clock() - tmp_tick)/ticks_per_msec;
        
        int output_offset = 0;

        int currHeightStart = -1;
        int currHeightEnd = min(h_per_op_hw - 2, in_.height_ - 1);

        for(int h = 0; h < out_.height_; h += out_h_per_op_hw) {

            int remain_num = out_.width_ * min(out_h_per_op_hw, out_.height_ - h);
            
            // read input from dram
            tmp_tick = clock();
            set_gemm_core_sel_cmd(15);
            int curr_h;
            if(currHeightStart < 0) {
                curr_h = 0;
            }
            else {
                curr_h = currHeightStart;
            }
            if(h == 0) {
                set_currinputtype_cmd(0);
                reset_sram_offset_cmd();
                set_reset_sram_cmd();
                set_sram_offset_cmd(in_padded_.width_ + 1); 

                set_length_cmd(((currHeightEnd - curr_h + 1)) * in_.width_);
                set_input_height_cmd(((currHeightEnd - curr_h + 1)));
                set_boundary_padding_size_cmd(in_padded_.width_ + 2);
            }
            else if(h + out_h_per_op_hw >= out_.height_ && h_stride_ == 1) {
                set_currinputtype_cmd(0);
                reset_sram_offset_cmd();
                set_reset_sram_cmd();
                set_sram_offset_cmd(paddingSize); 

                set_length_cmd(((currHeightEnd - curr_h + 1)) * in_.width_);
                set_input_height_cmd(((currHeightEnd - curr_h + 1)));
                set_boundary_padding_size_cmd(in_padded_.width_ + 2 + (h_per_op_hw - (currHeightEnd - curr_h + 1)-1) *  in_padded_.width_);

            }
            else {
                set_currinputtype_cmd(1);
                set_sram_offset_cmd(paddingSize);

                // set_length_cmd(((currHeightEnd - curr_h + 1)) * in_.width_);
                // set_input_height_cmd(((currHeightEnd - curr_h + 1)));
                set_length_cmd(h_per_op_hw * in_.width_);
                set_input_height_cmd(h_per_op_hw);
                set_boundary_padding_size_cmd(2);
            }
            
            currHeightStart += h_stride_ * out_h_per_op_hw;
            currHeightEnd   += h_stride_ * out_h_per_op_hw;
            currHeightEnd = min(currHeightEnd, in_.height_ - 1);

            int offset_i = (in_.width_ * (curr_h));
            my_float_t *pi = in + offset_i;
            uint32_t tmp_s;
            memcpy(&tmp_s, &pi, sizeof(tmp_s));

            read_conv_input_cmd(in_.depth_, in_.width_ * in_.height_ * 2,
                                tmp_s);
            send_data_time += (clock() - tmp_tick)/ticks_per_msec;

            tmp_tick = clock();
            
            trigger_conv_cmd();
            wait_idle_quick_cmd();
            wait_idle_2_quick_cmd();
            wait_idle_3_quick_cmd();
            wait_idle_4_quick_cmd();
            hardware_compute_time += (clock() - tmp_tick)/ticks_per_msec;
            
            // write result to dram
            // tmp_tick = clock();
            set_gemm_core_sel_cmd(1);
            set_dram_write_lens_cmd(remain_num);
            for(int s = 0; s < remain_oc; s++) {
                my_float_t *pa = &a[get_index(&out_, 0, 0, o + s)];
                set_num_lans_cmd(s%4);
                
                int base_idx = (s/4) * (input_num/4 + ((input_num%4) != 0));
                set_output_recv_cnt_cmd(base_idx);
                int tmp_s;
                pa = pa + output_offset;
                memcpy(&tmp_s, &pa, sizeof(tmp_s));
                set_dram_write_addr_cmd(s%4, tmp_s);

                set_dram_w_tr_cmd();
                wait_idle_quick_cmd();
            }
            
            wait_idle_cmd();
            // output_offset += input_num;

            if(second_core_en)
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

            if(third_core_en)
            {
                wait_idle_3_quick_cmd();
                set_gemm_core_sel_cmd(4);
                set_dram_write_lens_cmd(remain_num);
                for(int s = 0; s < remain_oc_3; s++) {
                    my_float_t *pa = &a[get_index(&out_, 0, 0, o_3 + s)];
                    set_num_lans_cmd(s%4);
                    
                    int base_idx = (s/4) * (input_num/4 + ((input_num%4) != 0));
                    set_output_recv_cnt_cmd(base_idx);
                    uint32_t tmp_s;
                    pa = pa + output_offset;
                    memcpy(&tmp_s, &pa, sizeof(tmp_s));
                    set_dram_write_addr_cmd(s%4, tmp_s);

                    set_dram_w_tr_cmd();
                    wait_idle_3_quick_cmd();
                }
                set_gemm_core_sel_cmd(1);
            }
            wait_idle_cmd();

            if(fourth_core_en)
            {
                wait_idle_4_quick_cmd();
                set_gemm_core_sel_cmd(8);
                set_dram_write_lens_cmd(remain_num);
                for(int s = 0; s < remain_oc_4; s++) {
                    my_float_t *pa = &a[get_index(&out_, 0, 0, o_4 + s)];
                    set_num_lans_cmd(s%4);
                    
                    int base_idx = (s/4) * (input_num/4 + ((input_num%4) != 0));
                    set_output_recv_cnt_cmd(base_idx);
                    uint32_t tmp_s;
                    pa = pa + output_offset;
                    memcpy(&tmp_s, &pa, sizeof(tmp_s));
                    set_dram_write_addr_cmd(s%4, tmp_s);

                    set_dram_w_tr_cmd();
                    wait_idle_4_quick_cmd();
                }
                set_gemm_core_sel_cmd(1);
            }
            wait_idle_cmd();
            output_offset += input_num;
            // store_data_time += (clock() - tmp_tick)/ticks_per_msec;
        }
    }
    // for 1x1 convolution
    else
    for (int o = start; o < end; o += weight_num * 4)
    {
        // printf("[%s]: %d/%d\n", entry->base.layer_name_, (int)(o + 1), (int)end);
        
        int remain_oc = min(weight_num, end - o);
        int remain_oc_2, remain_oc_3, remain_oc_4;
        int o_2, o_3, o_4;
        
        int second_core_en, third_core_en, fourth_core_en;

        second_core_en = 1;
        // second_core_en = 0;
        o_2 = o + weight_num;
        remain_oc_2 = min(weight_num, end - o_2);

        third_core_en = 1;
        // third_core_en = 0;
        o_3 = o_2 + weight_num;
        remain_oc_3 = min(weight_num, end - o_3);

        fourth_core_en = 1;
        // fourth_core_en = 0;
        o_4 = o_3 + weight_num;
        remain_oc_4 = min(weight_num, end - o_4);

        // read first conv weight
        tmp_tick = clock();
        set_gemm_core_sel_cmd(1);
        set_gemm_core_sel_cmd(1);
        int offset_w = (weight_.height_ * (in_.depth_ * o)) * weight_.width_;
        const my_float_t * ppw = W + offset_w;
        uint32_t tmp_s, tmp_s_2;
        memcpy(&tmp_s, &ppw, sizeof(tmp_s));
        read_conv_weight_cmd(weight_.height_ * weight_.width_ * in_.depth_, 
                             remain_oc, tmp_s);

        // read first batchNorm weight
        int batchNormIndex = 0;
        ppw = batchNorm_W + o;
        memcpy(&tmp_s, &ppw, sizeof(tmp_s));
        ppw = batchNorm_W + out_.depth_ + o;
        memcpy(&tmp_s_2, &ppw, sizeof(tmp_s));
        read_batchNorm_weight_cmd(remain_oc, tmp_s, tmp_s_2);
        set_dram_read_weight_cmd();
        
        // read second conv weight
        if(second_core_en)
        {
            set_gemm_core_sel_cmd(2);
            int offset_w = (weight_.height_ * (in_.depth_ * o_2)) * weight_.width_;
            const my_float_t * ppw = W + offset_w;
            uint32_t tmp_s, tmp_s_2;
            memcpy(&tmp_s, &ppw, sizeof(tmp_s));
            read_conv_weight_cmd(weight_.height_ * weight_.width_ * in_.depth_, 
                                remain_oc_2, tmp_s);
            
            // read second batchNorm weight
            batchNormIndex = 0;
            ppw = batchNorm_W + o_2;
            memcpy(&tmp_s, &ppw, sizeof(tmp_s));
            ppw = batchNorm_W + out_.depth_ + o_2;
            memcpy(&tmp_s_2, &ppw, sizeof(tmp_s));
            read_batchNorm_weight_cmd(remain_oc, tmp_s, tmp_s_2);
            set_dram_read_weight_cmd();
        }

        // read third conv weight
        if(third_core_en)
        {
            set_gemm_core_sel_cmd(4);
            int offset_w = (weight_.height_ * (in_.depth_ * o_3)) * weight_.width_;
            const my_float_t * ppw = W + offset_w;
            uint32_t tmp_s, tmp_s_2;
            memcpy(&tmp_s, &ppw, sizeof(tmp_s));
            read_conv_weight_cmd(weight_.height_ * weight_.width_ * in_.depth_, 
                                remain_oc_3, tmp_s);
            wait_idle_3_quick_cmd();
            
            // read third batchNorm weight
            batchNormIndex = 0;
            ppw = batchNorm_W + o_3;
            memcpy(&tmp_s, &ppw, sizeof(tmp_s));
            ppw = batchNorm_W + out_.depth_ + o_3;
            memcpy(&tmp_s_2, &ppw, sizeof(tmp_s));
            // ???
            read_batchNorm_weight_cmd(remain_oc, tmp_s, tmp_s_2);
            wait_idle_3_quick_cmd();
            set_dram_read_weight_cmd();
        }

        // read fourth conv weight
        if(fourth_core_en)
        {
            set_gemm_core_sel_cmd(8);
            int offset_w = (weight_.height_ * (in_.depth_ * o_4)) * weight_.width_;
            const my_float_t * ppw = W + offset_w;
            uint32_t tmp_s, tmp_s_2;
            memcpy(&tmp_s, &ppw, sizeof(tmp_s));
            read_conv_weight_cmd(weight_.height_ * weight_.width_ * in_.depth_, 
                                remain_oc_4, tmp_s);
            wait_idle_4_quick_cmd();
            
            // read fourth batchNorm weight
            batchNormIndex = 0;
            ppw = batchNorm_W + o_4;
            memcpy(&tmp_s, &ppw, sizeof(tmp_s));
            ppw = batchNorm_W + out_.depth_ + o_4;
            memcpy(&tmp_s_2, &ppw, sizeof(tmp_s));
            // ???
            read_batchNorm_weight_cmd(remain_oc, tmp_s, tmp_s_2);
            wait_idle_4_quick_cmd();
            set_dram_read_weight_cmd();
        }
        send_weight_time += (clock() - tmp_tick)/ticks_per_msec;
        
        int output_offset = 0;
        /*
         * CONGIF RACH READ LENGTH FROM DRAM
         */
        set_gemm_core_sel_cmd(15);
        set_dram_read_input_cmd();
        set_gemm_core_sel_cmd(1);

        int currHeight = 0;

        for(int h = 0; h < out_.height_; h += out_h_per_op_hw) {

            int remain_num = out_.width_ * min(out_h_per_op_hw, out_.height_ - h);

            // read input from dram
            tmp_tick = clock();
            set_gemm_core_sel_cmd(15);
            int curr_h;
            curr_h = currHeight;
            
            set_sram_offset_cmd(0); 

            set_length_cmd(in_.width_ * h_per_op_hw);
            set_input_height_cmd(h_per_op_hw);
            set_paddingsize_cmd(0);
            set_boundary_padding_size_cmd(0);

            currHeight += h_stride_ * out_h_per_op_hw;
            int offset_i = (in_.width_ * (curr_h));
            my_float_t *pi = in + offset_i;
            uint32_t tmp_s;
            memcpy(&tmp_s, &pi, sizeof(tmp_s));

            read_conv_input_cmd(in_.depth_, in_.width_ * in_.height_*2,
                                tmp_s);
            send_data_time += (clock() - tmp_tick)/ticks_per_msec;

            tmp_tick = clock();

            trigger_conv_cmd();
            wait_idle_quick_cmd();
            hardware_compute_time += (clock() - tmp_tick)/ticks_per_msec;
            
            // write result back to dram
            // tmp_tick = clock();
            set_gemm_core_sel_cmd(1);
            set_dram_write_lens_cmd(remain_num);
            for(int s = 0; s < remain_oc; s++) {
                my_float_t *pa = &a[get_index(&out_, 0, 0, o + s)];
                set_num_lans_cmd(s%4);
                
                int base_idx = (s/4) * (input_num/4 + ((input_num%4) != 0));
                set_output_recv_cnt_cmd(base_idx);
                int tmp_s;
                pa = pa + output_offset;
                memcpy(&tmp_s, &pa, sizeof(tmp_s));
                set_dram_write_addr_cmd(s%4, tmp_s);

                set_dram_w_tr_cmd();
                wait_idle_quick_cmd();
            }
            
            wait_idle_cmd();

            if(second_core_en)
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

            if(third_core_en)
            {
                wait_idle_3_quick_cmd();
                set_gemm_core_sel_cmd(4);
                set_dram_write_lens_cmd(remain_num);
                for(int s = 0; s < remain_oc_3; s++) {
                    my_float_t *pa = &a[get_index(&out_, 0, 0, o_3 + s)];
                    set_num_lans_cmd(s%4);
                    
                    int base_idx = (s/4) * (input_num/4 + ((input_num%4) != 0));
                    set_output_recv_cnt_cmd(base_idx);
                    uint32_t tmp_s;
                    pa = pa + output_offset;
                    memcpy(&tmp_s, &pa, sizeof(tmp_s));
                    set_dram_write_addr_cmd(s%4, tmp_s);

                    set_dram_w_tr_cmd();
                    wait_idle_3_quick_cmd();
                }
                set_gemm_core_sel_cmd(1);
            }
            wait_idle_cmd();

            if(fourth_core_en)
            {
                wait_idle_4_quick_cmd();
                set_gemm_core_sel_cmd(8);
                set_dram_write_lens_cmd(remain_num);
                for(int s = 0; s < remain_oc_4; s++) {
                    my_float_t *pa = &a[get_index(&out_, 0, 0, o_4 + s)];
                    set_num_lans_cmd(s%4);
                    
                    int base_idx = (s/4) * (input_num/4 + ((input_num%4) != 0));
                    set_output_recv_cnt_cmd(base_idx);
                    uint32_t tmp_s;
                    pa = pa + output_offset;
                    memcpy(&tmp_s, &pa, sizeof(tmp_s));
                    set_dram_write_addr_cmd(s%4, tmp_s);

                    set_dram_w_tr_cmd();
                    wait_idle_4_quick_cmd();
                }
                set_gemm_core_sel_cmd(1);
            }
            wait_idle_cmd();
            output_offset += input_num;
            // store_data_time += (clock() - tmp_tick)/ticks_per_msec;
        }
    }
    //###########################################################
    // print output tensor
    // set_gemm_core_sel_cmd(1);
    // set_read_rounds_cmd(1);
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
    //             for (int w = 0; w < 56; w++) {
    //                 // printf("%2.6f ", ((float)out[p_cnt]));
    //                 printf("%2.6f ", ((float_t)read_dram_value_cmd(&out[p_cnt])));
                    
    //                 p_cnt++;
    //             }
    //             printf("]\n");
    //         }
    //     }
    // }
    // if(conv_cnt == 15)
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

    if(entry->delete_input) {
        free(in_ptr_prev);
    }
    // free(entry->base.padded_ptr);
    conv_cnt++;
    
    set_gemm_core_sel_cmd(1);
    set_read_rounds_cmd(1);
    set_paddingsize_cmd(0);
    set_boundary_padding_size_cmd(0);

#ifdef PRINT_LAYER
    // printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, (float_t)out[0], (float_t)out[1], (float_t)out[entry->base.out_size_-2], (float_t)out[entry->base.out_size_-1]);
    // printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, (float_t)read_dram_value_cmd(&out[0]), (float_t)read_dram_value_cmd(&out[1]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-2]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-1]));
#endif

#ifdef USING_GEM5
    tick = (clock() - tick)/ticks_per_msec;
    printf("It took %7ld msec to perform conv.\n\n", tick);
    printf("It took %7ld msec to perform on preprocess_time.\n", preprocess_time);
    // printf("It took %7ld msec to perform on hardware_compute_time.\n", hardware_compute_time);
    // printf("It took %7ld msec to perform on send_data_time.\n", send_data_time);
    // printf("It took %7ld msec to perform on send_weight_time.\n", send_weight_time);
    // printf("It took %7ld msec to perform on send_idx_time.\n", send_idx_time);
    // printf("It took %7ld msec to perform on padding_time.\n", padding_time);
    // printf("It took %7ld msec to perform on store_data_time.\n\n", store_data_time);
    printf("[%s] done [%f, %f, ... , %f, %f]\n\n", entry->base.layer_name_, (float_t)read_dram_value_cmd(&out[0]), (float_t)read_dram_value_cmd(&out[1]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-2]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-1]));
#endif
}

layer_base * new_convolutional_layer(
                                     cnn_controller *ctrl,
                                     my_float_t(*activate) (my_float_t *, int, int),
                                     int in_width,
                                     int in_height,
                                     int window_width,
                                     int window_height,
                                     int in_channels,
                                     int out_channels,
                                     enum padding  pad_type,
                                     uint8_t  has_bias,
                                     int w_stride,
                                     int h_stride,
                                     int w_padding,
                                     int h_padding,
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
    static int call_time = 0;
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

