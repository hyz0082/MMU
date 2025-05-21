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
// #ifdef USING_GEM5
//     clock_t  tick, ticks_per_msec = CLOCKS_PER_SEC/1000;
//     clock_t  send_data_time = 0, hardware_compute_time = 0;
//     tick = clock();
// #endif
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
                wait_idle_cmd();
                /*
                * start BatchNorm
                */
                set_mode_cmd(1, remain_len);
                // reset_relu_cmd();
                trigger_add_cmd();
                wait_idle_cmd();
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
    clock_t  tick, ticks_per_msec = CLOCKS_PER_SEC/1000/1000;
    clock_t  send_data_time = 0, hardware_compute_time = 0;
    clock_t  send_weight_time = 0, send_idx_time = 0;
    clock_t  padding_time = 0, store_data_time = 0;
    clock_t  tmp_tick = clock();
    tick = clock();
#endif

    convolutional_layer *entry = get_convolutional_layer_entry(ptr);

    static int conv_cnt = 0;
    
    if (input->in_size_ != entry->base.in_size_)
    {
        if (hart_id == 0) 
            printf("Error input size not match %d/%d\n", input->in_size_, entry->base.in_size_);
        exit(-1);
    }

    // malloc input space
    entry->base.padded_ptr = (my_float_t *)malloc(entry->base.padding_size * sizeof(my_float_t));
    if (entry->base.padded_ptr != NULL) { // Check if memory was allocated
        // memset((void*)entry->base.padded_ptr, 0, entry->base.padding_size * sizeof(my_float_t));
        // for(int i = 0; i < entry->base.padding_size; i++) {
        //     write_dram_value_cmd(&entry->base.padded_ptr[i], 0);
        // }
        reset_dram_value_cmd(entry->base.padded_ptr, entry->base.padding_size);
    }
    else {
        printf("Error: Unable to allocate memory for layer->padded_ptr\n");
        exit(1);
    }

    // // malloc output space
    entry->base.a_ptr_ = (my_float_t *)malloc(entry->base.out_size_ * sizeof(my_float_t));
    if (entry->base.a_ptr_ != NULL) { // Check if memory was allocated
        // memset((void*)entry->base.a_ptr_, 0, entry->base.out_size_ * sizeof(my_float_t));
        // for(int i = 0; i < entry->base.out_size_; i++) {
        //     write_dram_value_cmd(&entry->base.a_ptr_[i], 0);
        // }
        // reset_dram_value_cmd(entry->base.a_ptr_, entry->base.out_size_);
    }
    else {
        printf("Error: Unable to allocate memory for entry->base.a_ptr_\n");
        exit(1);
    }

 
    entry->base.out_ptr_ = entry->base.a_ptr_;

    // conv_copy_and_pad_input_sw(entry, hart_id, input);
    // conv_copy_and_pad_input_sw(entry, hart_id, input);
    tmp_tick = clock();
    if(entry->weight_.width_ != 1) {
        conv_copy_and_pad_input(entry, hart_id, input);
    }
    padding_time += (clock() - tmp_tick)/ticks_per_msec;

    my_float_t *in_ptr_prev = input->in_ptr_;
    // if(entry->delete_input) {
    //     free(input->in_ptr_);
    // }
    // free(entry->base.padded_ptr);
    my_float_t *in;
    if(entry->weight_.width_ != 1) {
        // my_float_t *in = entry->base.padded_ptr;
        in = entry->base.padded_ptr;
    }
    else {
        in = input->in_ptr_;
    }
    my_float_t *a = entry->base.a_ptr_;

    my_float_t *W = weights + entry->base.weight_offset;//entry->base._W;
    
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

    uint64_t total_size = out_.depth_;
    uint64_t blocksize = compute_block_size(total_size);
    uint64_t start = (blocksize) * hart_id;
    uint64_t end = min((blocksize) * (hart_id+1), total_size);

    uint64_t out_dim = out_.height_*out_.width_;
    
    uint32_t tmp_3, tmp_4;
    uint32_t tmp[10];

    reset_cmd();

    int data_gbuff_size   = 32768;
    int weight_gbuff_size = 32768;
    int idx_gbuff_size    = 32768;
    /*
     * modify height_per_operation on each convolutional layer
     */
    /*
     * weight
     */
    int kernel_len = (in_.depth_) * (weight_.height_) * (weight_.width_);
    int weight_num = weight_gbuff_size / kernel_len;
    /*
     * data
     */
    int ch_per_op_hw = in_.depth_;
    /*
        I = Input Size
        K = Kernel Size
        P = Padding
        S = Stride
        O = Output Size
    Then:
        O = (I - K + 2P) / S + 1
    */
    int h_per_op_hw = data_gbuff_size / (ch_per_op_hw * in_padded_.width_);
    int input_num, out_h_per_op_hw, out_w_per_op_hw;

    input_num = out_.width_ * ((h_per_op_hw - weight_.height_) / h_stride_ + 1);
    out_w_per_op_hw = out_.width_;
    out_h_per_op_hw = ((h_per_op_hw - weight_.height_) / h_stride_ + 1);
    printf("original shape: %d, %d\n", (int)input_num, (int)weight_num);
    while(input_num * weight_num > 1024 || (input_num * in_.depth_ * weight_.height_ * weight_.width_) >= idx_gbuff_size) {
        h_per_op_hw--;
        weight_num--;

        input_num = out_.width_ * ((h_per_op_hw - weight_.height_) / h_stride_ + 1);
        out_w_per_op_hw = out_.width_;
        out_h_per_op_hw = ((h_per_op_hw - weight_.height_) / h_stride_ + 1);
    }

    // int size_per_channel_hw = h_per_op_hw * in_padded_.width_;

    // printf("each HW conv shape: %d %d %d\n", input_num, weight_num, kernel_len);
    if(input_num == 0 || weight_num == 0) {
        // printf("error shape\n");
        // exit(1);
        h_per_op_hw = weight_.height_; //h_stride_ + 
        weight_num  = 1024 / in_padded_.width_;//4;
    }
    // set h = 1
    /*
    have bug when h_per_op_hw >= weight_.height_;
    */
    // h_per_op_hw = weight_.height_;

    input_num = out_.width_ * ((h_per_op_hw - weight_.height_) / h_stride_ + 1);
    out_w_per_op_hw = out_.width_;
    out_h_per_op_hw = ((h_per_op_hw - weight_.height_) / h_stride_ + 1);

    int size_per_channel_hw = h_per_op_hw * in_padded_.width_;
    printf("val: %d %d\n", h_per_op_hw, in_padded_.width_);
    // printf("float: %d, _Float16: %d\n", sizeof(float_t), sizeof(my_float_t));

    /*
     * debug setting
     */
    // weight_num = 4;
    // if(1) {
    //     send_param1_cmd(60000);
    //     send_param2_cmd(60000);
    //     // read_dram_value_cmd(&out[entry->base.out_size_-1]);
    //     read_dram_value_cmd(&out[entry->base.out_size_-3]);
    // }
    if(input_num * weight_num > 1024) {
        printf("error shape\n");
        exit(1);
    }
    printf("input  shape: (%d, %d, %d)\n", (int)in_.width_, (int)in_.height_, (int)in_.depth_);
    printf("weight shape: (%d, %d, %d)\n", (int)weight_.width_, (int)weight_.height_, (int)in_.depth_);
    printf("input  sram usage: %d\n", (int)size_per_channel_hw * (int)in_.depth_);
    printf("weight sram usage: %d\n", (int)weight_.height_ * (int)weight_.width_ * (int)in_.depth_ * (int)weight_num);
    printf("each HW conv shape: %d %d %d\n", input_num, weight_num, kernel_len);
    printf("result sram %d\n", (int)input_num*(int)weight_num);

    set_KMN_cmd(kernel_len, input_num, weight_num);
    set_conv_cmd(kernel_len);
    set_idx_cmd(0, kernel_len, kernel_len*2, kernel_len*3);
    reset_preload_cmd();

    /*
     * idx ram setting
     */
    tmp_tick = clock();
    int index_offset = 0;

    for(uint64_t y = 0; y < out_h_per_op_hw; y++) {
        for(uint64_t x = 0; x < out_w_per_op_hw; x++) {
            int input_offset = (y * h_stride_) * in_padded_.width_ + x * w_stride_;
            for(uint64_t compute_ch = 0; compute_ch < in_.depth_; compute_ch++) {
                int ch_offset = (h_per_op_hw * compute_ch) * in_padded_.width_;
                for(uint64_t wy = 0; wy < weight_.height_; wy++) {
                    // int index_value = ch_offset + input_offset;
                    for(uint64_t wx = 0; wx < weight_.width_; wx++) {
                        int index_value = ch_offset + input_offset +
                                            (wy * in_padded_.width_ + wx);
                        // printf("send idx: %d %d\n", index_value, index_offset);
                        send_idx_cmd(index_value, index_offset++);
                        // if(index_offset >= 32768) {
                        //     printf("error idx\n");
                        //     exit(1);
                        // }
                    }
                }
            }
        }
    }
    send_idx_time += (clock() - tmp_tick)/ticks_per_msec;
    /*
     * end
     */

    int max_remain_num = 99999;
    for (uint64_t o = start; o < end; o += weight_num)
    {
        printf("[%s]: %d/%d\n", entry->base.layer_name_, (int)(o + 1), (int)end);
        
        // send remain_oc ch weight
        int send_weight_cnt = 0;
        int remain_oc = min(weight_num, end - o);

        /*
         * SW SEND WEIGHT TO GeMM HW
         */
        //  for(int curr_oc = o; curr_oc < o + remain_oc; curr_oc++) {
        //     int offset_w = (weight_.height_ * (in_.depth_ * curr_oc + 0)) * weight_.width_;
        //     const my_float_t *pw = W + offset_w;
        //     const my_float_t * ppw = pw;

        //     for (uint64_t inc = 0; inc < in_.depth_; inc++) {
        //         int offset_w = (weight_.height_ * (in_.depth_ * curr_oc + inc)) * weight_.width_;
        //         const my_float_t *pw = W + offset_w;
        //         const my_float_t * ppw = pw;
        //         for (uint64_t wy = 0; wy < weight_.height_; wy++) {
        //             for (uint64_t wx = 0; wx < weight_.width_; wx++) {
        //                 send_weight_cmd(*ppw++, send_weight_cnt++);
        //             }
        //         }
        //     }
        // }

        /*
         * GeMM HW READ WEIGHT FROM DRAM
         * READ (weight_.height_ * weight_.width_ * in_.depth_) 
         * PER OP
         */
        tmp_tick = clock();
        set_dram_read_weight_cmd();
        reset_sram_offset_cmd();
        set_length_cmd(weight_.height_ * weight_.width_ * in_.depth_);
        for(int curr_oc = o; curr_oc < o + remain_oc; curr_oc++) {
            int offset_w = (weight_.height_ * (in_.depth_ * curr_oc + 0)) * weight_.width_;
            const my_float_t *pw = W + offset_w;
            const my_float_t * ppw = pw;
            uint32_t tmp_s;
            memcpy(&tmp_s, &ppw, sizeof(tmp_s));
            set_addr_cmd(tmp_s);
            trigger_dram_read_cmd();
            wait_idle_cmd();
            // wait_idle_quick_cmd();
        }
        send_weight_time += (clock() - tmp_tick)/ticks_per_msec;
        
        int output_offset = 0;
        
        for(uint64_t h = 0; h < out_.height_; h += out_h_per_op_hw) {
            
            int send_data_offset = 0;
            int remain_num = out_.width_ * min(out_h_per_op_hw, out_.height_ - h);
            max_remain_num = min(max_remain_num, remain_num);
            
            /*
             * SW SEND INPUT
             */
            // uint64_t curr_h = h * h_stride_;
            // for(uint64_t inc = 0; inc < in_.depth_; inc++) {
            //     int offset_i = (in_padded_.width_ * in_padded_.height_ * inc) +
            //                     (in_padded_.width_ * (curr_h));
            //     my_float_t *pi = in + offset_i;

            //     for(uint64_t i = 0; i < size_per_channel_hw; i++) {
            //         send_data_cmd(*pi++, send_data_offset++);
            //     }
            // }
            /*
             * HW SEND INPUT
             */
            tmp_tick = clock();
            uint64_t curr_h = h * h_stride_;
            reset_sram_offset_cmd();
            set_length_cmd(size_per_channel_hw);
            set_dram_read_input_cmd();
            for(uint64_t inc = 0; inc < in_.depth_; inc++) {
                int offset_i = (in_padded_.width_ * in_padded_.height_ * inc) +
                                (in_padded_.width_ * (curr_h));
                my_float_t *pi = in + offset_i;
                uint32_t tmp_s;
                memcpy(&tmp_s, &pi, sizeof(tmp_s));
                set_addr_cmd(tmp_s);
                trigger_dram_read_cmd();
                wait_idle_cmd();
                // wait_idle_quick_cmd();
            }
            send_data_time += (clock() - tmp_tick)/ticks_per_msec;

#ifdef USING_GEM5
            // clock_t  tmp_tick = clock();
            __asm__ volatile ("nop");
            __asm__ volatile ("nop");
            __asm__ volatile ("nop");
            __asm__ volatile ("nop");
#endif
            tmp_tick = clock();
            trigger_conv_cmd();
            // wait_idle_cmd();
            wait_idle_quick_cmd();
            hardware_compute_time += (clock() - tmp_tick)/ticks_per_msec;

#ifdef USING_GEM5
            // hardware_compute_time += (clock() - tmp_tick)/(ticks_per_msec/1000);
            // printf("It took %ld msec to perform on HW.\n\n", hardware_compute_time);
#endif
            
            /*
             * SW WRITE RESULT
             */
            int write_source = 1; // 0:sw, 1:hw
            if(!write_source)
            {
            for(int s = 0; s < remain_oc; s++) {
                my_float_t *pa = &a[get_index(&out_, 0, 0, o + s)];
                int base_offset = (s%4)*4;
                int base_idx = (s/4) * (input_num/4 + ((input_num%4) != 0));
                for(int k = 0; k < remain_num; k++) {
                    int target_offset = base_offset + (k%4);
                    int target_idx    = base_idx    + (k/4);
                    
                    my_float_t ff = read_data_cmd(target_offset, target_idx);
                    pa[output_offset + k] = ff;
                    // write_dram_value_cmd(&pa[output_offset + k], ff);
                    /*
                     * print dram content
                     */
                    // my_float_t *tmp_ptr = &pa[output_offset + k];
                    // uint32_t tmp_d;
                    // memcpy(&tmp_d, &tmp_ptr, sizeof(tmp_d));
                    // printf("value: %x at %x\n", *tmp_ptr, tmp_d);
                }
            }
            }

            /*
             * HW WRITE RESULT
             */
            tmp_tick = clock();
            if(write_source)
            {
                // set_dram_write_lens_cmd(remain_num);
                // reset_output_recv_cnt_cmd();
                // for(int s = 0; s < remain_oc; s+=4) {
                //     int remain_lans = min(4, remain_oc - s);
                //     set_num_lans_cmd(remain_lans);
                //     int pos = 0;
                //     for(int ps = s; ps < s + remain_lans; ps++) {
                //         my_float_t *pi = a + (out_.height_ * (o + ps)) * out_.width_;
                //         pi += output_offset;
                //         uint32_t tmp_s;
                //         // my_float_t *pi = &a[get_index(&out_, 0, 0, o + s) + output_offset];
                //         memcpy(&tmp_s, &pi, sizeof(tmp_s));
                //         set_dram_write_addr_cmd(pos++, tmp_s);
                //         // printf("write addr: %x, len: %d\n", tmp_s, remain_num);
                //     }
                //     set_dram_w_tr_cmd();
                //     wait_idle_cmd();
                // }
                set_dram_write_lens_cmd(remain_num);
                for(int s = 0; s < remain_oc; s++) {
                    my_float_t *pa = &a[get_index(&out_, 0, 0, o + s)];
                    set_num_lans_cmd(s%4);
                    
                    // int base_offset = (s%4)*4;
                    int base_idx = (s/4) * (input_num/4 + ((input_num%4) != 0));
                    set_output_recv_cnt_cmd(base_idx);
                    uint32_t tmp_s;
                    pa = pa + output_offset;
                    memcpy(&tmp_s, &pa, sizeof(tmp_s));
                    set_dram_write_addr_cmd(s%4, tmp_s);

                    set_dram_w_tr_cmd();
                    wait_idle_cmd();
                    // wait_idle_cmd();
                    // wait_idle_quick_cmd();
                }
            }
            store_data_time += (clock() - tmp_tick)/ticks_per_msec;
            output_offset += input_num;
        }
        // printf("value: %f %f\n", (float)read_dram_value_cmd(&out[0]), (float_t)read_dram_value_cmd(&out[1]));
        // printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, (float_t)read_dram_value_cmd(&out[0]), (float_t)read_dram_value_cmd(&out[1]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-2]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-1]));
    }
    // printf("value: %f %f\n", (float)read_dram_value_cmd(&out[0]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-1]));
    // printf("value: %f %f\n", (float)read_dram_value_cmd(&out[0]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-1]));
    //###########################################################
    // print output tensor
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

// #ifdef STORE_MID
//     if(conv_cnt == 30) {
//         int my_size = 256*1024;
//         uint32_t tmp;
//         fprintf(mid_file, "int mid_weight[%d] = {", my_size );
//         for (uint64_t i = 0; i < my_size; i++)
//         {
//             memcpy(&tmp, &entry->base._W[i], sizeof(uint32_t));
//             // out[i];
//             if(i == my_size - 1) {
//                 fprintf(mid_file, "%d", tmp);
//             }
//             else {
//                 fprintf(mid_file, "%d, ", tmp);
//             }
//         }
//         fprintf(mid_file, "%s", "};");
//     }
// #endif
    // if(conv_cnt == 1) {
    //     send_param1_cmd(55000);
    //     send_param2_cmd(55000);
    //     read_dram_value_cmd(&out[0]);
    //     // read_dram_value_cmd(&out[0]);
    // }
    // printf("before free\n");
    // printf("value: %f %f\n", (float)read_dram_value_cmd(&out[0]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-1]));
    if(entry->delete_input) {
        free(in_ptr_prev);
    }
    free(entry->base.padded_ptr);
    // printf("after free\n");
    // printf("value: %f %f\n", (float)read_dram_value_cmd(&out[0]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-1]));
    conv_cnt++;
    // printf("value: %f %f\n", (float)read_dram_value_cmd(&out[0]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-1]));
#ifdef PRINT_LAYER
    if (hart_id == 0) 
    {
        // printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, (float_t)out[0], (float_t)out[1], (float_t)out[entry->base.out_size_-2], (float_t)out[entry->base.out_size_-1]);
        printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, (float_t)read_dram_value_cmd(&out[0]), (float_t)read_dram_value_cmd(&out[1]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-2]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-1]));
    }
#endif

#ifdef USING_GEM5
    tick = (clock() - tick)/ticks_per_msec;
    printf("It took %ld msec to perform conv.\n\n", tick);
    printf("It took %ld msec to perform on hardware_compute_time.\n", hardware_compute_time);
    printf("It took %ld msec to perform on send_data_time.\n", send_data_time);
    printf("It took %ld msec to perform on send_weight_time.\n", send_weight_time);
    printf("It took %ld msec to perform on send_idx_time.\n", send_idx_time);
    printf("It took %ld msec to perform on padding_time.\n", padding_time);
    printf("It took %ld msec to perform on store_data_time.\n\n", store_data_time);
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

