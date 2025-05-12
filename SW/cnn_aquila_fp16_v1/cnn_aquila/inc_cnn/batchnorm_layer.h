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

my_float_t _beta(batchnorm_layer *entry, uint64_t ind) {
    return entry->base._W[(entry->in_.depth_ * 1) + ind];
}

my_float_t _gamma(batchnorm_layer *entry, uint64_t ind) {
    return entry->base._W[(entry->in_.depth_ * 0) + ind];
}

my_float_t _mean(batchnorm_layer *entry, uint64_t ind) {
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

my_float_t _invstd(batchnorm_layer *entry, uint64_t ind) {
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
#endif

    batchnorm_layer *entry = get_batchnorm_layer_entry(ptr);
    static int bn_cnt = 0;
    if (input->in_size_ != entry->base.in_size_)
    {
        if (hart_id == 0) 
            printf("Error input size not match %d/%d\n", input->in_size_, entry->base.in_size_);
        exit(-1);
    }

    // add
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
    uint64_t dim = in_.height_*in_.width_;

    uint64_t total_size = dim;
    uint64_t blocksize = compute_block_size(total_size);
    uint64_t start = (blocksize) * hart_id;
    uint64_t end = min((blocksize) * (hart_id+1), total_size);

    reset_cmd();
    if(bn_cnt == 0) {
        int pad_size = 114*114*64;
        my_float_t *pad_ptr = (my_float_t *)malloc(pad_size * sizeof(my_float_t));
        if (pad_ptr != NULL) { // Check if memory was allocated
            // memset((void*)pad_ptr, 0, pad_size * sizeof(my_float_t));
            reset_dram_value_cmd(pad_ptr, pad_size);
        }
        else {
            printf("Error: Unable to allocate memory for pad_size\n");
            exit(1);
        }
        my_float_t *dst = pad_ptr;

        uint64_t total_size = 64 * 112;
        uint64_t end = total_size;

        for (uint64_t i = 0; i < end; i++)
        {
            uint64_t c = i / 112;
            uint64_t y = i % 112;
            int pad_pos = (114 * c + (1 + y)) * 114 + 1;
            my_float_t *pimg = &dst[pad_pos];
            pad_pos = (112 * c + y) * 112 + 0;
            my_float_t *pin = &in[pad_pos];

            // for (uint64_t x = 0; x < 112; x++)
            // {
            //     // pimg[x] = pin[x];
            //     write_dram_value_cmd(&pimg[x], read_dram_value_cmd(&pin[x]));
            // }
            send_bn_mul_data(1, 0);
            send_bn_add_data(0, 0);
            // int max_len = 112;
            int remain_len = 112;//min(max_len, in_.width_ - j);
            /*
            * send data
            */
            reset_sram_offset_cmd();
            set_length_cmd(remain_len);
            set_dram_read_input_cmd();
            uint32_t tmp_s;
            memcpy(&tmp_s, &pin, sizeof(tmp_s));
            set_addr_cmd(tmp_s);
            trigger_dram_read_cmd();
            wait_idle_cmd();
            /*
            * start BatchNorm
            */
            set_mode_cmd(1, remain_len);
            reset_relu_cmd();
            trigger_add_cmd();
            wait_idle_cmd();
            set_mode_cmd(0, 0);
 
            /*
            * write data
            */
            set_dram_write_lens_cmd(remain_len);
            set_num_lans_cmd(0);
            set_output_recv_cnt_cmd(0);
            memcpy(&tmp_s, &pimg, sizeof(tmp_s));
            set_dram_write_addr_cmd(0, tmp_s);
            set_dram_w_tr_cmd();
            wait_idle_cmd();

            pin += remain_len;
            pimg += remain_len; 
            __asm__ volatile ("nop");
            
#ifdef USING_GEM5
            clock_t  tmp_tick = clock();
            hardware_compute_time += (clock() - tmp_tick)/(ticks_per_msec/1000);
            tmp_tick = clock();
            hardware_compute_time += (clock() - tmp_tick)/(ticks_per_msec/100);
#endif
        }
        set_max_pooling_cmd();
        for (uint64_t ch = 0; ch < in_.depth_; ch++){
            my_float_t mul = _gamma(entry, ch) * _invstd(entry, ch);
            my_float_t add = _gamma(entry, ch) * (- _mean(entry, ch)) * _invstd(entry, ch) + _beta(entry, ch);

            send_bn_mul_data(mul, 0);
            send_bn_add_data(add, 0);

            for(int y = 0; y < 56; y++) {
                for(int x = 0; x < 56; x++) {
                    int data_pos = 0;
                    uint64_t base_pos = (ch * 114*114) + (y*114*2) + (x*2);
                    if(x == 0 || y == 0) {
                        float curr_max = (my_float_t)-DBL_MAX;
                        for(int py = 0; py < 3; py++) {
                            for(int px = 0; px < 3; px++) {
                                if(x + px == 0 || y + py == 0) {
                                    continue;
                                }
                                uint64_t target_pos = base_pos + py*114+px;
                                // curr_max = max(curr_max, pad_ptr[target_pos]*mul + add);
                                curr_max = max(curr_max, read_dram_value_cmd(&pad_ptr[target_pos])*mul + add);
                            }
                        }
                        curr_max = max(curr_max, (my_float_t)0);
                        uint64_t pos = (ch*56*56) + (y*56) + x;
                        // in[pos] = curr_max;
                        write_dram_value_cmd(&in[pos], curr_max);
                    }
                    else {
                        /*
                         * sw send input
                         */
                        // for(int py = 0; py < 3; py++) {
                        //     for(int px = 0; px < 3; px++) {
                        //         uint64_t target_pos = base_pos + py*114+px;
                        //         // send_bn_data(pad_ptr[target_pos], data_pos++);
                        //         // send_bn_data(read_dram_value_cmd(&pad_ptr[target_pos]), data_pos++);
                        //         send_data_cmd(read_dram_value_cmd(&pad_ptr[target_pos]), data_pos++);
                                
                        //     }
                        // }
                        /*
                         * hw send input
                         */
                        reset_sram_offset_cmd();
                        set_length_cmd(3);
                        set_dram_read_input_cmd();
                        
                        for(int py = 0; py < 3; py++) {
                            // for(int px = 0; px < 3; px++) {
                            //     uint64_t target_pos = base_pos + py*114+px;
                            //     // send_bn_data(pad_ptr[target_pos], data_pos++);
                            //     // send_bn_data(read_dram_value_cmd(&pad_ptr[target_pos]), data_pos++);
                            //     send_data_cmd(read_dram_value_cmd(&pad_ptr[target_pos]), data_pos++);
                            // }
                            uint64_t target_pos = base_pos + py*114;
                            my_float_t *pi = &pad_ptr[target_pos];
                            uint32_t tmp_s;
                            memcpy(&tmp_s, &pi, sizeof(tmp_s));
                            set_addr_cmd(tmp_s);
                            trigger_dram_read_cmd();
                            wait_idle_cmd();
                        }
                        // wait_idle_cmd();

                        set_mode_cmd(1, 9);
                        set_relu_cmd();
                        trigger_add_cmd();
                        wait_idle_cmd();
                        set_mode_cmd(0, 0);
                        // trigger_bn_cmd();
                        // wait_idle_cmd();
                        uint64_t pos = (ch*56*56) + (y*56) + x;
                        // in[pos] = read_max_pooling_cmd();
                        write_dram_value_cmd(&in[pos], read_max_pooling_cmd());
                        __asm__ volatile ("nop");

                    }
                }
            }
        }

        // for (uint64_t i = start; i < entry->base.out_size_; i++) {
        //     write_dram_value_cmd(&out[i], max(read_dram_value_cmd(&in[i]), 0));
        // }

        free(pad_ptr);
#ifdef PRINT_LAYER
    // if (hart_id == 0) 
    // {
    //     printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, (float_t)in[0], (float_t)in[1], (float_t)in[56*56*64-2], (float_t)in[56*56*64-1]);
    // }
#endif
    }
    else 
    for (uint64_t ch = 0; ch < in_.depth_; ch++){
        my_float_t mul = _gamma(entry, ch) * _invstd(entry, ch);
        my_float_t add = _gamma(entry, ch) * (- _mean(entry, ch)) * _invstd(entry, ch) + _beta(entry, ch);
        send_bn_mul_data(mul, 0);
        send_bn_add_data(add, 0);
        int max_len = 100;
        my_float_t *pi = in + (ch * dim);
        for (uint64_t j = start; j < end; j += max_len) {
            int remain_len = min(max_len, end - j);
            /*
             * send data
             */
            reset_sram_offset_cmd();
            set_length_cmd(remain_len);
            set_dram_read_input_cmd();
            uint32_t tmp_s;
            memcpy(&tmp_s, &pi, sizeof(tmp_s));
            set_addr_cmd(tmp_s);
            trigger_dram_read_cmd();
            wait_idle_cmd();
            /*
             * start BatchNorm
             */
            my_float_t actv_type[2] = {-1, -1};
            set_mode_cmd(1, remain_len);
            // check activation function is identity or relu
            if(entry->base.activate(actv_type, 0, entry->base.out_size_) == 1) {
                set_relu_cmd();
            }
            else {
                reset_relu_cmd();
            }
            
            trigger_add_cmd();
            wait_idle_cmd();
            set_mode_cmd(0, 0);

            /*
             * test read
             */
            // printf("data: %f, %f", (float)read_data_cmd(0, 0), (float)read_data_cmd(1, 0));
            // exit(1);
            
            /*
             * write data
             */
            // error max len once is 100
            set_dram_write_lens_cmd(remain_len);
            set_num_lans_cmd(0);
            set_output_recv_cnt_cmd(0);
            memcpy(&tmp_s, &pi, sizeof(tmp_s));
            set_dram_write_addr_cmd(0, tmp_s);
            set_dram_w_tr_cmd();
            wait_idle_cmd();

            pi += remain_len;
            // int ppa = 0, ppb = 0;
            // for(int i = 0; i < remain_len; i++) {
            //     my_float_t tmpp = read_data_cmd(ppa++, ppb);
            //     write_dram_value_cmd(pi++, tmpp);
            //     if(ppa == 4) {
            //         ppa = 0;
            //         ppb++;
            //     }
            // }
            
            __asm__ volatile ("nop");
        
#ifdef USING_GEM5
            clock_t  tmp_tick = clock();
#endif
            // trigger_bn_cmd();
            // wait_idle_cmd();
#ifdef USING_GEM5
            hardware_compute_time += (clock() - tmp_tick)/(ticks_per_msec/1000);
            // printf("It took %ld msec to perform on HW.\n\n", hardware_compute_time);
#endif
            // int var[16] = {0, 4, 8 , 12,
            //                1, 5, 9 , 13,
            //                2, 6, 10, 14,
            //                3, 7, 11, 15};
            // pos = ch * dim + j;
            // my_float_t t = entry->base.activate(in, 0, entry->base.out_size_);
            // for(int i = 0; i < remain_len; i++) {
            //     // my_float_t ff = read_data_cmd(var[i%16], i/16);
            //     // in[pos + i] = ff;
            //     // in[pos++] = read_data_cmd(var[i%16], i/16);
            //     // if(t == 0) {
            //     write_dram_value_cmd(&in[pos++], read_data_cmd(var[i%16], i/16));
            //     // }
            //     // else {
            //     //     write_dram_value_cmd(&in[pos++], max(0, read_data_cmd(var[i%16], i/16)));
            //     // }
                
            //     // in[pos + i] = ff;
            // }
        }
    }

    total_size = entry->base.out_size_;
    blocksize = compute_block_size(total_size);
    start = (blocksize) * hart_id;
    end = min((blocksize) * (hart_id+1), total_size);

    // for (uint64_t i = start; i < end; i++) {
    //     // out[i] = entry->base.activate(in, i, entry->base.out_size_);
    //     write_dram_value_cmd(&out[i], entry->base.activate(in, i, entry->base.out_size_));
    // }


    // for (uint64_t i = start; i < end; i++)
    //     out[i] = in[i];
    // free(entry->base.padded_ptr);
    // free(in);

    //###########################################################
    // print output tensor
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
    //             for (uint64_t w = 0; w < in_.width_; w++) {
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
#ifdef PRINT_LAYER
    if (hart_id == 0) 
    {
        // printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, (float_t)out[0], (float_t)out[1], (float_t)out[entry->base.out_size_-2], (float_t)out[entry->base.out_size_-1]);
        printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, (float_t)read_dram_value_cmd(&out[0]), (float_t)read_dram_value_cmd(&out[1]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-2]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-1]));
    }
#endif

#ifdef USING_GEM5
    tick = (clock() - tick)/ticks_per_msec;
    printf("It took %ld msec to perform batchNorm.\n\n", tick);
    printf("It took %ld msec to perform BN_HW.\n\n", hardware_compute_time);
#endif

}


layer_base * new_batchnorm_layer(
                                cnn_controller *ctrl,
                                my_float_t(*activate) (my_float_t *, uint64_t, uint64_t),
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