#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "config.h"
#include "layer.h"
#include "list.h"
#include "util.h"
#include "activation_function.h"
// #include <math.h>
#ifndef USING_GEM5
#include "loader.h"
#endif

#include <time.h>

typedef struct _residual_block_interface
{
    layer_base base;

    struct list_node path0;
    struct list_node path1;

} residual_block_interface;

residual_block_interface * get_residual_block_interface_entry(struct list_node *ptr)
{
    return list_entry(ptr, residual_block_interface, base.list);
}

void residual_block_interface_forward_propagation(struct list_node *ptr, unsigned int hart_id, input_struct *input)
{
    static int block_cnt = 0;
    residual_block_interface *entry = get_residual_block_interface_entry(ptr);

    ///////////////////////////////// first path /////////////////////////////////
    struct list_node * pos0;
    input_struct prev_output0 = { input->in_ptr_, input->in_size_ };
    list_for_each(pos0, &entry->path0)
    {
        layer_base *tmp = list_entry(pos0, layer_base, list);
        tmp->forward_propagation(pos0, _hart_id, &prev_output0);
    }

    layer_base *out0_entry;
    if (&entry->path0 == entry->path0.next)
        out0_entry = list_last_entry(ptr, layer_base, list);
    else
        out0_entry = list_last_entry(&entry->path0, layer_base, list);
    my_float_t* out0 = out0_entry->out_ptr_;

    ///////////////////////////////// first path /////////////////////////////////
    
    ///////////////////////////////// second path /////////////////////////////////
    struct list_node * pos1;
    input_struct prev_output1 = { input->in_ptr_, input->in_size_ };
    list_for_each(pos1, &entry->path1)
    {
        layer_base *tmp = list_entry(pos1, layer_base, list);
        tmp->forward_propagation(pos1, _hart_id, &prev_output1);
    }

    layer_base *out1_entry;
    if (&entry->path1 == entry->path1.next)
        out1_entry = list_last_entry(ptr, layer_base, list);
    else
        out1_entry = list_last_entry(&entry->path1, layer_base, list);
    my_float_t* out1 = out1_entry->out_ptr_;

    ///////////////////////////////// second path /////////////////////////////////

    if ((out0_entry->out_size_ != out1_entry->out_size_) ||
        (out0_entry->out_size_ != entry->base.out_size_) ||
        (out1_entry->out_size_ != entry->base.out_size_))
    {
        printf ("Residual block error: out size not match\n");
        exit(0);     
    }

    // // malloc a_ptr
    // entry->base.a_ptr_ = (my_float_t *)malloc(entry->base.out_size_ * sizeof(my_float_t));
    // if (entry->base.a_ptr_ != NULL) { // Check if memory was allocated
    //     memset((void*)entry->base.a_ptr_, 0, entry->base.out_size_ * sizeof(my_float_t));
    // }
    // else {
    //     printf("Error: Unable to allocate memory for entry->base.a_ptr_\n");
    //     // exit(1);
    // }

    // // if (entry->base.need_space_for_a){
    // //     entry->base.out_ptr_ = (my_float_t *)malloc(entry->base.out_size_ * sizeof(my_float_t));
    // //     if (entry->base.out_ptr_ != NULL) { // Check if memory was allocated
    // //         memset((void*)entry->base.out_ptr_, 0, entry->base.out_size_ * sizeof(my_float_t));
    // //     }
    // //     else {
    // //         printf("Error: Unable to allocate memory for entry->base.out_ptr_\n");
    // //         // exit(1);
    // //     }
    // // }
    // // else {
    //     entry->base.out_ptr_ = entry->base.a_ptr_;
    // // }

#ifdef USING_GEM5
    clock_t  tick, ticks_per_msec = CLOCKS_PER_SEC;
    tick = clock();
#endif

    entry->base.a_ptr_ = out0;
    entry->base.out_ptr_ = entry->base.a_ptr_;


    my_float_t *a = entry->base.a_ptr_;
    my_float_t *out = entry->base.out_ptr_;
    input->in_ptr_ = out;
    input->in_size_ = entry->base.out_size_;

    uint64_t total_size = entry->base.out_size_;
    uint64_t blocksize = compute_block_size(total_size);
    uint64_t start = (blocksize) * hart_id;
    uint64_t end = min((blocksize) * (hart_id+1), total_size);
    int max_len = 100;
    my_float_t *pi = out0;
    my_float_t *pj = out1;
    // my_float_t *po = out;
    int source = 1;// 0: sw, 1: hw
    /*
     * sw version
     */
    // 1353 + 6272
    if(!source)
    for (uint64_t i = start; i < end; i++)
    {
        my_float_t tmp_s = read_dram_value_cmd(&out0[i]) + read_dram_value_cmd(&out1[i]);
        if(tmp_s < (my_float_t)0) {
            tmp_s = 0;
        }
        // if(i >= (1353 + 6272 - 5) && i <= (1353 + 6272 + 5)) {
        //     printf("%d: %f = %f + %f\n",(int)i, (float)tmp_s, (float)read_dram_value_cmd(&out0[i]), (float)read_dram_value_cmd(&out1[i]));
        //     printf("%d: %x = %x + %x\n",(int)i, tmp_s, read_dram_value_cmd(&out0[i]), read_dram_value_cmd(&out1[i]));
        //     printf("\n");
        // }
        write_dram_value_cmd(&out[i], tmp_s);
    }
    /*
     * hw version
     */
    if(source)
    for (uint64_t i = start; i < end; i+=100)
    {
        int remain_len = min(max_len, end - i);
        /*
         * send out0
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
         * send out1
         */
        set_dram_read_weight_cmd();
        reset_sram_offset_cmd();
        set_length_cmd(remain_len);
        memcpy(&tmp_s, &pj, sizeof(tmp_s));
        set_addr_cmd(tmp_s);
        trigger_dram_read_cmd();
        wait_idle_cmd();

        /*
         * calc add
         */
        set_mode_cmd(2, remain_len);
        set_relu_cmd();
        trigger_add_cmd();
        wait_idle_cmd();
        set_mode_cmd(0, 0);
        /*
         * write data
         */
        set_dram_write_lens_cmd(remain_len);
        set_num_lans_cmd(0);
        set_output_recv_cnt_cmd(0);
        memcpy(&tmp_s, &pi, sizeof(tmp_s));
        set_dram_write_addr_cmd(0, tmp_s);
        set_dram_w_tr_cmd();
        wait_idle_cmd();

        pi += remain_len;
        pj += remain_len;
    }
   
    free(out1);

    //###########################################################
    // print output tensor
    // get size
    // struct list_node *p_tmp = &out0_entry->list;
    // batchnorm_layer *p_entry = get_batchnorm_layer_entry(p_tmp);

    // index3d in_ = p_entry->in_;

    // if(block_cnt == 0)
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
    // printf("\n\n[%s]\n", entry->base.layer_name_);
    // printf("shape depth:%d,  height:%d width:%d\n", (int)in_.depth_, (int)in_.height_, (int)in_.width_);
    // printf("    ");
    // for(int i = 0; i < in_.width_; i++) {
    //     printf("%6d ", (i));
    // }
    // printf("\n");
    // // const my_float_t *pi = &out[0];
    // int p_cnt = 0;
    // for (int inc = 0; inc < in_.depth_; inc++) {
    //     printf("\n[depth%d]\n", inc);
    //     // const my_float_t *pi = &in[get_index(&in_padded_, 0, 0, inc)];
    //     for (int h = 0; h < in_.height_; h++) {
    //         printf("%3d ", (h));
    //         printf("[ ");
    //         for (uint64_t w = 0; w < in_.width_; w++) {
    //             printf("%2.6f ", (out[p_cnt]));
    //             p_cnt++;
    //         }
    //         printf("]\n");
    //     }
    // }
    //###########################################################

    block_cnt++;
#ifdef PRINT_LAYER
    if (hart_id == 0) 
    {
        // printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, (float)out[0], (float)out[1], (float)out[entry->base.out_size_-2], (float)out[entry->base.out_size_-1]);
        printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, (float_t)read_dram_value_cmd(&out[0]), (float_t)read_dram_value_cmd(&out[1]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-2]), (float_t)read_dram_value_cmd(&out[entry->base.out_size_-1]));
    }
#endif

#ifdef USING_GEM5
    tick = (clock() - tick)/ticks_per_msec;
    printf("It took %ld msec to perform residual.\n\n", tick);
#endif
}


layer_base * new_residual_block_interface(
                                          cnn_controller *ctrl,
                                          my_float_t(*activate) (my_float_t *, uint64_t, uint64_t)
                                          )
{

// #ifndef USING_GEM5
    residual_block_interface *ret = (residual_block_interface *)malloc(sizeof(residual_block_interface));
// #else
//     residual_block_interface *ret = (residual_block_interface *) ctrl->nwk_cur_ptr;
//     ctrl->nwk_cur_ptr += sizeof(residual_block_interface);
// #endif

    
#ifdef PRINT_LAYER
    static uint64_t call_time = 0;
    // sprintf(ret->base.layer_name_, "residual%d", call_time++);
    my_sprintf(ret->base.layer_name_, "residual%d", call_time++);
#endif
    ret->path0.next = &ret->path0;
    ret->path0.prev = &ret->path0;

    ret->path1.next = &ret->path1;
    ret->path1.prev = &ret->path1;

    ret->base.activate = activate;
    // printf("insize of FC layer %d\n", ret->base.in_size_);
    // // printf("FC: in [%f, %f, ... , %f, %f]\n", ret->base.in_ptr_[0], ret->base.in_ptr_[1], ret->base.in_ptr_[ret->base.in_size_-2], ret->base.in_ptr_[ret->base.in_size_-1]);
    // printf("FC: W  [%f, %f, ... , %f, %f]\n", ret->base._W[0], ret->base._W[1], ret->base._W[in_dim * out_dim-2], ret->base._W[in_dim * out_dim-1]);
    // printf("FC: b  [%f, %f, ... , %f, %f]\n", ret->base._b[0], ret->base._b[1], ret->base._b[out_dim-2], ret->base._b[out_dim-1]);
    ret->base.forward_propagation = residual_block_interface_forward_propagation;
    return &ret->base;
}

void path_push_back(struct list_node *new_node, struct list_node *res, int sel)
{
    residual_block_interface *entry = get_residual_block_interface_entry(res);
    struct list_node *path_to_push;
    if (sel == 0)
        path_to_push = &entry->path0;
    else if (sel == 1)
        path_to_push = &entry->path1;
    else
    {
        printf("Unsupported path: %d\n", sel);
        exit(-1);
    }

    my_push_back(new_node, path_to_push);
}

void init_residual_block(struct list_node *res, cnn_controller *ctrl)
{
    residual_block_interface *entry = get_residual_block_interface_entry(res);
    struct list_node *in = res->prev;
    layer_base *in_entry = list_entry(in, layer_base, list);

    // if (entry->path0.next != &entry->path0 && entry->path1.next != &entry->path1)
    // {
    //     printf("Error: both path in residual block are empty\n");
    //     exit(-1);
    // }

    // // set up input address of first path
    // if (entry->path0.next != &entry->path0)
    // {
    //     layer_base *first_entry = list_first_entry(&entry->path0, layer_base, list);
    //     first_entry->in_ptr_ = in_entry->out_ptr_;
    // }


    // // set up input address of second path
    // if (entry->path1.next != &entry->path1)
    // {
    //     layer_base *first_entry = list_first_entry(&entry->path1, layer_base, list);
    //     first_entry->in_ptr_ = in_entry->out_ptr_;
    // }

    // check output dim
    uint64_t out_dim;
    if (entry->path0.next == &entry->path0 && entry->path1.next == &entry->path1) // both path are empty
    {
        out_dim = in_entry->out_size_;
    }
    else if (entry->path0.next == &entry->path0) // path 0 empty
    {
        layer_base *last_entry = list_last_entry(&entry->path1, layer_base, list);
        if (last_entry->out_size_ != in_entry->out_size_)
        {
            printf("Error: residual block output size not match\n");
            exit(-1);
        }
        out_dim = last_entry->out_size_;
    }
    else if (entry->path1.next == &entry->path1) // path 1 empty
    {
        layer_base *last_entry = list_last_entry(&entry->path0, layer_base, list);
        if (last_entry->out_size_ != in_entry->out_size_)
        {
            printf("Error: residual block output size not match %d/%d\n" ,last_entry->out_size_, in_entry->out_size_);
            exit(-1);
        }
        out_dim = last_entry->out_size_;
    }
    else
    {
        layer_base *last_entry1 = list_last_entry(&entry->path1, layer_base, list);
        layer_base *last_entry0 = list_last_entry(&entry->path0, layer_base, list);
        if (last_entry1->out_size_ != last_entry0->out_size_)
        {
            printf("Error: residual block output size not match\n");
            exit(-1);
        }
        out_dim = last_entry1->out_size_;
    }

    init_layer(&entry->base,
               ctrl,
               0,
               out_dim,
               0,
               0,
               entry->base.activate==softmax);
}