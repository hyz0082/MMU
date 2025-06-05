#pragma once

#include <stdint.h>
#include "config.h"
#include "list.h"
#include "activation_function.h"

#include <string.h>

#ifndef USING_GEM5
#include "loader.h"
#endif

typedef struct _input_struct
{
    my_float_t *in_ptr_;
    uint64_t in_size_;
} input_struct;

typedef struct _layer_base
{
    uint64_t mask;
    uint64_t a_done_flag;
    uint64_t done_flag;
    my_float_t (*activate) (my_float_t *, uint64_t, uint64_t);
    void (*forward_propagation) (struct list_node *, unsigned int, input_struct*);
    struct list_node* (*get_entry) ();
    uint64_t in_size_;
    uint64_t out_size_;
    uint64_t padding_size;
    uint8_t need_space_for_a;

    uint64_t weight_offset;

    my_float_t *a_ptr_;
    my_float_t *padded_ptr;
    my_float_t *out_ptr_;

    my_float_t *_W;
    my_float_t *_b;
    struct list_node list;
#ifdef PRINT_LAYER
    char layer_name_[20];
#endif
} layer_base;


void init_layer(layer_base *layer, cnn_controller *ctrl, uint64_t in_dim, uint64_t out_dim, uint64_t weight_dim, uint64_t bias_dim, uint8_t need_space_for_a)
{
    static uint64_t offset = 0;
    layer->in_size_ = in_dim;
    layer->out_size_ = out_dim;
    layer->done_flag = 0;
    layer->a_done_flag = 0;
    // layer->mask = (1LL << ctrl->total_CPUs) - 1;
    layer->mask = (ctrl->total_CPUs < 64) ? (1LL << ctrl->total_CPUs) - 1 : 0LL - 1;

// #ifndef USING_GEM5
    if (ctrl->padding_size) {
        layer->padding_size = ctrl->padding_size;

        // layer->padded_ptr = (my_float_t *)malloc(ctrl->padding_size * sizeof(my_float_t));
        // if (layer->padded_ptr != NULL) { // Check if memory was allocated
        //     memset((void*)layer->padded_ptr, 0, ctrl->padding_size * sizeof(my_float_t));
        // }
        // else {
        //     printf("Error: Unable to allocate memory for layer->padded_ptr\n");
        //     // exit(1);
        // }
    }else
        layer->padded_ptr = (my_float_t *)ctrl->lyr_cur_ptr;


    if (need_space_for_a) 
        layer->need_space_for_a = need_space_for_a;
    // layer->a_ptr_ = (my_float_t *)malloc(out_dim * sizeof(my_float_t));
    // if (layer->a_ptr_ != NULL) { // Check if memory was allocated
    //     memset((void*)layer->a_ptr_, 0, out_dim * sizeof(my_float_t));
    // }
    // else {
    //     printf("Error: Unable to allocate memory for layer->a_ptr_\n");
    //     // exit(1);
    // }

    // if (need_space_for_a) {
    //     layer->out_ptr_ = (my_float_t *)malloc(out_dim * sizeof(my_float_t));
    //     if (layer->out_ptr_ != NULL) { // Check if memory was allocated
    //         memset((void*)layer->out_ptr_, 0, out_dim * sizeof(my_float_t));
    //     }
    //     else {
    //         printf("Error: Unable to allocate memory for layer->out_ptr_\n");
    //         // exit(1);
    //     }
    // }
    // else 
    //     layer->out_ptr_ = layer->a_ptr_;
    
    ctrl->lyr_cur_ptr = (void *)layer->out_ptr_;
// #else
//     layer->padded_ptr = (my_float_t *)ctrl->lyr_cur_ptr;
//     if (ctrl->padding_size)
//         ctrl->lyr_cur_ptr += ctrl->padding_size * sizeof(my_float_t);
    
//     layer->a_ptr_ = (my_float_t *)ctrl->lyr_cur_ptr;
//     if (need_space_for_a) ctrl->lyr_cur_ptr += layer->out_size_ * sizeof(my_float_t);
//     layer->out_ptr_ = (my_float_t *)ctrl->lyr_cur_ptr;
//     ctrl->lyr_cur_ptr += layer->out_size_ * sizeof(my_float_t);
// #endif
    layer->weight_offset = offset;
    offset += weight_dim;
    // printf("offset: %d\n",layer->weight_offset );
    layer->_W = (my_float_t *)ctrl->wgt_cur_ptr;
    // printf("val: %f\n", (float)layer->_W[0]);
    ctrl->wgt_cur_ptr += weight_dim * sizeof(my_float_t);
    // printf("%d\n", sizeof(my_float_t));
    layer->_b = (my_float_t *)ctrl->wgt_cur_ptr;
    ctrl->wgt_cur_ptr += bias_dim * sizeof(my_float_t);
    // printf("val: %f\n", (float)layer->_W[0]);
}