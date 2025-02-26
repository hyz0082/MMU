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
    float_t *in_ptr_;
    uint64_t in_size_;
} input_struct;

typedef struct _layer_base
{
    uint64_t mask;
    uint64_t a_done_flag;
    uint64_t done_flag;
    float_t (*activate) (float_t *, uint64_t, uint64_t);
    void (*forward_propagation) (struct list_node *, unsigned int, input_struct*);
    struct list_node* (*get_entry) ();
    uint64_t in_size_;
    uint64_t out_size_;
    uint64_t padding_size;
    uint8_t need_space_for_a;

    float_t *a_ptr_;
    float_t *padded_ptr;
    float_t *out_ptr_;

    float_t *_W;
    float_t *_b;
    struct list_node list;
#ifdef PRINT_LAYER
    char layer_name_[20];
#endif
} layer_base;


void init_layer(layer_base *layer, cnn_controller *ctrl, uint64_t in_dim, uint64_t out_dim, uint64_t weight_dim, uint64_t bias_dim, uint8_t need_space_for_a)
{
    layer->in_size_ = in_dim;
    layer->out_size_ = out_dim;
    layer->done_flag = 0;
    layer->a_done_flag = 0;
    // layer->mask = (1LL << ctrl->total_CPUs) - 1;
    layer->mask = (ctrl->total_CPUs < 64) ? (1LL << ctrl->total_CPUs) - 1 : 0LL - 1;

// #ifndef USING_GEM5
    if (ctrl->padding_size) {
        layer->padding_size = ctrl->padding_size;

        // layer->padded_ptr = (float_t *)malloc(ctrl->padding_size * sizeof(float_t));
        // if (layer->padded_ptr != NULL) { // Check if memory was allocated
        //     memset((void*)layer->padded_ptr, 0, ctrl->padding_size * sizeof(float_t));
        // }
        // else {
        //     printf("Error: Unable to allocate memory for layer->padded_ptr\n");
        //     // exit(1);
        // }
    }else
        layer->padded_ptr = (float_t *)ctrl->lyr_cur_ptr;


    if (need_space_for_a) 
        layer->need_space_for_a = need_space_for_a;
    // layer->a_ptr_ = (float_t *)malloc(out_dim * sizeof(float_t));
    // if (layer->a_ptr_ != NULL) { // Check if memory was allocated
    //     memset((void*)layer->a_ptr_, 0, out_dim * sizeof(float_t));
    // }
    // else {
    //     printf("Error: Unable to allocate memory for layer->a_ptr_\n");
    //     // exit(1);
    // }

    // if (need_space_for_a) {
    //     layer->out_ptr_ = (float_t *)malloc(out_dim * sizeof(float_t));
    //     if (layer->out_ptr_ != NULL) { // Check if memory was allocated
    //         memset((void*)layer->out_ptr_, 0, out_dim * sizeof(float_t));
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
//     layer->padded_ptr = (float_t *)ctrl->lyr_cur_ptr;
//     if (ctrl->padding_size)
//         ctrl->lyr_cur_ptr += ctrl->padding_size * sizeof(float_t);
    
//     layer->a_ptr_ = (float_t *)ctrl->lyr_cur_ptr;
//     if (need_space_for_a) ctrl->lyr_cur_ptr += layer->out_size_ * sizeof(float_t);
//     layer->out_ptr_ = (float_t *)ctrl->lyr_cur_ptr;
//     ctrl->lyr_cur_ptr += layer->out_size_ * sizeof(float_t);
// #endif

    layer->_W = (float_t *)ctrl->wgt_cur_ptr;
    ctrl->wgt_cur_ptr += weight_dim * sizeof(float_t);

    layer->_b = (float_t *)ctrl->wgt_cur_ptr;
    ctrl->wgt_cur_ptr += bias_dim * sizeof(float_t);
}