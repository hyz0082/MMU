#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "config.h"
#include "layer.h"
#include "list.h"
#include "util.h"
#include "activation_function.h"

#ifndef USING_GEM5
#include "loader.h"
#endif

typedef struct _dummy_head_layer
{
    layer_base base;
} dummy_head_layer;

dummy_head_layer * get_dummy_head_layer_entry(struct list_node *ptr)
{
    return list_entry(ptr, dummy_head_layer, base.list);
}

void dummy_head_layer_forward_propagation(struct list_node *ptr, unsigned int hart_id, input_struct *input) {}


layer_base * new_dummy_head_layer(
                                    cnn_controller *ctrl,
                                    my_float_t(*activate) (my_float_t *, uint64_t, uint64_t),
                                    uint64_t image_size
                                    )
{

// #ifndef USING_GEM5
    dummy_head_layer *ret = (dummy_head_layer *)malloc(sizeof(dummy_head_layer));
// #else
//     dummy_head_layer *ret = (dummy_head_layer *) ctrl->nwk_cur_ptr;
//     ctrl->nwk_cur_ptr += sizeof(dummy_head_layer);
// #endif
    ret->base.out_size_ = image_size;

#ifndef USING_GEM5
    ret->base.out_ptr_ = (my_float_t *)ctrl->lyr_cur_ptr;
#else
    ret->base.out_ptr_ = (my_float_t *)ctrl->lyr_cur_ptr;
    ctrl->lyr_cur_ptr += ret->base.out_size_ * sizeof(my_float_t);
#endif
    ret->base.activate = activate;
    ret->base.forward_propagation = dummy_head_layer_forward_propagation;
    return &ret->base;
}