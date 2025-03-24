#pragma once

#include "list.h"
#include "layer.h"
#include "util.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

unsigned int _hart_id;

typedef struct _network
{
    uint64_t mask;
    uint64_t done_flag;
    struct list_node layers;
}network;

static inline void init_network(network *net)
{
    _hart_id = 0;
    // net->mask = (total_CPUs < 64) ? (1LL << total_CPUs) - 1 : 0LL - 1;
    if (_hart_id == 0)
    {
        printf("hart id 0:\n");
        net->layers.next = &net->layers;
        net->layers.prev = &net->layers;
    }
}

void predict(network *net, void *input_base, uint64_t size)
{
    while(net->done_flag != net->mask);
    struct list_node * pos;
    input_struct prev_output = { input_base, size };
    list_for_each(pos, &net->layers)
    {
        layer_base *tmp = list_entry(pos, layer_base, list);
        tmp->forward_propagation(pos, _hart_id, &prev_output);
        // prev_output.in_ptr_ = tmp->out_ptr_;
        // prev_output.in_size_ = tmp->out_size_;
    }
}