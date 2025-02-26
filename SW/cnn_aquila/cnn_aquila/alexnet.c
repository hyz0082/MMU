#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "config.h"
#include "list.h"
#include "layer.h"
#include "activation_function.h"
#include "fully_connected_layer.h"
#include "average_pooling_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "max_pooling_layer.h"
#include "network.h"
#include "dummy_head_layer.h"
#include "class_name.h"

#ifndef USING_GEM5
#include "loader.h"
#endif

int compare(const void *arg1, const void *arg2)
{
    return ((output_index_name *)arg1)->value < ((output_index_name *)arg2)->value;
}

void recognize(unsigned int total_CPUs, unsigned int hart_id)
{
#ifndef USING_GEM5
    cnn_controller ctrl = { (void *)0, input_base, weight_base, 0, total_CPUs };
    void *input_base_ = input_base;
    network *nn = (network *) malloc(sizeof(network));
#else
    cnn_controller ctrl = { (void *)0xC000000000000000, (void *)0xC100000000000000, (void *)0xC200000000000000, 0, total_CPUs };
    void *input_base_ = (void *)0xC100000000000000;
    network *nn = (network *)ctrl.nwk_cur_ptr;
    ctrl.nwk_cur_ptr += sizeof(network);
#endif    
    init_network(nn, total_CPUs, hart_id);

    uint64_t image_size = 224*224*3;
    if (hart_id == 0)
    {
        push_back(&new_dummy_head_layer(&ctrl, identity, image_size)->list, &nn->layers);
        push_back(&new_convolutional_layer(&ctrl, relu, 224, 224, 11, 11, 3, 64, same, 1, 4, 4, 2, 2)->list, &nn->layers);
        push_back(&new_max_pooling_layer(&ctrl, identity, 55, 55, 64, 3, 2, 0)->list, &nn->layers);

        push_back(&new_convolutional_layer(&ctrl, relu, 27, 27, 5, 5, 64, 192, same, 1, 1, 1, 5/2, 5/2)->list, &nn->layers);
        push_back(&new_max_pooling_layer(&ctrl, identity, 27, 27, 192, 3, 2, 0)->list, &nn->layers);


        push_back(&new_convolutional_layer(&ctrl, relu, 13, 13, 3, 3, 192, 384, same, 1, 1, 1, 3/2, 3/2)->list, &nn->layers);

        push_back(&new_convolutional_layer(&ctrl, relu, 13, 13, 3, 3, 384, 256, same, 1, 1, 1, 3/2, 3/2)->list, &nn->layers);

        push_back(&new_convolutional_layer(&ctrl, relu, 13, 13, 3, 3, 256, 256, same, 1, 1, 1, 3/2, 3/2)->list, &nn->layers);
        push_back(&new_max_pooling_layer(&ctrl, identity, 13, 13, 256, 3, 2, 0)->list, &nn->layers);

        push_back(&new_fully_connected_layer(&ctrl, relu, 9216, 4096, 1)->list, &nn->layers);
        push_back(&new_fully_connected_layer(&ctrl, relu, 4096, 4096, 1)->list, &nn->layers);
        push_back(&new_fully_connected_layer(&ctrl, softmax, 4096, 1000, 1)->list, &nn->layers);
        
        printf("nwk_cur_ptr: %llx\n", ctrl.nwk_cur_ptr);
        printf("lyr_cur_ptr: %llx\n", ctrl.lyr_cur_ptr);
        printf("wgt_cur_ptr: %llx\n", ctrl.wgt_cur_ptr);
        nn->done_flag = nn->mask;
    }
    predict(nn, input_base_, image_size);
    if (hart_id == 0)
    {
        printf ("Predict done\n");
        layer_base *tmp = list_last_entry(&nn->layers, layer_base, list);
        float_t* res = tmp->out_ptr_;
#ifndef USING_GEM5
        char *class_name = (char *) class_name_base;
#else
        char *class_name = (char *) 0xC300000000000000;
#endif
        output_index_name out[1000];
        for (int i = 0; i < 1000; i++)
        {
            out[i].index = i;
            out[i].value = res[i];
            mmap_fgets(out[i].name, sizeof(out[i].name), &class_name);
        }
        qsort((void *)out, 1000, sizeof(output_index_name), compare);
        printf("==============================================\n");
        printf(" idx | possibility(%) | class name\n");
        printf("----------------------------------------------\n");
        for (int i = 0; i < 4; i++)
            printf(" %3d |       %8.5f | %s\n", out[i].index, out[i].value, out[i].name);
        printf("==============================================\n");
        
    }
    
}

int main(int argc, char** argv)
{
#ifndef USING_GEM5
    if (argc < 6)
    {
        fprintf(stderr, "Usage: %s <weight_file> <image_to_inference> <class_name> <total_CPUs> <hart_id>\n", argv[0]);
        exit(-1);
    }
    init_loader(argv[1], argv[2]);
    init_class_name(argv[3]);
    unsigned int total_cpu = atoi(argv[4]);
    unsigned int hart_id = atoi(argv[5]);
#else
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s <total_CPUs> <hart_id>\n", argv[0]);
        exit(-1);
    }
    unsigned int total_cpu = atoi(argv[1]);
    unsigned int hart_id = atoi(argv[2]);
#endif
    init_util(total_cpu);
    recognize(total_cpu, hart_id);

    return 0;
}