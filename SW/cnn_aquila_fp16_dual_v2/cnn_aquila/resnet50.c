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
#include "residual_block_interface.h"
// #include "class_name.h"
#include "./inc_cnn/util.h"
#include "file_read.h"
#include "imagenet_classes_name.h"
// #include <math.h>
#include <stddef.h>

#include "./inc_cnn/global_var.h"

// #ifndef USING_GEM5
// #include "loader.h"
// #endif


int compare(const void *arg1, const void *arg2)
{
    return ((output_index_name *)arg1)->value < ((output_index_name *)arg2)->value;
}

// my_float_t *weights, *inputs;

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
#endif

    unsigned int total_cpu = 1;
    unsigned int hart_id = 0;

    clock_t  tick, ticks_per_msec = CLOCKS_PER_SEC;

    printf("start read input \n");
    inputs = read_weights("fp16_Samoyed.bin");
    printf("inputs[0]: %f %f %f\n", inputs[0], inputs[1], inputs[2]);
    printf("read input done\n");

    tick = clock();
    printf("start read weights \n");
    weights = read_weights("fp16_resnet50_Weight");
    printf("weights[0]: %f %f %f\n", weights[0], weights[1], weights[2]);
    printf("read weights done\n");
    tick = (clock() - tick)/ticks_per_msec;
    printf("It took %ld msec to read files from the SD card.\n\n", tick);

    for(int i = 0; i < 1000000; i++) {
        if(weights[i] == 777) {
            printf(" ");
        }
    }
    printf("\n");
    for(int i = 0; i < 1000000; i++) {
        if(weights[i] == 777) {
            printf(".");
        }
    }
    printf("\n");
    // printf("start read input \n");
    // inputs = read_weights("fp16_Samoyed.bin");
    // printf("inputs[0]: %f %f %f\n", inputs[0], inputs[1], inputs[2]);
    // printf("read input done\n");

    // tick = clock();
    // printf("start read weights \n");
    // weights = read_weights("fp16_resnet50_Weight");
    // printf("weights[0]: %f %f %f\n", weights[0], weights[1], weights[2]);
    // printf("read weights done\n");
    // tick = (clock() - tick)/ticks_per_msec;
    // printf("It took %ld msec to read files from the SD card.\n\n", tick);

    init_util(total_cpu);

    network *nn;
    cnn_controller ctrl;

    nn = (network *) malloc(sizeof(network));
    if(nn == NULL) {
        printf("malloc nn error\n");
        exit(1);
    }

    memset((void *) &ctrl, 0, sizeof(cnn_controller));
    ctrl.wgt_cur_ptr = weights;

    ctrl.total_CPUs = 1;

    init_network(nn);

    // construct resnet50
    int image_size = 224*224*3;
    if (hart_id == 0)
    {
        my_push_back(&new_dummy_head_layer(&ctrl, identity, image_size)->list, &nn->layers);
        // 0
        my_push_back(&new_convolutional_layer(&ctrl, identity, 224, 224, 7, 7, 3, 64, same, 0, 2, 2, 7/2, 7/2, 0)->list, &nn->layers);
        my_push_back(&new_batchnorm_layer(&ctrl, relu, 64, 112, 112)->list, &nn->layers);
        my_push_back(&new_max_pooling_layer(&ctrl, identity, 112, 112, 64, 3, 2, 1)->list, &nn->layers);

        //////////////////////////////////////// bottleneck 1_1 ////////////////////////////////////////
        {
            my_push_back(&new_residual_block_interface(&ctrl, relu)->list, &nn->layers);
            struct list_node *res = nn->layers.prev;
            ////////////////////////////////////// path0 //////////////////////////////////////
            // 1
            path_push_back(&new_convolutional_layer(&ctrl, identity, 56, 56, 1, 1, 64, 64, same, 0, 1, 1, 1/2, 1/2, 0)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 64, 56, 56)->list, res, 0);
            // 2
            path_push_back(&new_convolutional_layer(&ctrl, identity, 56, 56, 3, 3, 64, 64, same, 0, 1, 1, 3/2, 3/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 64, 56, 56)->list, res, 0);
            // 3
            path_push_back(&new_convolutional_layer(&ctrl, identity, 56, 56, 1, 1, 64, 256, same, 0, 1, 1, 1/2, 1/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 256, 56, 56)->list, res, 0);
            ////////////////////////////////////// path0 //////////////////////////////////////
            ////////////////////////////////////// path1 //////////////////////////////////////
            // 4
            path_push_back(&new_convolutional_layer(&ctrl, identity, 56, 56, 1, 1, 64, 256, same, 0, 1, 1, 1/2, 1/2, 1)->list, res, 1);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 256, 56, 56)->list, res, 1);
            ////////////////////////////////////// path1 //////////////////////////////////////
            init_residual_block(res, &ctrl);
        }
        //////////////////////////////////////// bottleneck 1_1 ////////////////////////////////////////

        //////////////////////////////////////// bottleneck 2_1 ////////////////////////////////////////
        {
            my_push_back(&new_residual_block_interface(&ctrl, relu)->list, &nn->layers);
            struct list_node *res = nn->layers.prev;
            ////////////////////////////////////// path0 //////////////////////////////////////
            // 5
            path_push_back(&new_convolutional_layer(&ctrl, identity, 56, 56, 1, 1, 256, 64, same, 0, 1, 1, 1/2, 1/2, 0)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 64, 56, 56)->list, res, 0);
            // 6
            path_push_back(&new_convolutional_layer(&ctrl, identity, 56, 56, 3, 3, 64, 64, same, 0, 1, 1, 3/2, 3/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 64, 56, 56)->list, res, 0);
            // 7
            path_push_back(&new_convolutional_layer(&ctrl, identity, 56, 56, 1, 1, 64, 256, same, 0, 1, 1, 1/2, 1/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 256, 56, 56)->list, res, 0);
            ////////////////////////////////////// path0 //////////////////////////////////////
            init_residual_block(res, &ctrl);
        }
        //////////////////////////////////////// bottleneck 2_1 ////////////////////////////////////////

        //////////////////////////////////////// bottleneck 2_2 ////////////////////////////////////////
        {
            my_push_back(&new_residual_block_interface(&ctrl, relu)->list, &nn->layers);
            struct list_node *res = nn->layers.prev;
            ////////////////////////////////////// path0 //////////////////////////////////////
            // 8
            path_push_back(&new_convolutional_layer(&ctrl, identity, 56, 56, 1, 1, 256, 64, same, 0, 1, 1, 1/2, 1/2, 0)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 64, 56, 56)->list, res, 0);
            // 9
            path_push_back(&new_convolutional_layer(&ctrl, identity, 56, 56, 3, 3, 64, 64, same, 0, 1, 1, 3/2, 3/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 64, 56, 56)->list, res, 0);
            // 10
            path_push_back(&new_convolutional_layer(&ctrl, identity, 56, 56, 1, 1, 64, 256, same, 0, 1, 1, 1/2, 1/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 256, 56, 56)->list, res, 0);
            ////////////////////////////////////// path0 //////////////////////////////////////
            init_residual_block(res, &ctrl);
        }
        //////////////////////////////////////// bottleneck 2_2 ////////////////////////////////////////

        //////////////////////////////////////// bottleneck 1_2 ////////////////////////////////////////
        {
            my_push_back(&new_residual_block_interface(&ctrl, relu)->list, &nn->layers);
            struct list_node *res = nn->layers.prev;
            ////////////////////////////////////// path0 //////////////////////////////////////
            // 11
            path_push_back(&new_convolutional_layer(&ctrl, identity, 56, 56, 1, 1, 256, 128, same, 0, 1, 1, 1/2, 1/2, 0)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 128, 56, 56)->list, res, 0);
            // 12
            path_push_back(&new_convolutional_layer(&ctrl, identity, 56, 56, 3, 3, 128, 128, same, 0, 2, 2, 3/2, 3/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 128, 28, 28)->list, res, 0);
            // 13
            path_push_back(&new_convolutional_layer(&ctrl, identity, 28, 28, 1, 1, 128, 512, same, 0, 1, 1, 1/2, 1/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 512, 28, 28)->list, res, 0);
            ////////////////////////////////////// path0 //////////////////////////////////////
            ////////////////////////////////////// path1 //////////////////////////////////////
            // 14
            path_push_back(&new_convolutional_layer(&ctrl, identity, 56, 56, 1, 1, 256, 512, same, 0, 2, 2, 1/2, 1/2, 1)->list, res, 1);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 512, 28, 28)->list, res, 1);
            ////////////////////////////////////// path2 //////////////////////////////////////
            init_residual_block(res, &ctrl);
        }
        //////////////////////////////////////// bottleneck 1_2 ////////////////////////////////////////

        //////////////////////////////////////// bottleneck 2_3 ////////////////////////////////////////
        {
            my_push_back(&new_residual_block_interface(&ctrl, relu)->list, &nn->layers);
            struct list_node *res = nn->layers.prev;
            ////////////////////////////////////// path0 //////////////////////////////////////
            // 15
            path_push_back(&new_convolutional_layer(&ctrl, identity, 28, 28, 1, 1, 512, 128, same, 0, 1, 1, 1/2, 1/2, 0)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 128, 28, 28)->list, res, 0);
            // 16
            path_push_back(&new_convolutional_layer(&ctrl, identity, 28, 28, 3, 3, 128, 128, same, 0, 1, 1, 3/2, 3/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 128, 28, 28)->list, res, 0);
            // 17
            path_push_back(&new_convolutional_layer(&ctrl, identity, 28, 28, 1, 1, 128, 512, same, 0, 1, 1, 1/2, 1/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 512, 28, 28)->list, res, 0);
            ////////////////////////////////////// path0 //////////////////////////////////////
            init_residual_block(res, &ctrl);
        }
        //////////////////////////////////////// bottleneck 2_3 ////////////////////////////////////////

        //////////////////////////////////////// bottleneck 2_4 ////////////////////////////////////////
        {
            my_push_back(&new_residual_block_interface(&ctrl, relu)->list, &nn->layers);
            struct list_node *res = nn->layers.prev;
            ////////////////////////////////////// path0 //////////////////////////////////////
            // 18
            path_push_back(&new_convolutional_layer(&ctrl, identity, 28, 28, 1, 1, 512, 128, same, 0, 1, 1, 1/2, 1/2, 0)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 128, 28, 28)->list, res, 0);
            // 19
            path_push_back(&new_convolutional_layer(&ctrl, identity, 28, 28, 3, 3, 128, 128, same, 0, 1, 1, 3/2, 3/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 128, 28, 28)->list, res, 0);
            // 20
            path_push_back(&new_convolutional_layer(&ctrl, identity, 28, 28, 1, 1, 128, 512, same, 0, 1, 1, 1/2, 1/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 512, 28, 28)->list, res, 0);
            ////////////////////////////////////// path0 //////////////////////////////////////
            init_residual_block(res, &ctrl);
        }
        //////////////////////////////////////// bottleneck 2_4 ////////////////////////////////////////

        //////////////////////////////////////// bottleneck 2_5 ////////////////////////////////////////
        {
            my_push_back(&new_residual_block_interface(&ctrl, relu)->list, &nn->layers);
            struct list_node *res = nn->layers.prev;
            ////////////////////////////////////// path0 //////////////////////////////////////
            // 21
            path_push_back(&new_convolutional_layer(&ctrl, identity, 28, 28, 1, 1, 512, 128, same, 0, 1, 1, 1/2, 1/2, 0)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 128, 28, 28)->list, res, 0);
            // 22
            path_push_back(&new_convolutional_layer(&ctrl, identity, 28, 28, 3, 3, 128, 128, same, 0, 1, 1, 3/2, 3/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 128, 28, 28)->list, res, 0);
            // 23
            path_push_back(&new_convolutional_layer(&ctrl, identity, 28, 28, 1, 1, 128, 512, same, 0, 1, 1, 1/2, 1/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 512, 28, 28)->list, res, 0);
            ////////////////////////////////////// path0 //////////////////////////////////////
            init_residual_block(res, &ctrl);
        }
        //////////////////////////////////////// bottleneck 2_5 ////////////////////////////////////////

        //////////////////////////////////////// bottleneck 1_3 ////////////////////////////////////////
        {
            my_push_back(&new_residual_block_interface(&ctrl, relu)->list, &nn->layers);
            struct list_node *res = nn->layers.prev;
            ////////////////////////////////////// path0 //////////////////////////////////////
            // 24
            path_push_back(&new_convolutional_layer(&ctrl, identity, 28, 28, 1, 1, 512, 256, same, 0, 1, 1, 1/2, 1/2, 0)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 256, 28, 28)->list, res, 0);
            // 25
            path_push_back(&new_convolutional_layer(&ctrl, identity, 28, 28, 3, 3, 256, 256, same, 0, 2, 2, 3/2, 3/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 256, 14, 14)->list, res, 0);
            // 26
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 1, 1, 256, 1024, same, 0, 1, 1, 1/2, 1/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 1024, 14, 14)->list, res, 0);
            ////////////////////////////////////// path0 //////////////////////////////////////
            ////////////////////////////////////// path1 //////////////////////////////////////
            // 27
            path_push_back(&new_convolutional_layer(&ctrl, identity, 28, 28, 1, 1, 512, 1024, same, 0, 2, 2, 1/2, 1/2, 1)->list, res, 1);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 1024, 14, 14)->list, res, 1);
            ////////////////////////////////////// path2 //////////////////////////////////////
            init_residual_block(res, &ctrl);
        }
        //////////////////////////////////////// bottleneck 1_3 ////////////////////////////////////////
        // 28 ~ 30
        //////////////////////////////////////// bottleneck 2_6 ////////////////////////////////////////
        {
            my_push_back(&new_residual_block_interface(&ctrl, relu)->list, &nn->layers);
            struct list_node *res = nn->layers.prev;
            ////////////////////////////////////// path0 //////////////////////////////////////
            // 28
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 1, 1, 1024, 256, same, 0, 1, 1, 1/2, 1/2, 0)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 256, 14, 14)->list, res, 0);
            // 29
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 3, 3, 256, 256, same, 0, 1, 1, 3/2, 3/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 256, 14, 14)->list, res, 0);
            // 30 
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 1, 1, 256, 1024, same, 0, 1, 1, 1/2, 1/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 1024, 14, 14)->list, res, 0);
            ////////////////////////////////////// path0 //////////////////////////////////////
            init_residual_block(res, &ctrl);
        }
        //////////////////////////////////////// bottleneck 2_6 ////////////////////////////////////////

        //////////////////////////////////////// bottleneck 2_7 ////////////////////////////////////////
        {
            my_push_back(&new_residual_block_interface(&ctrl, relu)->list, &nn->layers);
            struct list_node *res = nn->layers.prev;
            ////////////////////////////////////// path0 //////////////////////////////////////
            // 31
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 1, 1, 1024, 256, same, 0, 1, 1, 1/2, 1/2, 0)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 256, 14, 14)->list, res, 0);
            // 32
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 3, 3, 256, 256, same, 0, 1, 1, 3/2, 3/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 256, 14, 14)->list, res, 0);
            // 33
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 1, 1, 256, 1024, same, 0, 1, 1, 1/2, 1/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 1024, 14, 14)->list, res, 0);
            ////////////////////////////////////// path0 //////////////////////////////////////
            init_residual_block(res, &ctrl);
        }
        //////////////////////////////////////// bottleneck 2_7 ////////////////////////////////////////

        //////////////////////////////////////// bottleneck 2_8 ////////////////////////////////////////
        {
            my_push_back(&new_residual_block_interface(&ctrl, relu)->list, &nn->layers);
            struct list_node *res = nn->layers.prev;
            ////////////////////////////////////// path0 //////////////////////////////////////
            // 34
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 1, 1, 1024, 256, same, 0, 1, 1, 1/2, 1/2, 0)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 256, 14, 14)->list, res, 0);
            // 35
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 3, 3, 256, 256, same, 0, 1, 1, 3/2, 3/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 256, 14, 14)->list, res, 0);
            // 36
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 1, 1, 256, 1024, same, 0, 1, 1, 1/2, 1/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 1024, 14, 14)->list, res, 0);
            ////////////////////////////////////// path0 //////////////////////////////////////
            init_residual_block(res, &ctrl);
        }
        //////////////////////////////////////// bottleneck 2_8 ////////////////////////////////////////

        //////////////////////////////////////// bottleneck 2_9 ////////////////////////////////////////
        {
            my_push_back(&new_residual_block_interface(&ctrl, relu)->list, &nn->layers);
            struct list_node *res = nn->layers.prev;
            ////////////////////////////////////// path0 //////////////////////////////////////
            // 37
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 1, 1, 1024, 256, same, 0, 1, 1, 1/2, 1/2, 0)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 256, 14, 14)->list, res, 0);
            // 38
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 3, 3, 256, 256, same, 0, 1, 1, 3/2, 3/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 256, 14, 14)->list, res, 0);
            // 39
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 1, 1, 256, 1024, same, 0, 1, 1, 1/2, 1/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 1024, 14, 14)->list, res, 0);
            ////////////////////////////////////// path0 //////////////////////////////////////
            init_residual_block(res, &ctrl);
        }
        //////////////////////////////////////// bottleneck 2_9 ////////////////////////////////////////

        //////////////////////////////////////// bottleneck 2_10 ////////////////////////////////////////
        {
            my_push_back(&new_residual_block_interface(&ctrl, relu)->list, &nn->layers);
            struct list_node *res = nn->layers.prev;
            ////////////////////////////////////// path0 //////////////////////////////////////
            // 40
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 1, 1, 1024, 256, same, 0, 1, 1, 1/2, 1/2, 0)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 256, 14, 14)->list, res, 0);
            // 41
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 3, 3, 256, 256, same, 0, 1, 1, 3/2, 3/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 256, 14, 14)->list, res, 0);
            // 42
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 1, 1, 256, 1024, same, 0, 1, 1, 1/2, 1/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 1024, 14, 14)->list, res, 0);
            ////////////////////////////////////// path0 //////////////////////////////////////
            init_residual_block(res, &ctrl);
        }
        //////////////////////////////////////// bottleneck 2_10 ////////////////////////////////////////

        //////////////////////////////////////// bottleneck 1_4 ////////////////////////////////////////
        {
            my_push_back(&new_residual_block_interface(&ctrl, relu)->list, &nn->layers);
            struct list_node *res = nn->layers.prev;
            ////////////////////////////////////// path0 //////////////////////////////////////
            // 43
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 1, 1, 1024, 512, same, 0, 1, 1, 1/2, 1/2, 0)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 512, 14, 14)->list, res, 0);
            // 44
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 3, 3, 512, 512, same, 0, 2, 2, 3/2, 3/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 512, 7, 7)->list, res, 0);
            // 45
            path_push_back(&new_convolutional_layer(&ctrl, identity, 7, 7, 1, 1, 512, 2048, same, 0, 1, 1, 1/2, 1/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 2048, 7, 7)->list, res, 0);
            ////////////////////////////////////// path0 //////////////////////////////////////
            ////////////////////////////////////// path1 //////////////////////////////////////
            // 46
            path_push_back(&new_convolutional_layer(&ctrl, identity, 14, 14, 1, 1, 1024, 2048, same, 0, 2, 2, 1/2, 1/2, 1)->list, res, 1);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 2048, 7, 7)->list, res, 1);
            ////////////////////////////////////// path2 //////////////////////////////////////
            init_residual_block(res, &ctrl);
        }
        //////////////////////////////////////// bottleneck 1_4 ////////////////////////////////////////

        //////////////////////////////////////// bottleneck 2_11 ////////////////////////////////////////
        {
            my_push_back(&new_residual_block_interface(&ctrl, relu)->list, &nn->layers);
            struct list_node *res = nn->layers.prev;
            ////////////////////////////////////// path0 //////////////////////////////////////
            // 47
            path_push_back(&new_convolutional_layer(&ctrl, identity, 7, 7, 1, 1, 2048, 512, same, 0, 1, 1, 1/2, 1/2, 0)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 512, 7, 7)->list, res, 0);
            // 48
            path_push_back(&new_convolutional_layer(&ctrl, identity, 7, 7, 3, 3, 512, 512, same, 0, 1, 1, 3/2, 3/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 512, 7, 7)->list, res, 0);
            // 49
            path_push_back(&new_convolutional_layer(&ctrl, identity, 7, 7, 1, 1, 512, 2048, same, 0, 1, 1, 1/2, 1/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 2048, 7, 7)->list, res, 0);
            ////////////////////////////////////// path0 //////////////////////////////////////
            init_residual_block(res, &ctrl);
        }
        //////////////////////////////////////// bottleneck 2_11 ////////////////////////////////////////

        //////////////////////////////////////// bottleneck 2_12 ////////////////////////////////////////
        {
            my_push_back(&new_residual_block_interface(&ctrl, relu)->list, &nn->layers);
            struct list_node *res = nn->layers.prev;
            ////////////////////////////////////// path0 //////////////////////////////////////
            // 50
            path_push_back(&new_convolutional_layer(&ctrl, identity, 7, 7, 1, 1, 2048, 512, same, 0, 1, 1, 1/2, 1/2, 0)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 512, 7, 7)->list, res, 0);
            // 51
            path_push_back(&new_convolutional_layer(&ctrl, identity, 7, 7, 3, 3, 512, 512, same, 0, 1, 1, 3/2, 3/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, relu, 512, 7, 7)->list, res, 0);
            // 52
            path_push_back(&new_convolutional_layer(&ctrl, identity, 7, 7, 1, 1, 512, 2048, same, 0, 1, 1, 1/2, 1/2, 1)->list, res, 0);
            path_push_back(&new_batchnorm_layer(&ctrl, identity, 2048, 7, 7)->list, res, 0);
            ////////////////////////////////////// path0 //////////////////////////////////////
            init_residual_block(res, &ctrl);
        }
        //////////////////////////////////////// bottleneck 2_12 ////////////////////////////////////////
        
        my_push_back(&new_average_pooling_layer(&ctrl, identity, 7, 7, 2048, 7, 7)->list, &nn->layers);
        my_push_back(&new_fully_connected_layer(&ctrl, softmax, 2048, 1000, 1)->list, &nn->layers);
        printf("nwk_cur_ptr: %llx\n", ctrl.nwk_cur_ptr);
        printf("lyr_cur_ptr: %llx\n", ctrl.lyr_cur_ptr);
        printf("wgt_cur_ptr: %llx\n", ctrl.wgt_cur_ptr);
        nn->done_flag = nn->mask;
    }

    for(int i = 0; i < 1000000; i++) {
        if(weights[i] == 777) {
            printf("  ");
        }
    }
    printf("\n");
    for(int i = 0; i < 1000000; i++) {
        if(weights[i] == 777) {
            printf("..");
        }
    }

    for(int i = 1000000; i < 3000000; i++) {
        if(weights[i] == 777) {
            printf("..");
        }
    }

    printf("\n");
    printf("start predict\n");
    predict(nn, inputs, image_size);
    // predict(nn, input_base, image_size);

    // char *cc = "efsfesfe";
    // int s = strlen(cc);

    if (hart_id == 0)
    {
        printf ("Predict done\n");
        layer_base *tmp = list_last_entry(&nn->layers, layer_base, list);
        my_float_t* res = tmp->out_ptr_;
// #ifndef USING_GEM5
//         char *class_name = (char *) class_name_base;
// #else
//         char *class_name = (char *) 0xC300000000000000;
// #endif
        // char **class_name = class_name_f;
        output_index_name out[1000];
        for (int i = 0; i < 1000; i++)
        {
            out[i].index = i;
            out[i].value = res[i];
            // mmap_fgets(out[i].name, sizeof(out[i].name), &class_name);
            // memcpy(&out[i].name, &class_name_f[i], sizeof(out[i].name));
        }
        // qsort((void *)out, 1000, sizeof(output_index_name), compare);
        printf("==============================================\n");
        printf(" idx | possibility(%) | class name\n");
        printf("----------------------------------------------\n");
        for (int i = 0; i < 1000; i++) {
            int t = out[i].index;
            printf(" %3d |       %8.5f | %s\n", t, (float)out[i].value, " "); //out[i].name
        }
        printf("==============================================\n");
    }

    return 0;
}