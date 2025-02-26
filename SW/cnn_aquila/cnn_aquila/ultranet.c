#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
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



float_t sigmoid__(float_t x) {
    return (1 / (1 + exp(-x)));
}

#ifndef USING_GEM5
#include "loader.h"
#endif

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

    uint64_t image_size = 320*160*3;
    if (hart_id == 0)
    {
        push_back(&new_dummy_head_layer(&ctrl, identity, image_size)->list, &nn->layers);
        push_back(&new_convolutional_layer(&ctrl, identity, 320, 160, 3, 3, 3, 16, same, 0, 1, 1, 3/2, 3/2)->list, &nn->layers);
        push_back(&new_batchnorm_layer(&ctrl, bounded_relu, 16, 320, 160)->list, &nn->layers);
        push_back(&new_max_pooling_layer(&ctrl, identity, 320, 160, 16, 2, 2, 0)->list, &nn->layers);

        push_back(&new_convolutional_layer(&ctrl, identity, 160, 80, 3, 3, 16, 32, same, 0, 1, 1, 3/2, 3/2)->list, &nn->layers);
        push_back(&new_batchnorm_layer(&ctrl, bounded_relu, 32, 160, 80)->list, &nn->layers);
        push_back(&new_max_pooling_layer(&ctrl, identity, 160, 80, 32, 2, 2, 0)->list, &nn->layers);

        push_back(&new_convolutional_layer(&ctrl, identity, 80, 40, 3, 3, 32, 64, same, 0, 1, 1, 3/2, 3/2)->list, &nn->layers);
        push_back(&new_batchnorm_layer(&ctrl, bounded_relu, 64, 80, 40)->list, &nn->layers);
        push_back(&new_max_pooling_layer(&ctrl, identity, 80, 40, 64, 2, 2, 0)->list, &nn->layers);

        push_back(&new_convolutional_layer(&ctrl, identity, 40, 20, 3, 3, 64, 64, same, 0, 1, 1, 3/2, 3/2)->list, &nn->layers);
        push_back(&new_batchnorm_layer(&ctrl, bounded_relu, 64, 40, 20)->list, &nn->layers);
        push_back(&new_max_pooling_layer(&ctrl, identity, 40, 20, 64, 2, 2, 0)->list, &nn->layers);

        push_back(&new_convolutional_layer(&ctrl, identity, 20, 10, 3, 3, 64, 64, same, 0, 1, 1, 3/2, 3/2)->list, &nn->layers);
        push_back(&new_batchnorm_layer(&ctrl, bounded_relu, 64, 20, 10)->list, &nn->layers);

        push_back(&new_convolutional_layer(&ctrl, identity, 20, 10, 3, 3, 64, 64, same, 0, 1, 1, 3/2, 3/2)->list, &nn->layers);
        push_back(&new_batchnorm_layer(&ctrl, bounded_relu, 64, 20, 10)->list, &nn->layers);

        push_back(&new_convolutional_layer(&ctrl, identity, 20, 10, 3, 3, 64, 64, same, 0, 1, 1, 3/2, 3/2)->list, &nn->layers);
        push_back(&new_batchnorm_layer(&ctrl, bounded_relu, 64, 20, 10)->list, &nn->layers);

        push_back(&new_convolutional_layer(&ctrl, identity, 20, 10, 3, 3, 64, 64, same, 0, 1, 1, 3/2, 3/2)->list, &nn->layers);
        push_back(&new_batchnorm_layer(&ctrl, bounded_relu, 64, 20, 10)->list, &nn->layers);

        push_back(&new_convolutional_layer(&ctrl, identity, 20, 10, 1, 1, 64, 36, same, 0, 1, 1, 1/2, 1/2)->list, &nn->layers);
        
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

        for (int box = 0; box < 6; box++) {
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 20; j++) {
                    int xidx = (box * 6 + 0) * 200 + i * 20 + j;
                    int yidx = (box * 6 + 1) * 200 + i * 20 + j;
                    int widx = (box * 6 + 2) * 200 + i * 20 + j;
                    int hidx = (box * 6 + 3) * 200 + i * 20 + j;
                    int confidx = (box * 6 + 4) * 200 + i * 20 + j;
                    res[xidx] = (sigmoid__(res[xidx]) + j) * 16;
                    res[yidx] = (sigmoid__(res[yidx]) + i) * 16;
                    res[widx] = (exp      (res[widx])    ) * 20;
                    res[hidx] = (exp      (res[hidx])    ) * 20;
                    res[confidx] = sigmoid__(res[confidx]);
                }
            }
        }

        float_t box_avg_res[1200] = { 0.0 };

        for (int box = 0; box < 6; box++) {
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 20; j++) {
                    int xidx = (box * 6 + 0) * 200 + i * 20 + j;
                    int yidx = (box * 6 + 1) * 200 + i * 20 + j;
                    int widx = (box * 6 + 2) * 200 + i * 20 + j;
                    int hidx = (box * 6 + 3) * 200 + i * 20 + j;
                    int confidx = (box * 6 + 4) * 200 + i * 20 + j;
                    box_avg_res[(i*20+j) * 6 + 0] += res[xidx] / 6.0;
                    box_avg_res[(i*20+j) * 6 + 1] += res[yidx] / 6.0;
                    box_avg_res[(i*20+j) * 6 + 2] += res[widx] / 6.0;
                    box_avg_res[(i*20+j) * 6 + 3] += res[hidx] / 6.0;
                    box_avg_res[(i*20+j) * 6 + 4] += res[confidx] / 6.0;
                }
            }
        }
        
        float_t max_conf_value = 0;
        int max_conf_i = -1, max_conf_j = -1;
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 20; j++) {
                int offset = (i * 20 + j) * 6 + 4;
                if (box_avg_res[offset] > max_conf_value) {
                    max_conf_value = box_avg_res[offset];
                    max_conf_i = i;
                    max_conf_j = j;
                }
            }
        }

        printf("%f\n%d\n%d\n", max_conf_value, max_conf_i, max_conf_j);

        // cout << max_conf_value << endl;
        // cout << max_conf_i << endl;
        // cout << max_conf_j << endl;

        int minx = box_avg_res[(max_conf_i*20+max_conf_j)*6 + 0] - box_avg_res[(max_conf_i*20+max_conf_j)*6 + 2] / 2;
        int maxx = box_avg_res[(max_conf_i*20+max_conf_j)*6 + 0] + box_avg_res[(max_conf_i*20+max_conf_j)*6 + 2] / 2;
        int miny = box_avg_res[(max_conf_i*20+max_conf_j)*6 + 1] - box_avg_res[(max_conf_i*20+max_conf_j)*6 + 3] / 2;
        int maxy = box_avg_res[(max_conf_i*20+max_conf_j)*6 + 1] + box_avg_res[(max_conf_i*20+max_conf_j)*6 + 3] / 2;

        // cout << "(" << miny << ", " << minx << "), " << "(" << maxy << ", " << maxx << ")" << endl;

        printf("(%d, %d), (%d, %d)\n", miny, minx, maxy, maxx);
    }
    
}

int main(int argc, char** argv)
{
#ifndef USING_GEM5
    if (argc < 5)
    {
        fprintf(stderr, "Usage: %s <weight_file> <image_to_inference> <total_CPUs> <hart_id>\n", argv[0]);
        exit(-1);
    }
    init_loader(argv[1], argv[2]);
    unsigned int total_cpu = atoi(argv[3]);
    unsigned int hart_id = atoi(argv[4]);
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