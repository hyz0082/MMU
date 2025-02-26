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
        return (int)(((float_t)in_length + 2 * padding_size - window_size) / stride) + 1;
    }
    else {
        uint64_t tmp = (in_length - window_size + 1);
        if(tmp % stride == 0) {
            return tmp / stride;
        }
        else {
            return tmp / stride + 1;
        }
        // return (uint64_t)ceil((float_t)(in_length - window_size + 1) / stride);
    }
    // return pad_type == same ? (int)(((float_t)in_length + 2 * padding_size - window_size) / stride) + 1 : (uint64_t)ceil((float_t)(in_length - window_size + 1) / stride);
}

static uint64_t conv_out_dim(uint64_t in_width, uint64_t in_height, uint64_t window_width,  uint64_t window_height, uint64_t w_padding, uint64_t h_padding, uint64_t w_stride, uint64_t h_stride, enum padding pad_type)
{
    return conv_out_length(in_width, window_width, w_padding, w_stride, pad_type) * conv_out_length(in_height, window_height, h_padding, h_stride, pad_type);
}


void conv_copy_and_pad_input(convolutional_layer *entry, unsigned int hart_id, input_struct *input)
{
    if (entry->pad_type_ == same)
    {
        index3d in_ = entry->in_;
        index3d in_padded_ = entry->in_padded_;
        index3d padding_ = entry->padding_;

        uint64_t c = 0;
        uint64_t y = 0;

        float_t *in = input->in_ptr_;
        float_t *dst = entry->base.padded_ptr;
        uint64_t total_size = in_.depth_ * in_.height_;

        for (uint64_t i = 0; i < total_size; i++)
        {
            float_t *pimg = &dst[get_index(&in_padded_, padding_.width_, padding_.height_ + y, c)];
            const float_t *pin = &in[get_index(&in_, 0, y, c)];

            for (uint64_t x = 0; x < in_.width_; x++)
            {
                pimg[x] = pin[x];
            }
            
            y++;
            if (y == in_.height_)
            {
                y = 0;
                c++;
            }
        }
    }
}

/*
void conv_copy_and_pad_input(convolutional_layer *entry, unsigned int hart_id, input_struct *input)
{
    if (entry->pad_type_ == same)
    {
        index3d in_ = entry->in_;
        index3d in_padded_ = entry->in_padded_;
        index3d padding_ = entry->padding_;
        // printf("%d %d\n", padding_.width_, padding_.height_);

        float_t *in = input->in_ptr_;
        float_t *dst = entry->base.padded_ptr;
        
        uint64_t total_size = in_.depth_ * in_.height_;
        uint64_t blocksize = compute_block_size(total_size);
        uint64_t start = (blocksize) * hart_id;
        uint64_t end = min((blocksize) * (hart_id+1), total_size);
    
        for (uint64_t i = start; i < end; i++)
        {

            uint64_t c = i / in_.height_;
            uint64_t y = i % in_.height_;
            // printf("[%d, %d]\n", c, y);
            float_t *pimg = &dst[get_index(&in_padded_, padding_.width_, padding_.height_ + y, c)];
            const float_t *pin = &in[get_index(&in_, 0, y, c)];

            for (uint64_t x = 0; x < in_.width_; x++)
            {
                pimg[x] = pin[x];
            }
            
        }
        // wait for other process done
        // atomic_or(&entry->padding_done_flag, 1LL << hart_id);
        // while (entry->padding_done_flag != entry->padding_mask);
    }
}
*/
void convolutional_layer_forward_propagation(struct list_node *ptr, unsigned int hart_id, input_struct *input)
{
    convolutional_layer *entry = get_convolutional_layer_entry(ptr);

    static int conv_cnt = 0;
    
    if (input->in_size_ != entry->base.in_size_)
    {
        if (hart_id == 0) 
            printf("Error input size not match %d/%d\n", input->in_size_, entry->base.in_size_);
        exit(-1);
    }

    // malloc input space
    entry->base.padded_ptr = (float_t *)malloc(entry->base.padding_size * sizeof(float_t));
    if (entry->base.padded_ptr != NULL) { // Check if memory was allocated
        memset((void*)entry->base.padded_ptr, 0, entry->base.padding_size * sizeof(float_t));
    }
    else {
        printf("Error: Unable to allocate memory for layer->padded_ptr\n");
        // exit(1);
    }

    // malloc output space
    entry->base.a_ptr_ = (float_t *)malloc(entry->base.out_size_ * sizeof(float_t));
    if (entry->base.a_ptr_ != NULL) { // Check if memory was allocated
        memset((void*)entry->base.a_ptr_, 0, entry->base.out_size_ * sizeof(float_t));
    }
    else {
        printf("Error: Unable to allocate memory for entry->base.a_ptr_\n");
        // exit(1);
    }

    // if (entry->base.need_space_for_a){
    //     entry->base.out_ptr_ = (float_t *)malloc(entry->base.out_size_ * sizeof(float_t));
    //     if (entry->base.out_ptr_ != NULL) { // Check if memory was allocated
    //         memset((void*)entry->base.out_ptr_, 0, entry->base.out_size_ * sizeof(float_t));
    //     }
    //     else {
    //         printf("Error: Unable to allocate memory for entry->base.out_ptr_\n");
    //         // exit(1);
    //     }
    // }
    // else {
        entry->base.out_ptr_ = entry->base.a_ptr_;
    // }

    conv_copy_and_pad_input(entry, hart_id, input);

    if(entry->delete_input) {
        free(input->in_ptr_);
    }

    float_t *in = entry->base.padded_ptr;
    float_t *a = entry->base.a_ptr_;
    float_t *W = entry->base._W;
    float_t *b = entry->base._b;
    float_t *out = entry->base.out_ptr_;
    input->in_ptr_ = out;
    input->in_size_ = entry->base.out_size_;

    index3d in_ = entry->in_;
    index3d in_padded_ = entry->in_padded_;
    index3d out_ = entry->out_;
    index3d weight_ = entry->weight_;
    uint64_t h_stride_ = entry->h_stride_;
    uint64_t w_stride_ = entry->w_stride_;

    uint64_t total_size = out_.depth_;
    uint64_t blocksize = compute_block_size(total_size);
    uint64_t start = (blocksize) * hart_id;
    uint64_t end = min((blocksize) * (hart_id+1), total_size);

    uint64_t out_dim = out_.height_*out_.width_;
    
    // printf("cpu parameter:\n");
    // printf("hart_id: %d\n", hart_id);

    // printf("conv parameter:\n");
    // printf("out_.depth_: %d\n in_.depth_: %d\n out_.height_: %d\n out_.width_: %d\n weight_.height_: %d\n weight_.width_: %d\n", out_.depth_, in_.depth_, out_.height_, out_.width_,  weight_.height_, weight_.width_);
    uint32_t tmp_1, tmp_2;
    uint32_t tmp_3, tmp_4;
    tmp_1 = start;
    tmp_2 = end;
    uint32_t tmp[10];
    // printf("start %d, end: %d\n", tmp_1, tmp_2);
    // printf("start conv:\n");
    for (uint64_t o = start; o < end; o++)
    {
        uint32_t tmp_3, tmp_4;
        tmp_3 = o + 1;
        tmp_4 = end;
        printf("[%s]: %d/%d\n", entry->base.layer_name_, tmp_3, tmp_4);
        // [%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_
        // int o = 0;
        float_t *pa = &a[get_index(&out_, 0, 0, o)];
        memset((void*)pa, 0, out_dim *sizeof(float_t));
        // printf("get pa\n");
        for (uint64_t inc = 0; inc < in_.depth_; inc++) {
            // tmp_3 = inc;
            // printf("in.depth_: %d\n", tmp_3);
            // printf("((%d, %d))\n", o, inc);
            const float_t *pw = &W[get_index(&weight_, 0, 0, in_.depth_ * o + inc)];
            const float_t *pi = &in[get_index(&in_padded_, 0, 0, inc)];
            
            for (uint64_t y = 0; y < out_.height_; y++) {
                tmp_4 = y;
                // printf("out_.height_: %d\n", tmp_4);
                for (uint64_t x = 0; x < out_.width_; x++) {
                    // printf("calc %d %d\n", x, y);
                    const float_t * ppw = pw;
                    const float_t * ppi = pi + (y * h_stride_) * in_padded_.width_ + x * w_stride_;
                    float_t sum = (float_t)0;

                    // should be optimized for small kernel(3x3,5x5)
                    for (uint64_t wy = 0; wy < weight_.height_; wy++) {
                        for (uint64_t wx = 0; wx < weight_.width_; wx++) {
                            // if (get_index(&out_, x, y, o) == 1) printf("%f:%x [(%d) %f:%x %f:%x]\n", sum, *(long *)&sum, get_index(&in_padded_, x * w_stride_+wx, y * h_stride_+wy, inc), *ppw, *(long *)ppw, ppi[wy * in_padded_.width_ + wx], *(long *)&ppi[wy * in_padded_.width_ + wx]);
                            if(conv_cnt == 31) {
                                if(x == 0 && y == 0) {
                                    tmp[0] = o;
                                    tmp[1] = inc;
                                    printf("[(%d, %d) %f+= %f*%f]\n", tmp[0], tmp[1], sum, *ppw, ppi[wy * in_padded_.width_ + wx]);
                                    // fprintf(fp_tmp, "start\n");
                                }
                            }
                            sum += *ppw++ * ppi[wy * in_padded_.width_ + wx];
                        }
                    }

                    if(conv_cnt == 31) {
                        if(x == 0 && y == 0) {
                            tmp[0] = o;
                            tmp[1] = inc;
                            printf("[(%d, %d) %f+= %f]\n", tmp[0], tmp[1], pa[y * out_.width_ + x], sum);
                            // fprintf(fp_tmp, "start\n");
                        }
                    }
                    pa[y * out_.width_ + x] += sum;

                    // if(x == 0 && y == 0) {
                    //     tmp[0] = o;
                    //     tmp[1] = inc;
                    //     printf("[(%d, %d) %f]\n", tmp[0], tmp[1], pa[y * out_.width_ + x]);
                    //     // fprintf(fp_tmp, "start\n");
                    // }
                    // printf("calc %d %d end\n", x, y);
                    // if (get_index(&out_, x, y, o) == 1) printf("[(%d, %d, %d) %f]\n", x, y, o, pa[y * out_.width_ + x]);
                }
            }
        }

        // if (entry->has_bias_) {
        //     for (uint64_t index = 0; index < out_dim; index++)
        //         pa[index] += b[o];
        // }

    }
    // wait for other process done
    // atomic_or(&entry->base.a_done_flag, 1LL << hart_id);
    // while (entry->base.a_done_flag != entry->base.mask);

    // total_size = entry->base.out_size_;
    // blocksize = compute_block_size(total_size);
    // start = (blocksize) * hart_id;
    // end = end = min((blocksize) * (hart_id+1), total_size);

    // for (uint64_t c = start; c < end; c++)
    //     out[c] = entry->base.activate(a, c, entry->base.out_size_);
    
    // wait for other process done
    // atomic_or(&entry->base.done_flag, 1LL << hart_id);
    // while (entry->base.done_flag != entry->base.mask);

    free(entry->base.padded_ptr);
    conv_cnt++;

#ifdef PRINT_LAYER
    if (hart_id == 0) 
    {
        printf("[%s] done [%f, %f, ... , %f, %f]\n", entry->base.layer_name_, out[0], out[1], out[entry->base.out_size_-2], out[entry->base.out_size_-1]);
    }
    // int f2i;
    // memcpy(&f2i, &out[0], sizeof(int));
    // printf("value: %d\n", f2i);
#endif
}

layer_base * new_convolutional_layer(
                                     cnn_controller *ctrl,
                                     float_t(*activate) (float_t *, uint64_t, uint64_t),
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

