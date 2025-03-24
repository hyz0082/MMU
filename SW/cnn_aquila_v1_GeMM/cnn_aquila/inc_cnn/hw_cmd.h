#pragma once
#include <stdint.h>
#include "config.h"

#define  RESET              0
#define  TRIGGER_CONV       1
#define  TRIGGER_CONV_LAST  2
#define  SET_MUL_VAL        3
#define  SET_ADD_VAL        4
#define  SET_PE_VAL         5
#define  SET_CONV_MODE      6
#define  SET_FIX_MAC_MODE   7
#define  FORWARD            8
#define  SW_WRITE_DATA      9
#define  SW_WRITE_WEIGHT    10
#define  SW_WRITE_I         11
#define  SET_ROW_IDX        12
#define  SET_KMN            13
#define  SW_READ_DATA       14
#define  SET_PRELOAD        15
#define  SW_WRITE_PARTIAL   16
#define  TRIGGER_BN         17

volatile unsigned int * TPU_CMD      = (unsigned int *)0xC4000000;
volatile unsigned int * PARAM_1_ADDR = (unsigned int *)0xC4000004;
volatile unsigned int * PARAM_2_ADDR = (unsigned int *)0xC4000008;
volatile unsigned int * BUSY_ADDR    = (unsigned int *)0xC4000020;
volatile unsigned int * RET_ADDR     = (unsigned int *)0xC4000010;

volatile float_t * TPU_DATA_ADDR[16] = {(float_t *)0xC4001000, 
                                        (float_t *)0xC4001100, 
                                        (float_t *)0xC4001200, 
                                        (float_t *)0xC4001300, 
                                        (float_t *)0xC4001400, 
                                        (float_t *)0xC4001500, 
                                        (float_t *)0xC4001600, 
                                        (float_t *)0xC4001700, 
                                        (float_t *)0xC4001800, 
                                        (float_t *)0xC4001900, 
                                        (float_t *)0xC4001A00, 
                                        (float_t *)0xC4001B00, 
                                        (float_t *)0xC4001C00, 
                                        (float_t *)0xC4001D00, 
                                        (float_t *)0xC4001E00, 
                                        (float_t *)0xC4001F00, 
                                    };

// localparam [31:0] TPU_DATA_ADDR [0:15] = {32'hC4001000, 32'hC4001100, 32'hC4001200, 32'hC4001300,
//     32'hC4001400, 32'hC4001500, 32'hC4001600, 32'hC4001700,
//     32'hC4001800, 32'hC4001900, 32'hC4001A00, 32'hC4001B00,
//     32'hC4001C00, 32'hC4001D00, 32'hC4001E00, 32'hC4001F00};

// volatile unsigned int * p_dsa_cnt = (unsigned int *) DSA_CNT_ADDR;
// volatile float * p_dsa_result = (float *) DSA_RESULT_ADDR;
// volatile unsigned int * p_dsa_trigger = (unsigned int *) DSA_TRIGGER_ADDR;
// volatile unsigned int * p_dsa_base = (unsigned int *) DSA_BASE_ADDR;
// volatile unsigned int * p_dsa_top = (unsigned int *) DSA_TOP_ADDR;
// volatile float * p_dsa_buff_1 = (float *) DSA_BUFF_1;
// volatile float * p_dsa_buff_2 = (float *) DSA_BUFF_2;
// volatile float * p_dsa_buff_3 = (float *) DSA_BUFF_3;
// volatile float * p_dsa_weight[2] = {(float *) DSA_BUFF_2, (float *) DSA_BUFF_3};
static inline void send_bn_data(float_t data, int pos){
    // uint32_t tmp;
    // memcpy(&tmp, &data, sizeof(float_t));
    *(TPU_DATA_ADDR[pos]) = data;//tmp;

    // __asm__ volatile ("nop");
    // __asm__ volatile ("nop");
}

static inline void write_bn_data(int pos){
    *PARAM_2_ADDR = pos;//pos;
    *TPU_CMD = SW_WRITE_PARTIAL;
    __asm__ volatile ("nop");
    // __asm__ volatile ("nop");
    // __asm__ volatile ("nop");
}

void send_bn_mul_data(float_t data, int pos){
    uint32_t tmp;
    memcpy(&tmp, &data, sizeof(float_t));
    *PARAM_1_ADDR = pos;
    *PARAM_2_ADDR = tmp;
    *TPU_CMD = SET_MUL_VAL;
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
}

void send_bn_add_data(float_t data, int pos){
    uint32_t tmp;
    memcpy(&tmp, &data, sizeof(float_t));
    *PARAM_1_ADDR = pos;
    *PARAM_2_ADDR = tmp;
    *TPU_CMD = SET_ADD_VAL;
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
}

void reset_cmd() {
    *TPU_CMD = RESET;
    __asm__ volatile ("nop");
}

void set_KMN_cmd(int K, int M, int N) {
    *PARAM_1_ADDR = 0; // K
    *PARAM_2_ADDR = K;
    *TPU_CMD = SET_KMN;
    __asm__ volatile ("nop");
    *PARAM_1_ADDR = 1; // M
    *PARAM_2_ADDR = M;
    *TPU_CMD = SET_KMN;
    __asm__ volatile ("nop");
    *PARAM_1_ADDR = 2; // N
    *PARAM_2_ADDR = N;
    *TPU_CMD = SET_KMN;
    __asm__ volatile ("nop");
}

void send_data_cmd(float_t data, int pos) {
    uint32_t tmp;
    memcpy(&tmp, &data, sizeof(float_t));
    *PARAM_1_ADDR = tmp;
    *PARAM_2_ADDR =  pos;
    *TPU_CMD = SW_WRITE_DATA;
    __asm__ volatile ("nop"); 
}

void send_weight_cmd(float_t data, int pos) {
    uint32_t tmp;
    memcpy(&tmp, &data, sizeof(float_t));
    *PARAM_1_ADDR = tmp;
    *PARAM_2_ADDR =  pos;
    *TPU_CMD = SW_WRITE_WEIGHT;
    __asm__ volatile ("nop"); 
}

void send_idx_cmd(int data, int pos) {
    *PARAM_1_ADDR = data;
    *PARAM_2_ADDR = pos;
    *TPU_CMD = SW_WRITE_I;
    __asm__ volatile ("nop"); 
}

void set_conv_cmd(int len) {
    *PARAM_1_ADDR = len;
    *TPU_CMD = SET_CONV_MODE;
    __asm__ volatile ("nop");
}

void set_bn_cmd(int len) {
    *PARAM_1_ADDR = len;
    *TPU_CMD = SET_FIX_MAC_MODE;
    __asm__ volatile ("nop");
}

void set_idx_cmd(int offset_0, int offset_1, int offset_2, int offset_3) {
    *PARAM_1_ADDR = 0;
    *PARAM_2_ADDR = offset_0;
    *TPU_CMD = SET_ROW_IDX;
    __asm__ volatile ("nop");
    *PARAM_1_ADDR = 1;
    *PARAM_2_ADDR = offset_1;
    *TPU_CMD = SET_ROW_IDX;
    __asm__ volatile ("nop");
    *PARAM_1_ADDR = 2;
    *PARAM_2_ADDR = offset_2;
    *TPU_CMD = SET_ROW_IDX;
    __asm__ volatile ("nop");
    *PARAM_1_ADDR = 3;
    *PARAM_2_ADDR = offset_3;
    *TPU_CMD = SET_ROW_IDX;
    __asm__ volatile ("nop");
}

void reset_preload_cmd() {
    *PARAM_1_ADDR = 0;
    *PARAM_2_ADDR = 0;
    *TPU_CMD = SET_PRELOAD;
    __asm__ volatile ("nop");
}

void trigger_conv_cmd() {
    *TPU_CMD = TRIGGER_CONV;
    __asm__ volatile ("nop");
}

void trigger_bn_cmd() {
    *TPU_CMD = TRIGGER_BN;
    __asm__ volatile ("nop");
}

void wait_idle_cmd() {
    while((*BUSY_ADDR));
    __asm__ volatile ("nop");
}

float_t read_data_cmd(int offset, int pos) {
    unsigned int tmp;
    float_t ret;

    *PARAM_1_ADDR = offset;
    *PARAM_2_ADDR = pos;
    *TPU_CMD = SW_READ_DATA;
    for(int i = 0; i < 5; i++)
        __asm__ volatile ("nop");
    
    tmp = *RET_ADDR;
    memcpy(&ret, &tmp, sizeof(float_t));
    return ret;
}