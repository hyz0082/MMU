#pragma once
#include <stdint.h>
#include "config.h"
#include "string.h"

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
#define  SET_MAX_POOLING    18
#define  SET_DIVISOR        19
#define  SET_SOFTMAX        20
#define  TRIGGER_SOFTMAX    21

#define    SET_MODE     22 // param_1: mode
                           // param_2: len
#define    TRIGGER_ADD  23 
#define    SET_RELU     24
#define    SET_AVERAGE_POOLING 25

#define    SET_BN_MUL_SRAM_0 26
#define    SET_BN_MUL_SRAM_1 27
#define    SET_BN_MUL_SRAM_2 28
#define    SET_BN_MUL_SRAM_3 29
#define    SET_BN_ADD_SRAM_0 30
#define    SET_BN_ADD_SRAM_1 31
#define    SET_BN_ADD_SRAM_2 32
#define    SET_BN_ADD_SRAM_3 33

#define    SET_I_OFFSET_1 34
#define    SET_I_OFFSET_2 35
#define    SET_I_OFFSET_3 36
#define    SET_I_OFFSET_4 37

#define    SET_COL_IDX 38 

#define    RESET_LANS      39 
#define    SET_LANS_IDX    40
#define    SRAM_NEXT       41
#define    POOLING_START   42
#define    RESET_POOLING_IDX  43

volatile unsigned int * TPU_CMD      = (unsigned int *)0xC4000000;
volatile unsigned int * PARAM_1_ADDR = (unsigned int *)0xC4000004;
volatile my_float_t   * PARAM_1_ADDR_FP = (my_float_t*)0xC4000004;
volatile unsigned int * PARAM_2_ADDR = (unsigned int *)0xC4000008;
volatile unsigned int * BUSY_ADDR    = (unsigned int *)0xC4000020;
volatile unsigned int * RET_ADDR     = (unsigned int *)0xC4000010;
volatile my_float_t * RET_ADDR_FP     = (my_float_t*)0xC4000010;
volatile unsigned int * RET_MAX_POOLING_ADDR = (unsigned int *)0xC4002010;
volatile unsigned int * RET_SOFTMAX_ADDR     = (unsigned int *)0xC4002020;
volatile my_float_t * RET_SOFTMAX_ADDR_FP     = (my_float_t*)0xC4002020;

volatile unsigned int * DRAM_R_ADDR   = (unsigned int *)0xC4002024;
volatile unsigned int * DRAM_R_LENGTH = (unsigned int *)0xC4002028;
volatile unsigned int * DRAM_RW       = (unsigned int *)0xC400202C;
volatile unsigned int * DRAM_TR       = (unsigned int *)0xC4002030;
volatile unsigned int * SRAM_OFFSET   = (unsigned int *)0xC4002034;
volatile unsigned int * WRITE_DATA_TYPE_ADDR = (unsigned int *)0xC4002038;

volatile unsigned int * DRAM_WRITE_ADDR [4] = {(unsigned int *)0xC400203C,
                                                 (unsigned int *)0xC4002040,
                                                 (unsigned int *)0xC4002044,
                                                 (unsigned int *)0xC4002048};
volatile unsigned int * NUM_LANS_ADDR  = (unsigned int *)0xC400204C;
volatile unsigned int * DRAM_WRITE_LEN = (unsigned int *)0xC4002050;
volatile unsigned int * TR_DRAM_W      = (unsigned int *)0xC4002054;
volatile unsigned int * OUTPUT_RECV_CNT_ADDR = (unsigned int *)0xC4002058;

volatile my_float_t * SW_DATA_ADDR = (my_float_t*)0xC400205C;
volatile unsigned int * SW_WRITE_DRAM_MODE_ADDR = (unsigned int *)0xC4002060;
volatile my_float_t * RET_AVG_POOLING_ADDR = (my_float_t*)0xC4002064;

volatile unsigned int * GEMM_CORE_SEL_ADDR = (unsigned int *)0xC4002068;
volatile unsigned int * BUSY_ADDR_2        = (unsigned int *)0xC400206C;

volatile unsigned int * READ_OFFSET = (unsigned int *)0xC4002070;
volatile unsigned int * READ_ROUNDS = (unsigned int *)0xC4002074;

volatile my_float_t * TPU_DATA_ADDR[16] = {(my_float_t*)0xC4001000, 
                                        (my_float_t*)0xC4001100, 
                                        (my_float_t*)0xC4001200, 
                                        (my_float_t*)0xC4001300, 
                                        (my_float_t*)0xC4001400, 
                                        (my_float_t*)0xC4001500, 
                                        (my_float_t*)0xC4001600, 
                                        (my_float_t*)0xC4001700, 
                                        (my_float_t*)0xC4001800, 
                                        (my_float_t*)0xC4001900, 
                                        (my_float_t*)0xC4001A00, 
                                        (my_float_t*)0xC4001B00, 
                                        (my_float_t*)0xC4001C00, 
                                        (my_float_t*)0xC4001D00, 
                                        (my_float_t*)0xC4001E00, 
                                        (my_float_t*)0xC4001F00, 
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
void set_gemm_core_sel_cmd(int sel) {
    *GEMM_CORE_SEL_ADDR = sel;
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
}

static inline void send_bn_data(my_float_t data, int pos){
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

void send_bn_mul_data(my_float_t data, int pos){
    uint32_t tmp;
    memcpy(&tmp, &data, sizeof(my_float_t));
    *PARAM_1_ADDR = pos;
    *PARAM_2_ADDR = tmp;
    *TPU_CMD = SET_MUL_VAL;
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
}

void send_bn_add_data(my_float_t data, int pos){
    uint32_t tmp;
    memcpy(&tmp, &data, sizeof(my_float_t));
    *PARAM_1_ADDR = pos;
    *PARAM_2_ADDR = tmp;
    *TPU_CMD = SET_ADD_VAL;
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
}

void reset_cmd() {
    set_gemm_core_sel_cmd(3);
    *TPU_CMD = RESET;
    __asm__ volatile ("nop");

    // set_gemm_core_sel_cmd(2);
    
    // *TPU_CMD = RESET;
    set_gemm_core_sel_cmd(1);
    __asm__ volatile ("nop");
}

void set_KMN_cmd(int K, int M, int N) {
    set_gemm_core_sel_cmd(3);
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

    // set_gemm_core_sel_cmd(2);
    // *PARAM_1_ADDR = 0; // K
    // *PARAM_2_ADDR = K;
    // *TPU_CMD = SET_KMN;
    // __asm__ volatile ("nop");
    // *PARAM_1_ADDR = 1; // M
    // *PARAM_2_ADDR = M;
    // *TPU_CMD = SET_KMN;
    // __asm__ volatile ("nop");
    // *PARAM_1_ADDR = 2; // N
    // *PARAM_2_ADDR = N;
    // *TPU_CMD = SET_KMN;
    set_gemm_core_sel_cmd(1);
    __asm__ volatile ("nop");
}

void send_data_cmd(my_float_t data, int pos) {
    uint32_t tmp;
    // memcpy(&tmp, &data, sizeof(float_t));
    // *PARAM_1_ADDR = tmp;
    *PARAM_1_ADDR_FP = data;
    *PARAM_2_ADDR =  pos;
    *TPU_CMD = SW_WRITE_DATA;
    // __asm__ volatile ("nop");
}

void send_weight_cmd(my_float_t data, int pos) {
    uint32_t tmp;
    // memcpy(&tmp, &data, sizeof(float_t));
    // *PARAM_1_ADDR = tmp;
    *PARAM_1_ADDR_FP = data;
    *PARAM_2_ADDR =  pos;
    *TPU_CMD = SW_WRITE_WEIGHT;
    // __asm__ volatile ("nop"); 
}

void send_idx_cmd(int data, int pos) {
    *PARAM_1_ADDR = data;
    *PARAM_2_ADDR = pos;
    *TPU_CMD = SW_WRITE_I;
    __asm__ volatile ("nop"); 
}

void send_offset_1_cmd(int data, int pos) {
    *PARAM_1_ADDR = data;
    *PARAM_2_ADDR = pos;
    *TPU_CMD = SET_I_OFFSET_1;
    __asm__ volatile ("nop"); 
}

void send_offset_2_cmd(int data, int pos) {
    *PARAM_1_ADDR = data;
    *PARAM_2_ADDR = pos;
    *TPU_CMD = SET_I_OFFSET_2;
    __asm__ volatile ("nop"); 
}

void send_offset_3_cmd(int data, int pos) {
    *PARAM_1_ADDR = data;
    *PARAM_2_ADDR = pos;
    *TPU_CMD = SET_I_OFFSET_3;
    __asm__ volatile ("nop"); 
}

void send_offset_4_cmd(int data, int pos) {
    *PARAM_1_ADDR = data;
    *PARAM_2_ADDR = pos;
    *TPU_CMD = SET_I_OFFSET_4;
    __asm__ volatile ("nop"); 
}

void set_conv_cmd(int len) {
    set_gemm_core_sel_cmd(3);
    *PARAM_1_ADDR = len;
    *TPU_CMD = SET_CONV_MODE;
    __asm__ volatile ("nop");

    // set_gemm_core_sel_cmd(2);
    // *PARAM_1_ADDR = len;
    // *TPU_CMD = SET_CONV_MODE;
    set_gemm_core_sel_cmd(1);
    __asm__ volatile ("nop");
}

void set_bn_cmd(int len) {
    *PARAM_1_ADDR = len;
    *TPU_CMD = SET_FIX_MAC_MODE;
    __asm__ volatile ("nop");
}

void set_idx_cmd(int offset_0, int offset_1, int offset_2, int offset_3) {
    set_gemm_core_sel_cmd(3);
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

    // set_gemm_core_sel_cmd(2);
    // *PARAM_1_ADDR = 0;
    // *PARAM_2_ADDR = offset_0;
    // *TPU_CMD = SET_ROW_IDX;
    // __asm__ volatile ("nop");
    // *PARAM_1_ADDR = 1;
    // *PARAM_2_ADDR = offset_1;
    // *TPU_CMD = SET_ROW_IDX;
    // __asm__ volatile ("nop");
    // *PARAM_1_ADDR = 2;
    // *PARAM_2_ADDR = offset_2;
    // *TPU_CMD = SET_ROW_IDX;
    // __asm__ volatile ("nop");
    // *PARAM_1_ADDR = 3;
    // *PARAM_2_ADDR = offset_3;
    // *TPU_CMD = SET_ROW_IDX;
    set_gemm_core_sel_cmd(1);
    __asm__ volatile ("nop");
}

void set_col_idx_cmd(int offset_0, int offset_1, int offset_2, int offset_3) {
    set_gemm_core_sel_cmd(3);
    *PARAM_1_ADDR = 0;
    *PARAM_2_ADDR = offset_0;
    *TPU_CMD = SET_COL_IDX;
    __asm__ volatile ("nop");
    *PARAM_1_ADDR = 1;
    *PARAM_2_ADDR = offset_1;
    *TPU_CMD = SET_COL_IDX;
    __asm__ volatile ("nop");
    *PARAM_1_ADDR = 2;
    *PARAM_2_ADDR = offset_2;
    *TPU_CMD = SET_COL_IDX;
    __asm__ volatile ("nop");
    *PARAM_1_ADDR = 3;
    *PARAM_2_ADDR = offset_3;
    *TPU_CMD = SET_COL_IDX;
    __asm__ volatile ("nop");

    // set_gemm_core_sel_cmd(2);
    // *PARAM_1_ADDR = 0;
    // *PARAM_2_ADDR = offset_0;
    // *TPU_CMD = SET_COL_IDX;
    // __asm__ volatile ("nop");
    // *PARAM_1_ADDR = 1;
    // *PARAM_2_ADDR = offset_1;
    // *TPU_CMD = SET_COL_IDX;
    // __asm__ volatile ("nop");
    // *PARAM_1_ADDR = 2;
    // *PARAM_2_ADDR = offset_2;
    // *TPU_CMD = SET_COL_IDX;
    // __asm__ volatile ("nop");
    // *PARAM_1_ADDR = 3;
    // *PARAM_2_ADDR = offset_3;
    // *TPU_CMD = SET_COL_IDX;
    set_gemm_core_sel_cmd(1);
    __asm__ volatile ("nop");
}

void reset_preload_cmd() {
    set_gemm_core_sel_cmd(3);
    *PARAM_1_ADDR = 0;
    *PARAM_2_ADDR = 0;
    *TPU_CMD = SET_PRELOAD;
    __asm__ volatile ("nop");

    // set_gemm_core_sel_cmd(2);
    // *PARAM_1_ADDR = 0;
    // *PARAM_2_ADDR = 0;
    // *TPU_CMD = SET_PRELOAD;
    set_gemm_core_sel_cmd(1);
    __asm__ volatile ("nop");
}

void trigger_conv_cmd() {
    *TPU_CMD = TRIGGER_CONV;
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
}

void trigger_bn_cmd() {
    *TPU_CMD = TRIGGER_BN;
    __asm__ volatile ("nop");
}

void wait_idle_cmd() {
    while((*BUSY_ADDR));
    __asm__ volatile ("nop");
    while((*BUSY_ADDR));
    __asm__ volatile ("nop");
    while((*BUSY_ADDR));
    for(int i = 0; i < 50; i++) {
        __asm__ volatile ("nop");
    }
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");

}

void wait_idle_quick_cmd() {
    while((*BUSY_ADDR));
    __asm__ volatile ("nop");

}

my_float_t read_data_cmd(int offset, int pos) {
    unsigned int tmp;
    float_t ret;

    *PARAM_1_ADDR = offset;
    *PARAM_2_ADDR = pos;
    *TPU_CMD = SW_READ_DATA;
    for(int i = 0; i < 5; i++)
        __asm__ volatile ("nop");
    
    // tmp = *RET_ADDR;
    // memcpy(&ret, &tmp, sizeof(float_t));
    // return ret;
    return *RET_ADDR_FP;
}

my_float_t read_max_pooling_cmd() {
    unsigned int tmp;
    my_float_t ret;
    tmp = *RET_MAX_POOLING_ADDR;
    memcpy(&ret, &tmp, sizeof(my_float_t));
    return ret;
}

my_float_t read_avg_pooling_cmd() {
    return *RET_AVG_POOLING_ADDR;
}

void set_max_pooling_cmd() {
    *TPU_CMD = SET_MAX_POOLING;
    __asm__ volatile ("nop");
}

void set_avg_pooling_cmd() {
    *TPU_CMD = SET_AVERAGE_POOLING;
    __asm__ volatile ("nop");
}

void send_exp_acc_cmd(my_float_t data) {
    uint32_t tmp;
    // memcpy(&tmp, &data, sizeof(my_float_t));
    *PARAM_1_ADDR_FP = data;
    *PARAM_2_ADDR = 0;
    *TPU_CMD = SET_SOFTMAX;
    __asm__ volatile ("nop"); 
}

void sf_calc_cmd(my_float_t data) {
    // uint32_t tmp;
    // memcpy(&tmp, &data, sizeof(my_float_t));
    *PARAM_1_ADDR_FP = data;
    *PARAM_2_ADDR = 0;
    *TPU_CMD = TRIGGER_SOFTMAX;
    __asm__ volatile ("nop"); 
}

my_float_t read_sf_cmd() {
    // unsigned int tmp;
    // my_float_t ret;
    // tmp = *RET_SOFTMAX_ADDR;
    // memcpy(&ret, &tmp, sizeof(my_float_t));
    return *RET_SOFTMAX_ADDR_FP;//ret;
}

/**
 *  dram access cmd
 */
void set_addr_cmd(int addr) {
    // uint32_t tmp;
    // memcpy(&tmp, &addr, sizeof(tmp));
    // printf("addr: %d\n", addr);
    // uint32_t tmp;
    // memcpy(&tmp, &DRAM_R_ADDR, sizeof(tmp));
    // printf("addr: %d\n", tmp);
    *DRAM_R_ADDR = addr;
}

void set_length_cmd(int len) {
    *DRAM_R_LENGTH = len;
}

void reset_sram_offset_cmd() {
    *SRAM_OFFSET = 0;
}

void trigger_dram_read_cmd() {
    __asm__ volatile ("nop"); 
    __asm__ volatile ("nop"); 
    *DRAM_TR = 1;
}


void set_dram_read_input_cmd() {
    *WRITE_DATA_TYPE_ADDR = 0;
    __asm__ volatile ("nop");
}

void set_dram_read_weight_cmd() {
    *WRITE_DATA_TYPE_ADDR = 1;
    __asm__ volatile ("nop");
}

void set_dram_write_addr_cmd(int pos, int addr) {
    if(pos >= 4) {
        printf("*************Error************\n\n\n");
    }
    *(DRAM_WRITE_ADDR[pos]) = addr;
    // printf("set_dram_write_addr_cmd: pos: %x, addr: %x\n", pos, addr);
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
}

void set_num_lans_cmd(int num) {
    // printf("set_num_lans_cmd: %d\n", num);
    *(NUM_LANS_ADDR) = num;
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
}

void set_dram_write_lens_cmd(int len) {
    // printf("write len: %d\n", len);
    *(DRAM_WRITE_LEN) = len;
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
}

void set_dram_w_tr_cmd() {
    *(TR_DRAM_W) = 1;
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
}

void set_output_recv_cnt_cmd(int pos){
    *(OUTPUT_RECV_CNT_ADDR) = pos;
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
}

void send_param1_cmd(int data){
    *PARAM_1_ADDR = data;
}

void send_param2_cmd(int data){
    *PARAM_2_ADDR = data;
}


my_float_t read_dram_value_cmd(my_float_t *addr) {

    uint32_t mask = 0b00111110;  // Mask to isolate bits 1 to 5
    // uint32_t result;

    reset_sram_offset_cmd();
    set_length_cmd(1);
    uint32_t tmp_s;
    memcpy(&tmp_s, &addr, sizeof(tmp_s));

    
    // result = (tmp_s & mask) >> 1;
    // printf("read addr: %x\n", tmp_s);
    // printf("data pos: %d\n", result);
    set_addr_cmd(tmp_s);
    // trigger_dram_read_cmd();
    *DRAM_TR = 0;
    wait_idle_cmd();

    // printf("read %f from addr %x\n", (float)(*SW_DATA_ADDR), tmp_s);
    return *SW_DATA_ADDR;
}

void write_dram_value_cmd(my_float_t *addr, my_float_t data) {

    *PARAM_1_ADDR_FP = data;
    set_dram_write_lens_cmd(1);
    set_output_recv_cnt_cmd(0);
    set_num_lans_cmd(0);
    *SW_WRITE_DRAM_MODE_ADDR = 1;

    my_float_t *pi = addr;
    uint32_t tmp_s;
    memcpy(&tmp_s, &pi, sizeof(tmp_s));
    // printf("write %f to addr %x\n", (float)data, tmp_s);
    set_dram_write_addr_cmd(0, tmp_s);

    // set_dram_w_tr_cmd();
    *(TR_DRAM_W) = 0;
    wait_idle_cmd();

    *SW_WRITE_DRAM_MODE_ADDR = 0;
    // for(int i = 0; i < 100; i++) {
    //     __asm__ volatile ("nop");
    // }

}


void reset_dram_value_cmd(my_float_t *addr, int len) {

    // set_gemm_core_sel_cmd(1);

    *PARAM_1_ADDR_FP = 0;
    *SW_WRITE_DRAM_MODE_ADDR = 1;

    my_float_t *ptr = addr;
    set_output_recv_cnt_cmd(0);
    set_num_lans_cmd(0);

    for(int i = 0; i < len; i += 10000) {
        int remain = min(len - i, 10000);
        set_dram_write_lens_cmd(remain);

        uint32_t tmp_s;
        memcpy(&tmp_s, &ptr, sizeof(tmp_s));
        set_dram_write_addr_cmd(0, tmp_s);
        *(TR_DRAM_W) = 0;
        ptr += remain;
        wait_idle_cmd();
    }
    
    *SW_WRITE_DRAM_MODE_ADDR = 0;
    // for(int i = 0; i < 100; i++) {
    //     __asm__ volatile ("nop");
    // }

}

void set_mode_cmd(int mode, int length) {
    *PARAM_1_ADDR = mode;
    *PARAM_2_ADDR = length;
    *TPU_CMD = SET_MODE;
    __asm__ volatile ("nop");
}

void set_relu_cmd() {
    set_gemm_core_sel_cmd(3);
    *PARAM_1_ADDR = 1;
    *TPU_CMD = SET_RELU;
    __asm__ volatile ("nop");

    // set_gemm_core_sel_cmd(2);
    // *PARAM_1_ADDR = 1;
    // *TPU_CMD = SET_RELU;
    set_gemm_core_sel_cmd(1);
    __asm__ volatile ("nop");

    // set_gemm_core_sel_cmd(1);
}

void reset_relu_cmd() {
    set_gemm_core_sel_cmd(3);
    *PARAM_1_ADDR = 0;
    *TPU_CMD = SET_RELU;
    __asm__ volatile ("nop");

    // set_gemm_core_sel_cmd(2);
    // *PARAM_1_ADDR = 0;
    // *TPU_CMD = SET_RELU;
    set_gemm_core_sel_cmd(1);
    __asm__ volatile ("nop");
    // set_gemm_core_sel_cmd(1);
}

void trigger_add_cmd() {
    *TPU_CMD = TRIGGER_ADD;
    __asm__ volatile ("nop");
}


void set_mul_sram_0_cmd(my_float_t data, int index) {
    *PARAM_1_ADDR_FP = data;
    *PARAM_2_ADDR    = index;
    *TPU_CMD = SET_BN_MUL_SRAM_0;
    __asm__ volatile ("nop");
}

void set_mul_sram_1_cmd(my_float_t data, int index) {
    *PARAM_1_ADDR_FP = data;
    *PARAM_2_ADDR    = index;
    *TPU_CMD = SET_BN_MUL_SRAM_1;
    __asm__ volatile ("nop");
}

void set_mul_sram_2_cmd(my_float_t data, int index) {
    *PARAM_1_ADDR_FP = data;
    *PARAM_2_ADDR    = index;
    *TPU_CMD = SET_BN_MUL_SRAM_2;
    __asm__ volatile ("nop");
}

void set_mul_sram_3_cmd(my_float_t data, int index) {
    *PARAM_1_ADDR_FP = data;
    *PARAM_2_ADDR    = index;
    *TPU_CMD = SET_BN_MUL_SRAM_3;
    __asm__ volatile ("nop");
}

void set_add_sram_0_cmd(my_float_t data, int index) {
    *PARAM_1_ADDR_FP = data;
    *PARAM_2_ADDR    = index;
    *TPU_CMD = SET_BN_ADD_SRAM_0;
    __asm__ volatile ("nop");
}

void set_add_sram_1_cmd(my_float_t data, int index) {
    *PARAM_1_ADDR_FP = data;
    *PARAM_2_ADDR    = index;
    *TPU_CMD = SET_BN_ADD_SRAM_1;
    __asm__ volatile ("nop");
}

void set_add_sram_2_cmd(my_float_t data, int index) {
    *PARAM_1_ADDR_FP = data;
    *PARAM_2_ADDR    = index;
    *TPU_CMD = SET_BN_ADD_SRAM_2;
    __asm__ volatile ("nop");
}

void set_add_sram_3_cmd(my_float_t data, int index) {
    *PARAM_1_ADDR_FP = data;
    *PARAM_2_ADDR    = index;
    *TPU_CMD = SET_BN_ADD_SRAM_3;
    __asm__ volatile ("nop");
}

void reset_lans_cmd() {
    *TPU_CMD = RESET_LANS;
    __asm__ volatile ("nop");
    __asm__ volatile ("nop");
    while((*BUSY_ADDR));
    __asm__ volatile ("nop");
}

void set_lans_idx_cmd() {
    *TPU_CMD = SET_LANS_IDX;
    __asm__ volatile ("nop");
}

void set_sram_next_cmd() {
    *TPU_CMD = SRAM_NEXT;
    __asm__ volatile ("nop");
}

void set_pooling_start_cmd() {
    *TPU_CMD = POOLING_START;
    __asm__ volatile ("nop");
}

void reset_pooling_idx_cmd() {
    *TPU_CMD = RESET_POOLING_IDX;
    __asm__ volatile ("nop");
}

void wait_idle_2_quick_cmd() {
    while((*BUSY_ADDR_2));
    __asm__ volatile ("nop");

}


void set_read_offset_cmd(int val) {
    *READ_OFFSET = val;
    __asm__ volatile ("nop");
}

void set_read_rounds_cmd(int val) {
    *READ_ROUNDS = val;
    __asm__ volatile ("nop");
}