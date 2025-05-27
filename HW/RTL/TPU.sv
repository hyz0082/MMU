`timescale 1ns / 1ps
// =============================================================================
//  Program : TPU.sv
//  Author  : 
//  Date    : 
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// =============================================================================
// `include "config.vh"
// address
// C40X_XXXX buffer 1~4
// C41X_XXXX index ram
// 

module TPU
#(parameter ACLEN=8,
  parameter ADDR_BITS=16,
  parameter DATA_WIDTH = 32
)
(
    /////////// System signals   ///////////////////////////////////////////////
    input   logic                        clk_i, rst_i,
    /////////// MMU command ///////////////////////////////////////////////
    input   logic                        tpu_cmd_valid,     // tpu valid
    input   logic   [ACLEN-1 : 0]        tpu_cmd,           // tpu
    input   logic   [DATA_WIDTH-1 : 0]   tpu_param_1_in,    // data 1
    input   logic   [DATA_WIDTH-1 : 0]   tpu_param_2_in,     // data 2

    // partial load
    input   logic   [DATA_WIDTH*4-1 : 0]   tpu_data_1_in,
    input   logic   [DATA_WIDTH*4-1 : 0]   tpu_data_2_in,
    input   logic   [DATA_WIDTH*4-1 : 0]   tpu_data_3_in,
    input   logic   [DATA_WIDTH*4-1 : 0]   tpu_data_4_in,

    output  logic   [DATA_WIDTH*4-1 : 0]   tpu_data_1_out,
    output  logic   [DATA_WIDTH*4-1 : 0]   tpu_data_2_out,
    output  logic   [DATA_WIDTH*4-1 : 0]   tpu_data_3_out,
    output  logic   [DATA_WIDTH*4-1 : 0]   tpu_data_4_out,
    /////////// DRAM input  ///////////////////////////////////////////////
    // input   logic   [DATA_WIDTH-1 : 0]   data_1_in,
    // input   logic   [DATA_WIDTH-1 : 0]   data_2_in,
    // input   logic   [DATA_WIDTH-1 : 0]   data_3_in,
    // input   logic   [DATA_WIDTH-1 : 0]   data_4_in,
    // input   logic   [DATA_WIDTH-1 : 0]   weight_1_in,
    // input   logic   [DATA_WIDTH-1 : 0]   weight_2_in,
    // input   logic   [DATA_WIDTH-1 : 0]   weight_3_in,
    // input   logic   [DATA_WIDTH-1 : 0]   weight_4_in,
    /////////// TPU outpupt ///////////////////////////////////////////////
    output  logic                      ret_valid,
    output  logic   [DATA_WIDTH-1 : 0] ret_data_out,

    output  logic   [DATA_WIDTH-1 : 0] ret_max_pooling,
    output  logic   [DATA_WIDTH-1 : 0] ret_avg_pooling,
    output  logic   [DATA_WIDTH-1 : 0] ret_softmax_result,
    
    // output  logic   [DATA_WIDTH*4-1 : 0] rdata_2_out,
    // output  logic   [DATA_WIDTH*4-1 : 0] rdata_3_out,
    // output  logic   [DATA_WIDTH*4-1 : 0] rdata_4_out, 

    output  logic                        tpu_busy     // 0->idle, 1->busy
);

//#########################
//    TPU cmd table
//#########################
localparam  RESET             = 0;  // whole
localparam  TRIGGER_CONV      = 1;  // whole
localparam  TRIGGER_CONV_LAST = 2;  // whole
localparam  SET_MUL_VAL       = 3;  // tpu_cmd   : 7
                                   // param_1_in: column number
                                   // param_2_in: multipled value
localparam  SET_ADD_VAL       = 4;  // partial
                                   // param_1_in: column number
                                   // param_2_in: added value
localparam  SET_PE_VAL        = 5;  // partial
                                   // param_1_in: PE number
                                   // param_2_in: index
localparam  SET_CONV_MODE     = 6; // whole
localparam  SET_FIX_MAC_MODE  = 7; // whole
localparam  FORWARD           = 8;
localparam  SW_WRITE_DATA     = 9;  // tpu_cmd   : 2
                                   // param_1_in: index
                                   // param_2_in: data
localparam  SW_WRITE_WEIGHT   = 10;  // tpu_cmd   : 3
                                    // param_1_in: index
                                    // param_2_in: data

localparam  SW_WRITE_I    = 11;  // tpu_cmd   : 4
                                   // param_1_in: index
                                   // param_2_in: data
localparam  SET_ROW_IDX       = 12; // partial
                                   // param_1_in: idx (0~4)
                                   // param_2_in: value
localparam  SET_KMN           = 13; // partial
                                   // param_1_in: idx (0:K, 1:M, 2:N)
                                   // param_2_in: value
// localparam  SET_ST_IDX        = 13; // partial
//                                    // param_1_in: idx (0~4)
//                                    // param_2_in: value
localparam  SW_READ_DATA       = 14; // whole
localparam  SET_PRELOAD        = 15; // whole
localparam  SW_WRITE_PARTIAL   = 16; // whole
localparam  TRIGGER_BN   = 17; // whole
localparam  SET_MAX_POOLING = 18; // whole
localparam  SET_DIVISOR = 19; // whole
localparam  SET_SOFTMAX = 20; // whole
localparam  TRIGGER_SOFTMAX = 21; // whole
localparam  SET_MODE = 22; // param_1: mode
                           // param_2: len
localparam  TRIGGER_ADD = 23; 
localparam  SET_RELU = 24;
localparam  SET_AVERAGE_POOLING = 25;
localparam  SET_BN_MUL_SRAM_0 = 26;
localparam  SET_BN_MUL_SRAM_1 = 27;
localparam  SET_BN_MUL_SRAM_2 = 28;
localparam  SET_BN_MUL_SRAM_3 = 29;
localparam  SET_BN_ADD_SRAM_0 = 30;
localparam  SET_BN_ADD_SRAM_1 = 31;
localparam  SET_BN_ADD_SRAM_2 = 32;
localparam  SET_BN_ADD_SRAM_3 = 33;

localparam  SET_I_OFFSET_1    = 34;
localparam  SET_I_OFFSET_2    = 35;
localparam  SET_I_OFFSET_3    = 36;
localparam  SET_I_OFFSET_4    = 37;

localparam  SET_COL_IDX       = 38; 

typedef enum {IDLE_S, HW_RESET_S, 
              LOAD_IDX_S,
              READ_DATA_1_S, READ_DATA_2_S, 
              READ_DATA_3_S, READ_DATA_4_S,
              READ_DATA_5_S,
              TRIGGER_S, TRIGGER_LAST_S,
              FORWARD_S,
              WAIT_IDLE_S,
              START_BATCHNORM_S,
              WAIT_BATCHNORM_S,
              STORE_S, 
              NEXT_ROW_S,
              NEXT_COL_S,
              PRELOAD_DATA_S,
              BN_S,
              FMA_1_S,
              FMA_2_S,
              FMA_3_S,
              FMA_WAIT_IDLE_S,
              START_POOLING_S,
              WAIT_POOLING_S,
              AVG_POOLING_ACC_S,
              WAIT_AVG_POOLING_ACC_S,
              AVG_POOLING_DIV_S,
              WAIT_AVG_POOLING_DIV_S,
              WAIT_BN_S,
              STORE_BN_S,
              BN_INC_S,
              BN_INC_2_S,
              BN_INC_3_S,
              WAIT_MAX_POOLING_S,
              SET_MUL_VAL_S, 
              SET_ADD_VAL_S, 
              SET_PE_VAL_S, 
              SET_CONV_MODE_S, 
              SET_FIX_MAC_MODE_S,
              SW_READ_DATA_S,
              WAIT_SF_ACC_S,
              WAIT_SF_S,
              OUTPUT_1_S,
              OUTPUT_2_S,
              OUTPUT_3_S} state_t;
(* mark_debug="true" *)    state_t curr_state, next_state;


typedef enum {SW_READ,   SW_WRITE,
              TPU_READ,  TPU_WRITE,
              DRAM_READ, DRAM_WRITE} gbuff_state_t;
gbuff_state_t gbuff_1_status, gbuff_2_status,
              gbuff_3_status, gbuff_4_status;
gbuff_state_t gbuff_status  [0 : 3];
gbuff_state_t weight_status [0 : 3];
gbuff_state_t I_status      [0 : 3];
gbuff_state_t P_status      [0 : 3];

logic                        tpu_cmd_valid_reg; // tpu valid
logic   [ACLEN : 0]          tpu_cmd_reg;       // tpu
logic   [DATA_WIDTH-1 : 0]   param_1_in_reg;    // data 1
logic   [DATA_WIDTH-1 : 0]   param_2_in_reg;    // data 2

logic   [DATA_WIDTH*4-1 : 0] rdata_out [0 : 3];

logic                        mmu_busy;     // 0->idle, 1->busy

//#########################
//#    INDEX START REG    #
//#########################
/*
row_idx: base address
*/
logic [ADDR_BITS-1  : 0] row_idx       [0 : 3];
logic [ADDR_BITS-1  : 0] row_acc;
logic [ADDR_BITS-1  : 0] row_offset;
logic [ADDR_BITS-1  : 0] row_idx_start [0 : 3];
logic [ADDR_BITS-1  : 0] col_idx       [0 : 3];
logic [ADDR_BITS-1  : 0] col_acc;
logic [ADDR_BITS-1  : 0] col_offset;
logic [ADDR_BITS-1  : 0] col_idx_start [0 : 3];
logic [DATA_WIDTH-1 : 0] row_input     [0 : 3];
logic [DATA_WIDTH-1 : 0] col_input     [0 : 3];

logic [ADDR_BITS-1 : 0] store_idx [0 : 3];

//#########################
//#    K     M     N      #
//#########################
logic [ADDR_BITS-1 : 0] K_reg, M_reg, N_reg; // (M * K) (K * N)

logic                        mmu_cmd_valid;  // cmd
logic [ACLEN : 0]            mmu_cmd;        // cmd
logic [DATA_WIDTH*4-1 : 0]   mmu_param_in [0 : 3]; // data 1
// logic   [DATA_WIDTH*4-1 : 0]   mmu_param_2_in; // data 2


//#########################
//#         GBUFF         #
//#########################
logic                        gbuff_wr_en        [0 : 3];
logic   [ADDR_BITS-1  : 0]   gbuff_index        [0 : 3];
logic   [DATA_WIDTH-1 : 0]   gbuff_data_in      [0 : 3];
logic   [DATA_WIDTH-1 : 0]   gbuff_data_out     [0 : 3];
logic   [DATA_WIDTH-1 : 0]   gbuff_data_out_reg [0 : 3];

//#########################
//#        WEIGHT         #
//#########################
logic                        weight_wr_en   [0 : 3];
logic   [ADDR_BITS-1  : 0]   weight_index   [0 : 3];
logic   [DATA_WIDTH-1 : 0]   weight_in      [0 : 3];
logic   [DATA_WIDTH-1 : 0]   weight_out     [0 : 3];
logic   [DATA_WIDTH-1 : 0]   weight_out_reg [0 : 3];

//#########################
//#        INDEX          #
//#########################
logic                        I_wr_en   [0 : 3];
logic   [ADDR_BITS-1  : 0]   I_index   [0 : 3];
logic   [DATA_WIDTH-1 : 0]   I_in      [0 : 3];
logic   [DATA_WIDTH-1 : 0]   I_out     [0 : 3];
logic   [DATA_WIDTH-1 : 0]   I_out_reg [0 : 3];

logic   [ADDR_BITS-1 : 0]   I_out_offset_index [0 : 3];
logic   [ADDR_BITS-1 : 0]   I_out_offset_out   [0 : 3];
logic   [ADDR_BITS-1 : 0]   I_out_offset_out_r [0 : 3];

//#########################
//#      PARTIAL SUM      #
//#########################
logic                        P_wr_en        [0 : 3];
logic   [ADDR_BITS-1    : 0] P_index        [0 : 3];
logic   [DATA_WIDTH*4-1 : 0] P_data_in      [0 : 3];
logic   [DATA_WIDTH*4-1 : 0] P_data_out     [0 : 3];
logic   [DATA_WIDTH*4-1 : 0] P_data_out_reg [0 : 3];

logic   [ADDR_BITS-1    : 0] P_index_reg [0 : 3];


//#########################
//#    CONV CTRL SIGNAL   #
//#########################
logic   [ADDR_BITS-1  : 0]   K_cnt;
logic conv_end, conv_next_row, conv_next_col;
logic preload;

logic [8:0] sa_in_cnt, sa_forward_cnt;

//#########################
//#    BN CTRL SIGNAL   #
//#########################
/*
 * mode: 0 -> conv
 * mode: 1 -> BatchNorm
 * mode: 2 -> skip add
 */
logic   [DATA_WIDTH*4-1 : 0]   tpu_data [0 : 3];
logic [1 : 0] mode;
logic   [ADDR_BITS-1  : 0]   bn_len;
logic   [ADDR_BITS-1  : 0]   bn_cnt;
logic   [DATA_WIDTH*4-1 : 0] bn_data_out [0 : 3];
logic bn_valid;
logic bn_valid_reg;

/*
 * BatchNorm and add signal (follow by GeMM)
 * 4x4 FMA (same as GeMM)
 * 4 sram for mul value
 * 4 sram for add value
 */
(* mark_debug="true" *)    logic   [DATA_WIDTH-1   : 0]  bn_fma_a_data    [0 : 15];
(* mark_debug="true" *)    logic                         bn_fma_a_valid   [0 : 15];
logic   [DATA_WIDTH-1   : 0]  bn_fma_b_data    [0 : 15];
logic                         bn_fma_b_valid   [0 : 15];
logic   [DATA_WIDTH-1   : 0]  bn_fma_c_data    [0 : 15];
logic                         bn_fma_c_valid   [0 : 15];
logic   [DATA_WIDTH-1   : 0]  bn_fma_out       [0 : 15];
logic   [DATA_WIDTH-1   : 0]  bn_fma_out_relu  [0 : 15];
logic                         bn_fma_out_valid [0 : 15];

logic                         bn_fma_out_valid_r [0 : 15];
logic   [DATA_WIDTH*4-1 : 0]  bn_fma_out_r      [0 : 3];

/*
 * BatchNorm mul & add sram signal
 */
(* mark_debug="true" *)logic [3 : 0]       bn_mul_wren;
(* mark_debug="true" *)logic [3 : 0]       bn_add_wren;
logic                         bn_source         [0 : 3];
logic   [ADDR_BITS-1  : 0]    bn_sram_idx       [0 : 3];
logic   [DATA_WIDTH-1 : 0]    bn_mul_sram_in    [0 : 3];
logic   [DATA_WIDTH-1 : 0]    bn_add_sram_in    [0 : 3];
logic   [DATA_WIDTH-1 : 0]    bn_mul_sram_out   [0 : 3];
logic   [DATA_WIDTH-1 : 0]    bn_mul_sram_out_r [0 : 3];
logic   [DATA_WIDTH-1 : 0]    bn_add_sram_out   [0 : 3];
logic   [DATA_WIDTH-1 : 0]    bn_add_sram_out_r [0 : 3];

logic   [DATA_WIDTH-1 : 0]    bn_mul_val_r [0 : 3];
logic   [DATA_WIDTH-1 : 0]    bn_add_val_r [0 : 3];

/*
 * BatchNorm and add signal 
 * 
 */
logic   [DATA_WIDTH-1   : 0]  fma_a_data  [0 : 3];
logic                         fma_a_valid [0 : 3];
logic   [DATA_WIDTH-1   : 0]  fma_b_data  [0 : 3];
logic                         fma_b_valid [0 : 3];
logic   [DATA_WIDTH-1   : 0]  fma_c_data  [0 : 3];
logic                         fma_c_valid [0 : 3];
logic   [DATA_WIDTH-1   : 0]  fma_out        [0 : 3];
logic   [DATA_WIDTH-1   : 0]  fma_out_relu   [0 : 3];
logic                         fma_out_valid   [0 : 3];
logic                         fma_out_valid_r [0 : 3];
logic   [DATA_WIDTH*4-1 : 0]  fma_out_r;
logic   [ADDR_BITS-1    : 0]  sram_r_idx [0 : 3];
logic   [ADDR_BITS-1    : 0]  sram_w_idx;

// logic   [ADDR_BITS-1    : 0]  calc_num;
(* mark_debug="true" *)    logic                         relu_en;

logic   [ADDR_BITS-1    : 0]  send_cnt;
logic   [ADDR_BITS-1    : 0]  recv_cnt;
logic   [ADDR_BITS-1    : 0]  calc_len;

logic   [DATA_WIDTH-1 : 0] mul_val_r;
logic   [DATA_WIDTH-1 : 0] add_val_r;


/*
 * MAX / AVERAGE POOLING SIGNAL (NEW)
 * 9 COMPARATER FOR MAX POOLING
 * 1 ACCUMULATOR FOR AVERAGE POOLING
 */
logic   [DATA_WIDTH-1   : 0] pooling_data_r [0 : 51];
logic   [ADDR_BITS-1    : 0] pooling_index;
logic   [DATA_WIDTH-1   : 0] pooling_result;
logic pooling_type; // 0: max pooling, 1: average pooling
logic   [DATA_WIDTH-1   : 0] acc_data_in , div_data_in;
logic   [DATA_WIDTH-1   : 0] acc_data_out, div_data_out;
logic                        acc_data_in_valid, div_data_valid;
logic                        acc_data_out_valid;
logic                        acc_data_in_last, acc_data_out_last;
logic   [DATA_WIDTH-1   : 0] avg_pooling_data;

logic   [ADDR_BITS-1    : 0] acc_recv_cnt;

/*
 * MAX POOLING SIGNAL (OLD)
 */
logic enable_max_pooling;
logic enable_avg_pooling;
logic   [DATA_WIDTH-1 : 0] max_pooling_data [0 : 19];
logic                      max_pooling_data_valid [0 : 20];
logic   [7 : 0] cmp_result [0 : 20];
logic                        cmp_result_valid [0 : 20];
logic                        cmp_result_valid_reg [0 : 20];
logic                        cmp_in_valid [0 : 20];
// logic   [DATA_WIDTH-1 : 0  ] max_pooling_result;
// 3x3
// 1 2 3 4 5 6 7 8 9
// 12 34 56 78 9 L1
// 1234 5678 9   L2
// 12345678 9    L3
// 123456789     L4
//#########################
//       SOFTMAX
//#########################
// divisor 
logic   [DATA_WIDTH-1 : 0]   divisor;
logic   [DATA_WIDTH-1 : 0]   exp_acc, exp_acc_reg;
logic                        exp_acc_valid, exp_acc_last;
logic   [DATA_WIDTH-1 : 0]   exp_1_in, exp_2_in;
logic   [DATA_WIDTH-1 : 0]   exp_1_out, exp_2_out;
logic                        exp_1_out_valid, exp_2_out_valid;
logic                        exp_1_out_valid_reg, exp_2_out_valid_reg;
logic   [DATA_WIDTH-1 : 0]   exp_1_out_reg, exp_2_out_reg;
// output
logic   [DATA_WIDTH-1 : 0]   softmax_result;
logic                        softmax_result_valid;

assign conv_end      = (row_acc + 4 >= M_reg && col_acc + 4 >= N_reg);
assign conv_next_row = (row_acc + 4 < M_reg);
assign conv_next_col = (col_acc + 4 < N_reg);

always_ff @( posedge clk_i ) begin
    if(tpu_cmd_valid && tpu_cmd == SET_CONV_MODE) begin
        mode <= 0;
    end
    else if(tpu_cmd_valid && tpu_cmd == SET_FIX_MAC_MODE) begin
        mode <= 1;
        bn_len <= tpu_param_1_in;
    end
    else if(tpu_cmd_valid && tpu_cmd == SET_MODE) begin
        mode <= tpu_param_1_in;
    end
end

always_ff @( posedge clk_i ) begin
    if(tpu_cmd_valid && tpu_cmd == SET_PRELOAD) begin
        preload <= tpu_param_1_in;
    end
    
end

// max pooling
always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        enable_max_pooling <= 0;
    end
    else if(tpu_cmd_valid && tpu_cmd == RESET) begin
        enable_max_pooling <= 0;
    end
    else if(tpu_cmd_valid && tpu_cmd == SET_MAX_POOLING) begin
        enable_max_pooling <= 1;
    end
end

// avg pooling
always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        enable_avg_pooling <= 0;
    end
    else if(tpu_cmd_valid && tpu_cmd == RESET) begin
        enable_avg_pooling <= 0;
    end
    else if(tpu_cmd_valid && tpu_cmd == SET_AVERAGE_POOLING) begin
        enable_avg_pooling <= 1;
    end
end


// error
always_ff @(posedge clk_i) begin
    tpu_cmd_valid_reg <= tpu_cmd_valid;
    if(tpu_cmd_valid) begin
        tpu_cmd_reg       <= tpu_cmd;  
        param_1_in_reg    <= tpu_param_1_in;
        // don't update  
        param_2_in_reg    <= tpu_param_2_in;
    end
end

always_ff @(posedge clk_i) begin
    if(rst_i) begin
        curr_state <= IDLE_S;
    end
    else begin
        curr_state <= next_state;
    end
end
//#########################
//   NEXT STATE LOGIC
//#########################
always_comb begin
    case (curr_state)
    IDLE_S: if(tpu_cmd_valid && tpu_cmd == TRIGGER_CONV    ) next_state = LOAD_IDX_S;
            // else if(tpu_cmd_valid && tpu_cmd == SET_MUL_VAL     ) next_state = SET_MUL_VAL_S;
            // else if(tpu_cmd_valid && tpu_cmd == SET_ADD_VAL     ) next_state = SET_ADD_VAL_S;
            // else if(tpu_cmd_valid && tpu_cmd == SET_PE_VAL    ) next_state = SET_PE_VAL_S;
            // else if(tpu_cmd_valid && tpu_cmd == SET_CONV_MODE   ) next_state = SET_CONV_MODE_S;
            // else if(tpu_cmd_valid && tpu_cmd == SET_FIX_MAC_MODE) next_state = SET_FIX_MAC_MODE_S;
            else if(tpu_cmd_valid && tpu_cmd == SW_READ_DATA    ) next_state = SW_READ_DATA_S;
            else if(tpu_cmd_valid && tpu_cmd == TRIGGER_BN    ) next_state = BN_S;
            else if(tpu_cmd_valid && tpu_cmd == SET_SOFTMAX    ) next_state = WAIT_SF_ACC_S;
            else if(tpu_cmd_valid && tpu_cmd == TRIGGER_SOFTMAX    ) next_state = WAIT_SF_S;
            else if(tpu_cmd_valid && tpu_cmd == TRIGGER_ADD) next_state = FMA_1_S;
            else         next_state = IDLE_S;
    HW_RESET_S    : next_state = IDLE_S;
    LOAD_IDX_S    : next_state = PRELOAD_DATA_S;
    PRELOAD_DATA_S: next_state = READ_DATA_1_S;
    READ_DATA_1_S : next_state = READ_DATA_2_S;
    READ_DATA_2_S : next_state = READ_DATA_3_S;
    READ_DATA_3_S : next_state = READ_DATA_4_S;
    READ_DATA_4_S : next_state = READ_DATA_5_S;
    READ_DATA_5_S : next_state = TRIGGER_S;
    // READ_DATA_6_S : next_state = TRIGGER_S;
    TRIGGER_S  : if(K_cnt == K_reg - 2) next_state = TRIGGER_LAST_S;
                    else                   next_state = TRIGGER_S;
    TRIGGER_LAST_S : next_state = FORWARD_S;
    FORWARD_S: if(sa_forward_cnt == 6)  next_state = WAIT_IDLE_S;
               else                     next_state = FORWARD_S;
    WAIT_IDLE_S: if   (mmu_busy) next_state = WAIT_IDLE_S;
                 else            next_state = START_BATCHNORM_S;
    START_BATCHNORM_S: next_state = WAIT_BATCHNORM_S;
    WAIT_BATCHNORM_S : if(bn_fma_out_valid[0]) next_state = STORE_S;
                       else next_state = WAIT_BATCHNORM_S;
    STORE_S: if(conv_end)           next_state = IDLE_S;
             else if(conv_next_row) next_state = NEXT_ROW_S;
             else                   next_state = NEXT_COL_S;
    NEXT_ROW_S: next_state = PRELOAD_DATA_S;
    NEXT_COL_S: next_state = PRELOAD_DATA_S;
    BN_S: next_state = WAIT_BN_S;
    WAIT_BN_S: if(bn_valid && !enable_max_pooling) next_state = STORE_BN_S;
               else if(cmp_result_valid_reg[7] && enable_max_pooling) next_state = IDLE_S;
               else next_state = WAIT_BN_S;
    STORE_BN_S: if(P_index_reg[0] == bn_len - 1) next_state = IDLE_S;
                else next_state = BN_INC_S;
    BN_INC_S: next_state = BN_INC_2_S;
    BN_INC_2_S: next_state = BN_INC_3_S;
    BN_INC_3_S: next_state = BN_S;
    FMA_1_S: next_state = FMA_2_S;
    FMA_2_S: next_state = FMA_3_S;
    FMA_3_S: if(send_cnt + 4 >= calc_len) 
                next_state = FMA_WAIT_IDLE_S;
             else
                next_state = FMA_3_S;
    FMA_WAIT_IDLE_S: if(recv_cnt + 4 >= calc_len && !enable_max_pooling && !enable_avg_pooling)
                        next_state = IDLE_S;
                     else if(recv_cnt + 4 >= calc_len && enable_max_pooling)
                        next_state = START_POOLING_S;
                     else if(recv_cnt + 4 >= calc_len && enable_avg_pooling)
                        next_state = AVG_POOLING_ACC_S;
                     else
                        next_state = FMA_WAIT_IDLE_S;
    START_POOLING_S: next_state = WAIT_POOLING_S;
    WAIT_POOLING_S : if(cmp_result_valid_reg[7]) 
                        next_state = IDLE_S;
                     else 
                        next_state = WAIT_POOLING_S;
    AVG_POOLING_ACC_S: if(pooling_index == 48)
                            next_state = WAIT_AVG_POOLING_ACC_S;
                       else 
                            next_state = AVG_POOLING_ACC_S;
    WAIT_AVG_POOLING_ACC_S: if(acc_recv_cnt == 49)
                                next_state = AVG_POOLING_DIV_S;
                            else
                                next_state = WAIT_AVG_POOLING_ACC_S;
    AVG_POOLING_DIV_S: next_state = WAIT_AVG_POOLING_DIV_S;
    WAIT_AVG_POOLING_DIV_S: if(div_data_valid) next_state = IDLE_S;
                            else next_state = WAIT_AVG_POOLING_DIV_S;
    WAIT_SF_ACC_S: if(exp_acc_valid) next_state = IDLE_S;
                   else next_state = WAIT_SF_ACC_S;
    WAIT_SF_S: if(softmax_result_valid) next_state = IDLE_S;
               else next_state = WAIT_SF_S;
    SW_READ_DATA_S: next_state = OUTPUT_1_S;
    OUTPUT_1_S    : next_state = OUTPUT_2_S;
    OUTPUT_2_S    : next_state = OUTPUT_3_S;
    OUTPUT_3_S    : next_state = IDLE_S;
    default: next_state = IDLE_S;
    endcase
end

assign tpu_busy = (curr_state != IDLE_S);

always_ff @( posedge clk_i ) begin
    if(curr_state == IDLE_S || curr_state == NEXT_ROW_S || curr_state == NEXT_COL_S) begin
        K_cnt <= 0;
    end
    else if(tpu_cmd_valid && tpu_cmd == RESET) begin
        K_cnt <= 0;
    end
    else if(curr_state == TRIGGER_S || curr_state == TRIGGER_LAST_S) begin
        K_cnt <= K_cnt + 1;
    end
end

//#########################
//#  SA_FORWARD COUNTER   #
//#########################
always_ff @(posedge clk_i) begin
    if(rst_i) begin
        sa_forward_cnt <= 0; 
    end
    else if(curr_state == NEXT_ROW_S || curr_state == NEXT_COL_S || curr_state == IDLE_S) begin
        sa_forward_cnt <= 0;
    end
    else if(curr_state == FORWARD_S) begin
        sa_forward_cnt <= sa_forward_cnt + 1;
    end
end

//#########################
//#       STORE KMN       #
//#########################
always_ff @(posedge clk_i) begin
    if(tpu_cmd_valid && tpu_cmd == SET_KMN) begin
        if(tpu_param_1_in == 0) begin
            K_reg <= tpu_param_2_in;
        end
        else if(tpu_param_1_in == 1) begin
            M_reg <= tpu_param_2_in;
        end
        else if(tpu_param_1_in == 2) begin
            N_reg <= tpu_param_2_in;
        end
    end
end


//#########################
//#      SEND OUTPUT      #
//#########################
always_comb begin
    tpu_data_1_out = P_data_out_reg[0];
    tpu_data_2_out = P_data_out_reg[1];
    tpu_data_3_out = P_data_out_reg[2];
    tpu_data_4_out = P_data_out_reg[3];
    if(curr_state == OUTPUT_3_S) begin
        ret_valid = 1;
        case (param_1_in_reg)  //DATA_WIDTH
            0 : ret_data_out = P_data_out_reg[0][(DATA_WIDTH*4-1)-:DATA_WIDTH]; // 127:96
            1 : ret_data_out = P_data_out_reg[0][(DATA_WIDTH*3-1)-:DATA_WIDTH]; // 95:64
            2 : ret_data_out = P_data_out_reg[0][(DATA_WIDTH*2-1)-:DATA_WIDTH]; // 63:32
            3 : ret_data_out = P_data_out_reg[0][(DATA_WIDTH*1-1)-:DATA_WIDTH]; // 31:0
            4 : ret_data_out = P_data_out_reg[1][(DATA_WIDTH*4-1)-:DATA_WIDTH];
            5 : ret_data_out = P_data_out_reg[1][(DATA_WIDTH*3-1)-:DATA_WIDTH];
            6 : ret_data_out = P_data_out_reg[1][(DATA_WIDTH*2-1)-:DATA_WIDTH];
            7 : ret_data_out = P_data_out_reg[1][(DATA_WIDTH*1-1)-:DATA_WIDTH];
            8 : ret_data_out = P_data_out_reg[2][(DATA_WIDTH*4-1)-:DATA_WIDTH];
            9 : ret_data_out = P_data_out_reg[2][(DATA_WIDTH*3-1)-:DATA_WIDTH];
            10: ret_data_out = P_data_out_reg[2][(DATA_WIDTH*2-1)-:DATA_WIDTH];
            11: ret_data_out = P_data_out_reg[2][(DATA_WIDTH*1-1)-:DATA_WIDTH];
            12: ret_data_out = P_data_out_reg[3][(DATA_WIDTH*4-1)-:DATA_WIDTH];
            13: ret_data_out = P_data_out_reg[3][(DATA_WIDTH*3-1)-:DATA_WIDTH];
            14: ret_data_out = P_data_out_reg[3][(DATA_WIDTH*2-1)-:DATA_WIDTH];
            15: ret_data_out = P_data_out_reg[3][(DATA_WIDTH*1-1)-:DATA_WIDTH];
            default: ret_data_out = 0;
        endcase
    end
    else begin
        ret_valid = 0;
        ret_data_out = 0;
    end
end

//#########################
//#       SEND INPUT      #
//#########################
always_ff @(posedge clk_i) begin
    for (int i = 0; i < 4; i++) begin
        if( curr_state == TRIGGER_LAST_S ||
            curr_state == FORWARD_S) begin
            row_input[i] <= 0;
        end
        else begin
            row_input[i] <= gbuff_data_out[i];
        end
        
        col_input[i] <= weight_out[i];
    end
end

//#########################
//#   ROW INDEX  START    #
//#########################
always_ff @(posedge clk_i) begin
    if(tpu_cmd_valid && tpu_cmd == SET_ROW_IDX)begin
        row_idx_start[tpu_param_1_in] <= tpu_param_2_in;
    end
end

//#########################
//#       ROW INDEX       #
//#########################
always_ff @(posedge clk_i) begin
    if(curr_state == LOAD_IDX_S) begin
        for(int i = 0; i < 4; i++) begin
            row_idx[i] <= row_idx_start[i];
        end
    end
    else if(curr_state == NEXT_ROW_S) begin
        for(int i = 0; i < 4; i++) begin
            // row_idx[i] <= row_idx[i] + K_cnt * 4;
            row_idx[i] <= row_idx_start[i];
        end
    end
    else if(curr_state == NEXT_COL_S) begin
        for(int i = 0; i < 4; i++) begin
            row_idx[i] <= row_idx_start[i];
        end
    end
end

//#########################
//#      ROW OFFSET       # 
//#########################
always_ff @(posedge clk_i) begin
    if(curr_state == IDLE_S) begin
        row_offset <= 0;
    end
    else if(curr_state == NEXT_ROW_S || curr_state == NEXT_COL_S) begin
        row_offset <= 0;
    end
    else if(curr_state == PRELOAD_DATA_S||
            curr_state == READ_DATA_1_S || 
            curr_state == READ_DATA_2_S || 
            curr_state == READ_DATA_3_S || 
            curr_state == READ_DATA_4_S ||
            curr_state == READ_DATA_5_S || 
            curr_state == TRIGGER_S  ||
            curr_state == TRIGGER_LAST_S) begin
        row_offset <= row_offset + 1;
    end
end

//#########################
//#       ROW ACC         # 
//#########################
always_ff @(posedge clk_i) begin
    if(curr_state == IDLE_S) begin
        row_acc <= 0;
    end
    else if(curr_state == NEXT_ROW_S) begin
        row_acc <= row_acc + 4;
    end
    else if(curr_state == NEXT_COL_S) begin
        row_acc <= 0;
    end
end

//#########################
//#   COL INDEX  START    #
//#########################
always_ff @(posedge clk_i) begin
    if(tpu_cmd_valid && tpu_cmd == SET_COL_IDX)begin
        col_idx_start[tpu_param_1_in] <= tpu_param_2_in;
    end
end

//#########################
//#       COL INDEX       #
//#########################
always_ff @(posedge clk_i) begin
    if(curr_state == LOAD_IDX_S) begin
        for(int i = 0; i < 4; i++) begin
            col_idx[i] <= col_idx_start[i];
        end
    end
    else if(curr_state == NEXT_COL_S) begin
        for(int i = 0; i < 4; i++) begin
            col_idx[i] <= col_idx[i] + K_cnt * 4;
        end
    end
end

//#########################
//#      COL OFFSET       # 
//#########################
always_ff @(posedge clk_i) begin
    if(curr_state == IDLE_S) begin
        col_offset <= 0;
    end
    else if(curr_state == NEXT_ROW_S || curr_state == NEXT_COL_S ) begin
        col_offset <= 0;
    end
    else if(curr_state == READ_DATA_3_S ||
            curr_state == READ_DATA_4_S || 
            curr_state == READ_DATA_5_S || 
            curr_state == TRIGGER_S  ||
            curr_state == TRIGGER_LAST_S) begin
        col_offset <= col_offset + 1;
    end
end

//#########################
//#       COL ACC         # 
//#########################
always_ff @(posedge clk_i) begin
    if(curr_state == IDLE_S) begin
        col_acc <= 0;
    end
    // else if(curr_state == NEXT_ROW_S) begin
    //     col_acc <= 0;
    // end
    else if(curr_state == NEXT_COL_S) begin
        col_acc <= col_acc + 4;
    end
end

//#########################
//#      GBUFF STATUS     #
//#########################
always_comb begin
    for (int i = 0; i < 4; i++) begin
        if(tpu_cmd_valid_reg && tpu_cmd_reg == SW_WRITE_DATA) begin
            gbuff_status[i] = SW_WRITE;
        end
        else begin
            gbuff_status[i] = TPU_READ;
        end
    end
end

//#########################
//#      GBUFF INDEX      #
//#########################
always_ff @(posedge clk_i) begin
    for (int i = 0; i < 4; i++) begin
        gbuff_index[i] <= (gbuff_status[i] == SW_READ   ) ? param_2_in_reg
                        : (gbuff_status[i] == SW_WRITE  ) ? param_2_in_reg 
                        : (gbuff_status[i] == TPU_READ  ) ? I_out_reg[i] + I_out_offset_out_r[i]
                        : (gbuff_status[i] == TPU_WRITE ) ? 0 
                        : (gbuff_status[i] == DRAM_READ ) ? 0  
                        : (gbuff_status[i] == DRAM_WRITE) ? 0  
                        : 0;
    end
end
//#########################
//#      GBUFF WR EN      #
//#########################
always_ff @(posedge clk_i) begin
    for (int i = 0; i < 4; i++) begin
        gbuff_wr_en[i] <= (gbuff_status[i] == SW_READ   ) ? 0
                        : (gbuff_status[i] == SW_WRITE  ) ? 1 
                        : (gbuff_status[i] == TPU_READ  ) ? 0 
                        : (gbuff_status[i] == TPU_WRITE ) ? 1 
                        : (gbuff_status[i] == DRAM_READ ) ? 0  
                        : (gbuff_status[i] == DRAM_WRITE) ? 1  
                        : 0;
    end
end

//#########################
//#     GBUFF DATA IN     #
//#########################
always_ff @(posedge clk_i) begin
    for (int i = 0; i < 4; i++) begin
        gbuff_data_in[i] <= param_1_in_reg;
    end
end

//#########################
//#     WEIGHT STATUS     #
//#########################
always_comb begin
    for (int i = 0; i < 4; i++) begin
        if(tpu_cmd_valid_reg && tpu_cmd_reg == SW_WRITE_WEIGHT) begin
            weight_status[i] = SW_WRITE;
        end
        else begin
            weight_status[i] = TPU_READ;
        end
    end
end

//#########################
//#     WEIGHT INDEX      #
//#########################
always_ff @(posedge clk_i) begin
    for (int i = 0; i < 4; i++) begin
        weight_index[i] <= (weight_status[i] == SW_READ   ) ? param_2_in_reg
                         : (weight_status[i] == SW_WRITE  ) ? param_2_in_reg 
                         : (weight_status[i] == TPU_READ  ) ? col_idx[i] + col_offset
                         : (weight_status[i] == TPU_WRITE ) ? 0 
                         : (weight_status[i] == DRAM_READ ) ? 0  
                         : (weight_status[i] == DRAM_WRITE) ? 0  
                         : 0;
    end
end
//#########################
//#     WEIGHT WR EN      #
//#########################
always_ff @(posedge clk_i) begin
    for (int i = 0; i < 4; i++) begin
        weight_wr_en[i] <= (weight_status[i] == SW_READ   ) ? 0
                         : (weight_status[i] == SW_WRITE  ) ? 1 
                         : (weight_status[i] == TPU_READ  ) ? 0 
                         : (weight_status[i] == TPU_WRITE ) ? 1 
                         : (weight_status[i] == DRAM_READ ) ? 0  
                         : (weight_status[i] == DRAM_WRITE) ? 1  
                         : 0;
    end
end

//#########################
//#    WEIGHT DATA IN     #
//#########################
always_ff @(posedge clk_i) begin
    for (int i = 0; i < 4; i++) begin
        weight_in[i] <= param_1_in_reg;
    end
end

//#########################
//#       I STATUS        #
//#########################
always_comb begin
    for (int i = 0; i < 4; i++) begin
        if(tpu_cmd_valid_reg && tpu_cmd_reg == SW_WRITE_I) begin
            I_status[i] = SW_WRITE;
        end
        else begin
            I_status[i] = TPU_READ;
        end
    end
end

//#########################
//#       I INDEX         #
//#########################
always_ff @(posedge clk_i) begin
    for (int i = 0; i < 4; i++) begin
        I_index[i] <= (I_status[i] == SW_READ   ) ? param_2_in_reg
                    : (I_status[i] == SW_WRITE  ) ? param_2_in_reg 
                    : (I_status[i] == TPU_READ  ) ? row_idx[i] + row_offset
                    : (I_status[i] == TPU_WRITE ) ? 0 
                    : (I_status[i] == DRAM_READ ) ? 0  
                    : (I_status[i] == DRAM_WRITE) ? 0  
                    : 0;
    end
end
//#########################
//#       I WR EN         #
//#########################
always_ff @(posedge clk_i) begin
    for (int i = 0; i < 4; i++) begin
        I_wr_en[i] <= (I_status[i] == SW_READ   ) ? 0
                    : (I_status[i] == SW_WRITE  ) ? 1 
                    : (I_status[i] == TPU_READ  ) ? 0 
                    : (I_status[i] == TPU_WRITE ) ? 1 
                    : (I_status[i] == DRAM_READ ) ? 0  
                    : (I_status[i] == DRAM_WRITE) ? 1  
                    : 0;
    end
end

//#########################
//#      I DATA IN        #
//#########################
always_ff @(posedge clk_i) begin
    for (int i = 0; i < 4; i++) begin
        I_in[i] <= param_1_in_reg;
    end
end

//#########################
//#       P STATUS        #
//#########################
always_comb begin
    for (int i = 0; i < 4; i++) begin
        if(curr_state == STORE_S || curr_state == STORE_BN_S) begin
            P_status[i] = TPU_WRITE;
        end
        else if(tpu_cmd_valid_reg && tpu_cmd_reg == SW_READ_DATA) begin
            P_status[i] = SW_READ;
        end
        else if(tpu_cmd_valid_reg && tpu_cmd_reg == SW_WRITE_PARTIAL) begin
            P_status[i] = SW_WRITE;
        end
        else begin
            P_status[i] = TPU_READ;
        end
    end
end

//#########################
//#       P IDX REG       #
//#########################
always_ff @(posedge clk_i)begin
    for (int i = 0; i < 4; i++) begin
        if(curr_state == IDLE_S) begin
            P_index_reg[i] <= 0;
        end
        else if(curr_state == STORE_S || curr_state == STORE_BN_S) begin
            P_index_reg[i] <= P_index_reg[i] + 1;
        end
    end
end

//#########################
//#       P INDEX         #
//#########################
always_ff @(posedge clk_i) begin
    for (int i = 0; i < 4; i++) begin
        P_index[i] <= (P_status[i] == SW_READ   ) ? param_2_in_reg
                    : (P_status[i] == SW_WRITE  ) ? param_2_in_reg 
                    : (P_status[i] == TPU_READ  ) ? P_index_reg[i]
                    : (P_status[i] == TPU_WRITE ) ? P_index_reg[i]
                    : (P_status[i] == DRAM_READ ) ? 0  
                    : (P_status[i] == DRAM_WRITE) ? 0  
                    : 0;
    end
end
//#########################
//#       P WR EN         #
//#########################
always_ff @(posedge clk_i) begin
    for (int i = 0; i < 4; i++) begin
        P_wr_en[i] <= (P_status[i] == SW_READ   ) ? 0
                    : (P_status[i] == SW_WRITE  ) ? 1 
                    : (P_status[i] == TPU_READ  ) ? 0 
                    : (P_status[i] == TPU_WRITE ) ? 1 
                    : (P_status[i] == DRAM_READ ) ? 0  
                    : (P_status[i] == DRAM_WRITE) ? 1  
                    : 0;
    end
end

//#########################
//#      P DATA ST        #
//#########################
always_comb begin
        tpu_data[0] <= tpu_data_1_in;
        tpu_data[1] <= tpu_data_2_in;
        tpu_data[2] <= tpu_data_3_in;
        tpu_data[3] <= tpu_data_4_in;
end

//#########################
//#      P DATA IN        #
//#########################
always_ff @(posedge clk_i) begin
    for (int i = 0; i < 4; i++) begin
        P_data_in[i] <= (P_status[i] == SW_READ   ) ? 0
                      : (P_status[i] == SW_WRITE  ) ? tpu_data[i] 
                      : (P_status[i] == TPU_READ  ) ? 0 
                    //   : (P_status[i] == TPU_WRITE && mode == 0 ) ? rdata_out[i]
                      : (P_status[i] == TPU_WRITE && mode == 0 ) ? bn_fma_out_r[i]
                      : (P_status[i] == TPU_WRITE && mode == 1 ) ? bn_data_out[i]
                      : (P_status[i] == DRAM_READ ) ? 0  
                      : (P_status[i] == DRAM_WRITE) ? 0  
                      : 0;
    end
end

//#########################
//#      SRAM OUT         #
//#########################
always_ff @( posedge clk_i ) begin
    for (int i = 0; i < 4; i++) begin
        gbuff_data_out_reg[i] <= gbuff_data_out[i];
        weight_out_reg[i]     <= weight_out[i];
        I_out_reg[i]          <= I_out[i];
        P_data_out_reg[i]     <= P_data_out[i];
    end
end

//#########################
//#     MMU CMD VALID     #
//#########################
always_comb begin
    if((tpu_cmd_valid_reg && tpu_cmd_reg == RESET           ) || 
       (tpu_cmd_valid_reg && tpu_cmd_reg == SET_MUL_VAL     ) || 
       (tpu_cmd_valid_reg && tpu_cmd_reg == SET_ADD_VAL     ) ||
       (tpu_cmd_valid_reg && tpu_cmd_reg == SET_CONV_MODE   ) || 
       (tpu_cmd_valid_reg && tpu_cmd_reg == SET_FIX_MAC_MODE) ||
       (curr_state == TRIGGER_S                             ) ||
       (curr_state == TRIGGER_LAST_S                        ) ||
       (curr_state == FORWARD_S                             ) ||
       (curr_state == READ_DATA_2_S && preload)               ||
       (curr_state == BN_S)) begin
        mmu_cmd_valid = 1; 
    end
    else begin
        mmu_cmd_valid = 0;
    end
end
//#########################
//#       MMU CMD         #
//#########################
/*
preload error
*/
always_comb begin
    if(curr_state == FORWARD_S) begin
        mmu_cmd         = FORWARD;
    end
    else if(curr_state == READ_DATA_2_S) begin
        mmu_cmd = SET_PE_VAL;
    end
    else if(curr_state == BN_S) begin
        mmu_cmd = TRIGGER_BN;
    end
    else begin
        mmu_cmd         = tpu_cmd_reg;
    end
    if(curr_state == READ_DATA_2_S) begin
        mmu_param_in[0] = P_data_out_reg[0];
        mmu_param_in[1] = P_data_out_reg[1];
    end
    else begin
        mmu_param_in[0] = param_1_in_reg;
        mmu_param_in[1] = param_2_in_reg;
    end

    mmu_param_in[2] = P_data_out_reg[2];
    mmu_param_in[3] = P_data_out_reg[3];
    
end
//##########################
//# MATRIX MULTIPLE ENGINE #
//##########################
MMU #(
    .ACLEN(ACLEN), // ADDR_BITS
    .DATA_WIDTH(DATA_WIDTH)  // DATA_WIDTH
)
M1 (
    .clk_i(clk_i), 
    .rst_i(rst_i),
    .mmu_cmd_valid(mmu_cmd_valid), 
    .mmu_cmd(mmu_cmd),
    .param_1_in(mmu_param_in[0]),
    .param_2_in(mmu_param_in[1]),
    .param_3_in(mmu_param_in[2]),
    .param_4_in(mmu_param_in[3]),

    .data_1_in(row_input[0]),
    .data_2_in(row_input[1]),
    .data_3_in(row_input[2]),
    .data_4_in(row_input[3]),

    .weight_1_in(col_input[0]),
    .weight_2_in(col_input[1]),
    .weight_3_in(col_input[2]),
    .weight_4_in(col_input[3]),

    .rdata_1_out(rdata_out[0]),
    .rdata_2_out(rdata_out[1]),
    .rdata_3_out(rdata_out[2]),
    .rdata_4_out(rdata_out[3]),

    .bn_data_1_in(P_data_out_reg[0]),
    .bn_data_2_in(P_data_out_reg[1]),
    .bn_data_3_in(P_data_out_reg[2]),
    .bn_data_4_in(P_data_out_reg[3]),

    .bn_data_1_out(bn_data_out[0]),
    .bn_data_2_out(bn_data_out[1]),
    .bn_data_3_out(bn_data_out[2]),
    .bn_data_4_out(bn_data_out[3]),
    .bn_valid(bn_valid),

    .mmu_busy(mmu_busy)
);
//#########################
//       SOFTMAX
//#########################
always_ff @( posedge clk_i ) begin
    if(tpu_cmd_valid && tpu_cmd == SET_DIVISOR) begin
        divisor <= tpu_data_1_in;
    end
end
always_ff @( posedge clk_i ) begin
    exp_1_out_reg       <= exp_1_out;
    exp_2_out_reg       <= exp_2_out;
    exp_1_out_valid_reg <= exp_1_out_valid;
    exp_2_out_valid_reg <= exp_2_out_valid;

end
always_ff @( posedge clk_i ) begin
    if(exp_acc_valid) begin
        exp_acc_reg <= exp_acc;
    end
end
always_ff @( posedge clk_i ) begin
    if(softmax_result_valid) begin
        ret_softmax_result <= softmax_result;
    end
end
// exp 1
floating_point_exp exp_1(
        .aclk(clk_i),
        .s_axis_a_tdata(tpu_param_1_in),
        .s_axis_a_tvalid(tpu_cmd_valid && tpu_cmd == SET_SOFTMAX),
        .m_axis_result_tdata(exp_1_out),
        .m_axis_result_tvalid(exp_1_out_valid)
);
// exp 2
floating_point_exp exp_2(
        .aclk(clk_i),
        .s_axis_a_tdata(tpu_param_1_in),
        .s_axis_a_tvalid(tpu_cmd_valid && tpu_cmd == TRIGGER_SOFTMAX),
        .m_axis_result_tdata(exp_2_out),
        .m_axis_result_tvalid(exp_2_out_valid)
);
// acc
floating_point_acc ACC(

    .aclk(clk_i),

    .s_axis_a_tdata(exp_1_out_reg),
    .s_axis_a_tlast(tpu_param_2_in),
    .s_axis_a_tvalid(exp_1_out_valid_reg),

    .m_axis_result_tdata(exp_acc),
    .m_axis_result_tlast(exp_acc_last),
    .m_axis_result_tvalid(exp_acc_valid)
);
// divid
floating_point_div div(
        .aclk(clk_i),
        .s_axis_a_tdata(exp_2_out_reg),
        .s_axis_a_tvalid(exp_2_out_valid_reg),
        .s_axis_b_tdata(exp_acc_reg),
        .s_axis_b_tvalid(exp_2_out_valid_reg),
        .m_axis_result_tdata(softmax_result),
        .m_axis_result_tvalid(softmax_result_valid)
    );

//#########################
//    AVG / MAX POOLING
//#########################

/*
 * store BatchNorm output
 */
always_ff @( posedge clk_i ) begin
    if(fma_out_valid_r[0]) begin
        pooling_data_r[sram_w_idx*4  ] <= fma_out_r[(DATA_WIDTH*4-1)-:DATA_WIDTH];
        pooling_data_r[sram_w_idx*4+1] <= fma_out_r[(DATA_WIDTH*3-1)-:DATA_WIDTH];
        pooling_data_r[sram_w_idx*4+2] <= fma_out_r[(DATA_WIDTH*2-1)-:DATA_WIDTH];
        pooling_data_r[sram_w_idx*4+3] <= fma_out_r[(DATA_WIDTH*1-1)-:DATA_WIDTH];
    end
end

always_ff @( posedge clk_i ) begin
    bn_valid_reg <= bn_valid;
    cmp_result_valid_reg <= cmp_result_valid;
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        for (int i = 0; i < 20; i++) begin
            max_pooling_data[i] <= 0;
        end
    end
    else begin
        max_pooling_data[0]  <= pooling_data_r[0];
        max_pooling_data[1]  <= pooling_data_r[1];
        max_pooling_data[2]  <= pooling_data_r[2];
        max_pooling_data[3]  <= pooling_data_r[3];
        max_pooling_data[4]  <= pooling_data_r[4];
        max_pooling_data[5]  <= pooling_data_r[5];
        max_pooling_data[6]  <= pooling_data_r[6];
        max_pooling_data[7]  <= pooling_data_r[7];
        max_pooling_data[15] <= pooling_data_r[8];
    end
    // else if(bn_valid_reg) begin
    //     max_pooling_data[0]  <= bn_data_out[0][(DATA_WIDTH*4-1)-:DATA_WIDTH];//[127:31];//DATA_WIDTH
    //     max_pooling_data[1]  <= bn_data_out[1][(DATA_WIDTH*4-1)-:DATA_WIDTH];
    //     max_pooling_data[2]  <= bn_data_out[2][(DATA_WIDTH*4-1)-:DATA_WIDTH];
    //     max_pooling_data[3]  <= bn_data_out[3][(DATA_WIDTH*4-1)-:DATA_WIDTH];
    //     max_pooling_data[4]  <= bn_data_out[0][(DATA_WIDTH*3-1)-:DATA_WIDTH];
    //     max_pooling_data[5]  <= bn_data_out[1][(DATA_WIDTH*3-1)-:DATA_WIDTH];
    //     max_pooling_data[6]  <= bn_data_out[2][(DATA_WIDTH*3-1)-:DATA_WIDTH];
    //     max_pooling_data[7]  <= bn_data_out[3][(DATA_WIDTH*3-1)-:DATA_WIDTH];
    //     max_pooling_data[15] <= bn_data_out[0][(DATA_WIDTH*2-1)-:DATA_WIDTH];
    // end
    // 0 ~ 6
    for(int i = 0; i < 7; i++) begin
        if(cmp_result_valid_reg[i]) begin
            max_pooling_data[i+8]  <= (cmp_result[i]==1) ? max_pooling_data[i*2] 
                                                      : max_pooling_data[i*2+1];
        end
    end
end

always_ff @( posedge clk_i ) begin
    for (int i = 0; i < 4; i++) begin
        if(curr_state == START_POOLING_S) begin
            cmp_in_valid[i] <= 1;
        end
        else begin
            cmp_in_valid[i] <= 0;
        end 
    end
    cmp_in_valid[4]  <= cmp_result_valid_reg[0];
    cmp_in_valid[5]  <= cmp_result_valid_reg[2];

    cmp_in_valid[6]  <= cmp_result_valid_reg[4];

    cmp_in_valid[7] <= cmp_result_valid_reg[6];


end

// mid
generate
for (genvar i = 0; i < 7; i++) begin
    floating_point_cmp cmp(
        .aclk(clk_i),
        .s_axis_a_tdata(max_pooling_data[i*2]),
        .s_axis_a_tvalid(cmp_in_valid[i]),
        .s_axis_b_tdata(max_pooling_data[i*2+1]),
        .s_axis_b_tvalid(cmp_in_valid[i]),
        .m_axis_result_tdata(cmp_result[i]),
        .m_axis_result_tvalid(cmp_result_valid[i])
    );
end
endgenerate
// last
floating_point_cmp cmp(
        .aclk(clk_i),
        .s_axis_a_tdata(max_pooling_data[14]),
        .s_axis_a_tvalid(cmp_in_valid[7]),
        .s_axis_b_tdata(max_pooling_data[15]),
        .s_axis_b_tvalid(cmp_in_valid[7]),
        .m_axis_result_tdata(cmp_result[7]),
        .m_axis_result_tvalid(cmp_result_valid[7])
    );

always_ff @( posedge clk_i ) begin
    if(cmp_result_valid_reg[7]) begin
        ret_max_pooling  <= (cmp_result[7]==1) ? max_pooling_data[14] 
                                               : max_pooling_data[15];
    end
end

//#########################
//#      LINE BUFFER      #
//#########################
/*
 * single port ram
 */
// generate
// for (genvar i = 0; i < 4; i++) begin
//     global_buffer #(
//     .ADDR_BITS(ADDR_BITS), // ADDR_BITS
//     .DATA_BITS(DATA_WIDTH)  // DATA_WIDTH
//     )
//     gbuff (
//         .clk_i   (clk_i),
//         .rst_i   (rst_i),
//         .wr_en   ((mode) ? 0             : gbuff_wr_en[i]),
//         .index   ((mode) ? sram_r_idx[i] : gbuff_index[i]),
//         .data_in (gbuff_data_in[i]),
//         .data_out(gbuff_data_out[i])
//     );
// end
// endgenerate
/*
 * dua; port sram
 */
global_buffer_dp #(
.ADDR_BITS(ADDR_BITS), // ADDR_BITS
.DATA_BITS(DATA_WIDTH)  // DATA_WIDTH
)
gbuff_1 (
    .clk_i   (clk_i),

    .wr_en_1   ((mode) ? 0             : gbuff_wr_en[0]),
    .index_1   ((mode) ? sram_r_idx[0] : gbuff_index[0]),
    .data_in_1 (gbuff_data_in[0]),
    .data_out_1(gbuff_data_out[0]),

    .index_2((mode) ? sram_r_idx[1] : gbuff_index[1]),
    .data_in_2(gbuff_data_in[1]),
    .data_out_2(gbuff_data_out[1])
);

global_buffer_dp #(
.ADDR_BITS(ADDR_BITS), // ADDR_BITS
.DATA_BITS(DATA_WIDTH)  // DATA_WIDTH
)
gbuff_2 (
    .clk_i   (clk_i),

    .wr_en_1   ((mode) ? 0             : gbuff_wr_en[2]),
    .index_1   ((mode) ? sram_r_idx[2] : gbuff_index[2]),
    .data_in_1 (gbuff_data_in[2]),
    .data_out_1(gbuff_data_out[2]),

    .index_2((mode) ? sram_r_idx[3] : gbuff_index[3]),
    .data_in_2(gbuff_data_in[3]),
    .data_out_2(gbuff_data_out[3])
);

//#########################
//#     WEIGHT BUFFER     #
//#########################
/*
 * single port sram
 */
// generate
// for (genvar i = 0; i < 4; i++) begin
//     global_buffer #(
//     .ADDR_BITS(ADDR_BITS),
//     .DATA_BITS(DATA_WIDTH)
//     )
//     weight (
//         .clk_i   (clk_i),
//         .rst_i   (rst_i),
//         .wr_en   ((mode) ? 0             : weight_wr_en[i]),
//         .index   ((mode) ? sram_r_idx[i] : weight_index[i]),
//         .data_in (weight_in[i]),
//         .data_out(weight_out[i])
//     );
// end
// endgenerate
/*
 * dual port sram
 */
global_buffer_dp #(
.ADDR_BITS(ADDR_BITS), // ADDR_BITS
.DATA_BITS(DATA_WIDTH)  // DATA_WIDTH
)
weight_1 (
    .clk_i   (clk_i),

    .wr_en_1   ((mode) ? 0             : weight_wr_en[0]),
    .index_1   ((mode) ? sram_r_idx[0] : weight_index[0]),
    .data_in_1 (weight_in[0]),
    .data_out_1(weight_out[0]),

    .index_2((mode) ? sram_r_idx[1] : weight_index[1]),
    .data_in_2(weight_in[1]),
    .data_out_2(weight_out[1])
);

global_buffer_dp #(
.ADDR_BITS(ADDR_BITS), // ADDR_BITS
.DATA_BITS(DATA_WIDTH)  // DATA_WIDTH
)
weight_2 (
    .clk_i   (clk_i),

    .wr_en_1   ((mode) ? 0             : weight_wr_en[2]),
    .index_1   ((mode) ? sram_r_idx[2] : weight_index[2]),
    .data_in_1 (weight_in[2]),
    .data_out_1(weight_out[2]),

    .index_2((mode) ? sram_r_idx[3] : weight_index[3]),
    .data_in_2(weight_in[3]),
    .data_out_2(weight_out[3])
);

//#########################
//#       I BUFFER        #
//#########################
/*
 * single port sram
*/
// generate
// for (genvar i = 0; i < 4; i++) begin
//     global_buffer #(
//     .ADDR_BITS(ADDR_BITS),
//     .DATA_BITS(DATA_WIDTH)
//     )
//     I_gbuff (
//         .clk_i(clk_i),
//         .rst_i(rst_i),
//         .wr_en(I_wr_en[i]),
//         .index(I_index[i]),
//         .data_in(I_in[i]),
//         .data_out(I_out[i])
//     );
// end
// endgenerate
/*
 * dual port sram
 */
global_buffer_dp #(
.ADDR_BITS(13), // ADDR_BITS
.DATA_BITS(DATA_WIDTH)  // DATA_WIDTH
)
index_1 (
    .clk_i   (clk_i),

    .wr_en_1   (I_wr_en[0]),
    .index_1   (I_index[0]),
    .data_in_1 (I_in[0]),
    .data_out_1(I_out[0]),

    .index_2(I_index[1]),
    .data_in_2(I_in[1]),
    .data_out_2(I_out[1])
);

global_buffer_dp #(
.ADDR_BITS(13), // ADDR_BITS
.DATA_BITS(DATA_WIDTH)  // DATA_WIDTH
)
index_2 (
    .clk_i   (clk_i),

    .wr_en_1   (I_wr_en[2]),
    .index_1   (I_index[2]),
    .data_in_1 (I_in[2]),
    .data_out_1(I_out[2]),

    .index_2(I_index[3]),
    .data_in_2(I_in[3]),
    .data_out_2(I_out[3])
);

/*
 * I_OUT_OFFSET BUFFER
 * I_OUT_OFFSET
 * I_out_offset_index
 */
generate
for (genvar i = 0; i < 4; i++) begin
    global_buffer #(
    .ADDR_BITS(10),
    .DATA_BITS(DATA_WIDTH)
    )
    P_gbuff (
        .clk_i   (clk_i),
        .rst_i   (rst_i),
        .wr_en   ( (tpu_cmd_valid && tpu_cmd == (34 + i)) ),
        .index   ((tpu_cmd_valid && tpu_cmd == (34 + i)) ? tpu_param_2_in : I_out_offset_index[i]),
        .data_in (tpu_param_1_in),
        .data_out(I_out_offset_out[i])
    );
end
endgenerate

always_ff @( posedge clk_i ) begin
    for(int i = 0; i < 4; i++) begin
        I_out_offset_out_r[i] <= I_out_offset_out[i];
    end
end

always_ff @( posedge clk_i ) begin
    for (int i = 0; i < 4; i++) begin
        if(rst_i) begin
            I_out_offset_index[i] <= 0;
        end
        else if(curr_state == IDLE_S) begin
            I_out_offset_index[i] <= 0;
        end
        else if(next_state == NEXT_ROW_S) begin
            I_out_offset_index[i] <= I_out_offset_index[i] + 1;
        end
        else if(next_state == NEXT_COL_S) begin
            I_out_offset_index[i] <= 0;
        end

    end
end


//#########################
//#   PARTIAL SUM BUFFER  #
//#########################
generate
for (genvar i = 0; i < 4; i++) begin
    global_buffer #(
    .ADDR_BITS(10),
    .DATA_BITS(DATA_WIDTH*4)
    )
    P_gbuff (
        .clk_i   (clk_i),
        .rst_i   (rst_i),
        .wr_en   ((mode) ? fma_out_valid_r[i] : P_wr_en[i]),
        .index   ((mode) ? sram_w_idx         : P_index[i]),
        .data_in ((mode) ? fma_out_r          : P_data_in[i]),
        .data_out(P_data_out[i])
    );
end
endgenerate

/*
 * BatchNorm and skip add ctrl signal
 */

always_ff @( posedge clk_i ) begin
    if(tpu_cmd_valid && tpu_cmd == SET_RELU) begin
        relu_en <= tpu_param_1_in;
    end
end
always_ff @( posedge clk_i ) begin
    if(tpu_cmd_valid && tpu_cmd == SET_MODE) begin
        calc_len <= tpu_param_2_in;
    end
end

always_ff @( posedge clk_i ) begin
    if(tpu_cmd_valid && tpu_cmd == SET_MUL_VAL) begin
        mul_val_r <= tpu_param_2_in;
    end
    if(tpu_cmd_valid && tpu_cmd == SET_ADD_VAL) begin
        add_val_r <= tpu_param_2_in;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        send_cnt <= 0;
    end
    else if(curr_state == IDLE_S) begin
        send_cnt <= 0;
    end
    else if(curr_state == FMA_3_S) begin
        send_cnt <= send_cnt + 4;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        recv_cnt <= 0;
    end
    else if(curr_state == IDLE_S) begin
        recv_cnt <= 0;
    end
    else if(fma_out_valid_r[0]) begin
        recv_cnt <= recv_cnt + 4;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        sram_r_idx[0] <= 0;
        sram_r_idx[1] <= 1;
        sram_r_idx[2] <= 2;
        sram_r_idx[3] <= 3;
    end
    else if(curr_state == IDLE_S) begin
        sram_r_idx[0] <= 0;
        sram_r_idx[1] <= 1;
        sram_r_idx[2] <= 2;
        sram_r_idx[3] <= 3;
    end
    else if(curr_state == FMA_1_S || 
            curr_state == FMA_2_S || 
            curr_state == FMA_3_S) begin
        sram_r_idx[0] <= sram_r_idx[0] + 4;
        sram_r_idx[1] <= sram_r_idx[1] + 4;
        sram_r_idx[2] <= sram_r_idx[2] + 4;
        sram_r_idx[3] <= sram_r_idx[3] + 4;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        sram_w_idx <= 0;
    end
    else if(curr_state == IDLE_S) begin
        sram_w_idx <= 0;
    end
    else if(fma_out_valid_r[0] == 1) begin
        sram_w_idx <= sram_w_idx + 1;
    end
end

generate
    always_ff @( posedge clk_i ) begin
        for (int i = 0; i < 4; i++) begin
            fma_out_valid_r[i] <= fma_out_valid[i];
        end
    end
endgenerate

// should be nonblocking
generate
    always_comb begin
        for (int i = 0; i < 4; i++) begin
            if(fma_out[i][15]) begin
                fma_out_relu[i] = 0;
            end
            else begin
                fma_out_relu[i] = fma_out[i];
            end
        end
    end
endgenerate

generate
    always_ff @( posedge clk_i ) begin
        if(fma_out_valid[0] && !relu_en) begin
            fma_out_r <= {fma_out[0], fma_out[1], 
                          fma_out[2], fma_out[3]};
        end
        else if(fma_out_valid[0] && relu_en) begin
            fma_out_r <= {fma_out_relu[0], fma_out_relu[1], 
                          fma_out_relu[2], fma_out_relu[3]};
        end
    end
endgenerate
/*
 * BatchNorm: a * b + c
 * skip add : a * 1 + c
 */
always_comb begin
    for (int i = 0; i < 4; i++) begin
        fma_a_data[i] = gbuff_data_out_reg[i];
        fma_b_data[i] = (mode == 1) ? mul_val_r : 16'h3c00;
        fma_c_data[i] = (mode == 1) ? add_val_r : weight_out_reg[i];

        fma_a_valid[i] = (curr_state == FMA_3_S/**/);
        fma_b_valid[i] = (curr_state == FMA_3_S/**/);
        fma_c_valid[i] = (curr_state == FMA_3_S/**/);
    end     
end

/*
 * BatchNorm and skip add ip
 */

generate
for (genvar i = 0; i < 4; i++) begin
    floating_point_0 FP(

        .aclk(clk_i),

        .s_axis_a_tdata(fma_a_data[i]),
        .s_axis_a_tvalid(fma_a_valid[i]),

        .s_axis_b_tdata(fma_b_data[i]),
        .s_axis_b_tvalid(fma_b_valid[i]),

        .s_axis_c_tdata(fma_c_data[i]),
        .s_axis_c_tvalid(fma_c_valid[i]),

        .m_axis_result_tdata(fma_out[i]),
        .m_axis_result_tvalid(fma_out_valid[i])
    );
end
endgenerate


/*
 * avg pooling 
 */
always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        pooling_index <= 0;
    end
    else if(curr_state == IDLE_S) begin
        pooling_index <= 0;
    end
    else if(curr_state == AVG_POOLING_ACC_S) begin
        pooling_index <= pooling_index + 1;
    end
end

assign ret_avg_pooling = pooling_result;

/*
 * acc ip 
 */
assign acc_data_in = pooling_data_r[pooling_index];
assign acc_data_in_valid = (curr_state == AVG_POOLING_ACC_S);
assign acc_data_in_last = (pooling_index == 48);

always_ff @( posedge clk_i ) begin
    if(rst_i)                     acc_recv_cnt <= 0;
    else if(curr_state == IDLE_S) acc_recv_cnt <= 0;
    else if(acc_data_out_valid)   acc_recv_cnt <= acc_recv_cnt + 1;
end

always_ff @( posedge clk_i ) begin
    if(acc_data_out_valid)  pooling_result <= acc_data_out;
    else if(div_data_valid) pooling_result <= div_data_out;
end

floating_point_acc ACC2(

    .aclk(clk_i),

    .s_axis_a_tdata(acc_data_in),
    .s_axis_a_tlast(acc_data_in_last),
    .s_axis_a_tvalid(acc_data_in_valid),

    .m_axis_result_tdata(acc_data_out),
    .m_axis_result_tlast(acc_data_out_last),
    .m_axis_result_tvalid(acc_data_out_valid)
);
/*
 * div ip
 */
floating_point_div div2(
        .aclk(clk_i),
        .s_axis_a_tdata(pooling_result),
        .s_axis_a_tvalid((curr_state == AVG_POOLING_DIV_S)),
        .s_axis_b_tdata(16'h5220), // 49
        .s_axis_b_tvalid(1),
        .m_axis_result_tdata(div_data_out),
        .m_axis_result_tvalid(div_data_valid)
    );

/*
 * BATCHNORM HARDWARE
 * FOLLOW BY GeMM
 */
// a
always_comb begin
    bn_fma_a_data[ 0] = bn_mul_sram_out_r[0];
    bn_fma_a_data[ 4] = bn_mul_sram_out_r[0];
    bn_fma_a_data[ 8] = bn_mul_sram_out_r[0];
    bn_fma_a_data[12] = bn_mul_sram_out_r[0];

    bn_fma_a_data[ 1] = bn_mul_sram_out_r[1];
    bn_fma_a_data[ 5] = bn_mul_sram_out_r[1];
    bn_fma_a_data[ 9] = bn_mul_sram_out_r[1];
    bn_fma_a_data[13] = bn_mul_sram_out_r[1];

    bn_fma_a_data[ 2] = bn_mul_sram_out_r[2];
    bn_fma_a_data[ 6] = bn_mul_sram_out_r[2];
    bn_fma_a_data[10] = bn_mul_sram_out_r[2];
    bn_fma_a_data[14] = bn_mul_sram_out_r[2];

    bn_fma_a_data[ 3] = bn_mul_sram_out_r[3];
    bn_fma_a_data[ 7] = bn_mul_sram_out_r[3];
    bn_fma_a_data[11] = bn_mul_sram_out_r[3];
    bn_fma_a_data[15] = bn_mul_sram_out_r[3];
end
// b
always_comb begin
    {bn_fma_b_data[ 0], bn_fma_b_data[ 4], 
     bn_fma_b_data[ 8], bn_fma_b_data[12]} = rdata_out[0];

    {bn_fma_b_data[ 1], bn_fma_b_data[ 5], 
     bn_fma_b_data[ 9], bn_fma_b_data[13]} = rdata_out[1];

    {bn_fma_b_data[ 2], bn_fma_b_data[ 6], 
     bn_fma_b_data[10], bn_fma_b_data[14]} = rdata_out[2];

    {bn_fma_b_data[ 3], bn_fma_b_data[ 7], 
     bn_fma_b_data[11], bn_fma_b_data[15]} = rdata_out[3];
end
// c
always_comb begin
   bn_fma_c_data[ 0] = bn_add_sram_out_r[0];
   bn_fma_c_data[ 4] = bn_add_sram_out_r[0];
   bn_fma_c_data[ 8] = bn_add_sram_out_r[0];
   bn_fma_c_data[12] = bn_add_sram_out_r[0];

   bn_fma_c_data[ 1] = bn_add_sram_out_r[1];
   bn_fma_c_data[ 5] = bn_add_sram_out_r[1];
   bn_fma_c_data[ 9] = bn_add_sram_out_r[1];
   bn_fma_c_data[13] = bn_add_sram_out_r[1];

   bn_fma_c_data[ 2] = bn_add_sram_out_r[2];
   bn_fma_c_data[ 6] = bn_add_sram_out_r[2];
   bn_fma_c_data[10] = bn_add_sram_out_r[2];
   bn_fma_c_data[14] = bn_add_sram_out_r[2];

   bn_fma_c_data[ 3] = bn_add_sram_out_r[3];
   bn_fma_c_data[ 7] = bn_add_sram_out_r[3];
   bn_fma_c_data[11] = bn_add_sram_out_r[3];
   bn_fma_c_data[15] = bn_add_sram_out_r[3];
end

generate
    always_comb begin 
        for(int i = 0; i < 16; i++) begin
            bn_fma_a_valid[i] = (curr_state == START_BATCHNORM_S);
            bn_fma_b_valid[i] = (curr_state == START_BATCHNORM_S);
            bn_fma_c_valid[i] = (curr_state == START_BATCHNORM_S);
        end
    end
endgenerate

/*
 * 4x4 FMA
 */
generate
for (genvar i = 0; i < 16; i++) begin
    floating_point_0 FP(

        .aclk(clk_i),

        .s_axis_a_tdata(bn_fma_a_data[i]),
        .s_axis_a_tvalid(bn_fma_a_valid[i]),

        .s_axis_b_tdata(bn_fma_b_data[i]),
        .s_axis_b_tvalid(bn_fma_b_valid[i]),

        .s_axis_c_tdata(bn_fma_c_data[i]),
        .s_axis_c_tvalid(bn_fma_c_valid[i]),

        .m_axis_result_tdata(bn_fma_out[i]),
        .m_axis_result_tvalid(bn_fma_out_valid[i])
    );
end
endgenerate

/*
 * relu
 */
generate
    always_comb begin
        for (int i = 0; i < 16; i++) begin
            if(bn_fma_out[i][15]) begin
                bn_fma_out_relu[i] = 0;
            end
            else begin
                bn_fma_out_relu[i] = bn_fma_out[i];
            end
        end
    end
endgenerate

generate
    always_ff @( posedge clk_i ) begin
        if(bn_fma_out_valid[0] && !relu_en) begin
            bn_fma_out_r[0] <= {bn_fma_out[ 0], bn_fma_out[ 4], 
                                bn_fma_out[ 8], bn_fma_out[12]};
            bn_fma_out_r[1] <= {bn_fma_out[ 1], bn_fma_out[ 5], 
                                bn_fma_out[ 9], bn_fma_out[13]};
            bn_fma_out_r[2] <= {bn_fma_out[ 2], bn_fma_out[ 6], 
                                bn_fma_out[10], bn_fma_out[14]};
            bn_fma_out_r[3] <= {bn_fma_out[ 3], bn_fma_out[ 7], 
                                bn_fma_out[11], bn_fma_out[15]};
        end
        else if(bn_fma_out_valid[0] && relu_en) begin
            bn_fma_out_r[0] <= {bn_fma_out_relu[ 0], bn_fma_out_relu[ 4], 
                                bn_fma_out_relu[ 8], bn_fma_out_relu[12]};
            bn_fma_out_r[1] <= {bn_fma_out_relu[ 1], bn_fma_out_relu[ 5], 
                                bn_fma_out_relu[ 9], bn_fma_out_relu[13]};
            bn_fma_out_r[2] <= {bn_fma_out_relu[ 2], bn_fma_out_relu[ 6], 
                                bn_fma_out_relu[10], bn_fma_out_relu[14]};
            bn_fma_out_r[3] <= {bn_fma_out_relu[ 3], bn_fma_out_relu[ 7], 
                                bn_fma_out_relu[11], bn_fma_out_relu[15]};
        end
    end
endgenerate

/*
 * BatchNorm MUL ADD SRAM
 */
always_comb begin
    bn_mul_wren[0] = (tpu_cmd_valid) && (tpu_cmd == SET_BN_MUL_SRAM_0);
    bn_mul_wren[1] = (tpu_cmd_valid) && (tpu_cmd == SET_BN_MUL_SRAM_1);
    bn_mul_wren[2] = (tpu_cmd_valid) && (tpu_cmd == SET_BN_MUL_SRAM_2);
    bn_mul_wren[3] = (tpu_cmd_valid) && (tpu_cmd == SET_BN_MUL_SRAM_3);

    bn_add_wren[0] = (tpu_cmd_valid) && (tpu_cmd == SET_BN_ADD_SRAM_0);
    bn_add_wren[1] = (tpu_cmd_valid) && (tpu_cmd == SET_BN_ADD_SRAM_1);
    bn_add_wren[2] = (tpu_cmd_valid) && (tpu_cmd == SET_BN_ADD_SRAM_2);
    bn_add_wren[3] = (tpu_cmd_valid) && (tpu_cmd == SET_BN_ADD_SRAM_3);
end

generate
    always_ff @( posedge clk_i ) begin
        for(int i = 0; i < 4; i++) begin
            if(curr_state == IDLE_S) begin
                bn_sram_idx[i] <= 0;
            end
            else if(curr_state == NEXT_COL_S) begin
                bn_sram_idx[i] <= bn_sram_idx[i] + 1;
            end
        end
    end
endgenerate

generate
for (genvar i = 0; i < 4; i++) begin
    global_buffer #(
    .ADDR_BITS(8),
    .DATA_BITS(DATA_WIDTH)
    )
    BN_mul_gbuff (
        .clk_i   (clk_i),
        .rst_i   (rst_i),
        .wr_en   (bn_mul_wren[i]),
        .index   ((bn_mul_wren[i]) ? tpu_param_2_in : bn_sram_idx[i]),
        .data_in (tpu_param_1_in),
        .data_out(bn_mul_sram_out[i])
    );
end
endgenerate

generate
for (genvar i = 0; i < 4; i++) begin
    global_buffer #(
    .ADDR_BITS(8),
    .DATA_BITS(DATA_WIDTH)
    )
    BN_add_gbuff (
        .clk_i   (clk_i),
        .rst_i   (rst_i),
        .wr_en   (bn_add_wren[i]),
        .index   ((bn_add_wren[i]) ? tpu_param_2_in : bn_sram_idx[i]),
        .data_in (tpu_param_1_in),
        .data_out(bn_add_sram_out[i])
    );
end
endgenerate

always_ff @( posedge clk_i ) begin
    for(int i = 0; i < 4; i++) begin
        bn_mul_sram_out_r[i] <= bn_mul_sram_out[i];
        bn_add_sram_out_r[i] <= bn_add_sram_out[i];
    end
end

endmodule