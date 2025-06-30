`timescale 1ns / 1ps
// =============================================================================
//  Program : TPU.sv
//  Author  : 
//  Date    : 
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// =============================================================================

module TPU
#(parameter ACLEN=8,
  parameter ADDR_BITS=16,
  parameter DATA_WIDTH = 32
)
(
    input   logic                        clk_i, rst_i,
    // cmd
    input   logic                        tpu_cmd_valid,     // tpu valid
    input   logic   [ACLEN-1 : 0]        tpu_cmd,           // tpu
    input   logic   [DATA_WIDTH-1 : 0]   tpu_param_1_in,    // data 1
    input   logic   [DATA_WIDTH-1 : 0]   tpu_param_2_in,    // data 2

    // output data 64 bits x 4
    output  logic   [DATA_WIDTH*4-1 : 0]   tpu_data_1_out,
    output  logic   [DATA_WIDTH*4-1 : 0]   tpu_data_2_out,
    output  logic   [DATA_WIDTH*4-1 : 0]   tpu_data_3_out,
    output  logic   [DATA_WIDTH*4-1 : 0]   tpu_data_4_out,
    //
    output  logic                      ret_valid,
    output  logic   [DATA_WIDTH-1 : 0] ret_data_out,
    //
    output  logic   [DATA_WIDTH-1 : 0] ret_avg_pooling,
    output  logic   [DATA_WIDTH-1 : 0] ret_softmax_result,
    // first dual port sram control signal
    output  logic                     gbuff_wr_en_0,
    output  logic  [ADDR_BITS-1  : 0] gbuff_index_0,
    output  logic  [DATA_WIDTH-1 : 0] gbuff_data_in_0,
    input   logic  [DATA_WIDTH-1 : 0] gbuff_data_out_0,

    output  logic  [ADDR_BITS-1  : 0] gbuff_index_1,
    output  logic  [DATA_WIDTH-1 : 0] gbuff_data_in_1,
    input   logic  [DATA_WIDTH-1 : 0] gbuff_data_out_1,

    // second dual port sram control signal
    output  logic                     gbuff_wr_en_2,
    output  logic  [ADDR_BITS-1  : 0] gbuff_index_2,
    output  logic  [DATA_WIDTH-1 : 0] gbuff_data_in_2,
    input   logic  [DATA_WIDTH-1 : 0] gbuff_data_out_2,

    output  logic  [ADDR_BITS-1  : 0] gbuff_index_3,
    output  logic  [DATA_WIDTH-1 : 0] gbuff_data_in_3,
    input   logic  [DATA_WIDTH-1 : 0] gbuff_data_out_3,

    output  logic                        tpu_busy     // 0->idle, 1->busy
);
`include "tpu_cmd.svh"

function logic is_tpu_cmd_valid_and_match(input logic [ACLEN-1 : 0] cmd_type);
    return (tpu_cmd_valid && tpu_cmd == cmd_type);
endfunction


typedef enum {IDLE_S, 
              LOAD_IDX_S,
              READ_DATA_1_S, READ_DATA_2_S, READ_DATA_3_S, 
              READ_DATA_4_S, READ_DATA_5_S,
              TRIGGER_S, TRIGGER_LAST_S,
              FORWARD_S,
              WAIT_IDLE_S,
              START_BATCHNORM_S,
              WAIT_BATCHNORM_S,
              STORE_S, 
              NEXT_ROW_S,
              NEXT_COL_S,
              PRELOAD_DATA_S,
              FMA_1_S,
              FMA_2_S,
              FMA_3_S,
              FMA_WAIT_IDLE_S,
              AVG_POOLING_ACC_S,
            //   WAIT_AVG_POOLING_ACC_S,
            //   AVG_POOLING_DIV_S,
              WAIT_AVG_POOLING_DIV_S,
            //   WAIT_MAX_POOLING_S,
              SW_READ_DATA_S,
              WAIT_SF_ACC_S,
              WAIT_SF_S,
              OUTPUT_1_S,
              OUTPUT_2_S,
              OUTPUT_3_S} state_t;

state_t curr_state, next_state;

typedef enum {SW_READ,   SW_WRITE,
              TPU_READ,  TPU_WRITE,
              DRAM_READ, DRAM_WRITE} gbuff_state_t;

gbuff_state_t gbuff_status  [0 : 3];
gbuff_state_t weight_status [0 : 3];
gbuff_state_t I_status      [0 : 3];
gbuff_state_t P_status      [0 : 3];

logic                        tpu_cmd_valid_reg;
logic   [ACLEN : 0]          tpu_cmd_reg;
logic   [DATA_WIDTH-1 : 0]   param_1_in_reg; 
logic   [DATA_WIDTH-1 : 0]   param_2_in_reg;

logic   [DATA_WIDTH*4-1 : 0] rdata_out [0 : 3];

logic                        mmu_busy;     // 0->idle, 1->busy

//#########################
//#    INDEX START REG    #
//#########################
/*
 * row_idx: base address
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

// matrix for (M * K) (K * N)
logic [ADDR_BITS-1 : 0] K_reg, M_reg, N_reg;

logic                        mmu_cmd_valid;
logic [ACLEN : 0]            mmu_cmd;
logic [DATA_WIDTH*4-1 : 0]   mmu_param_in [0 : 3];

// input sram signal
logic                        gbuff_wr_en        [0 : 3];
logic   [ADDR_BITS-1  : 0]   gbuff_index        [0 : 3];
logic   [DATA_WIDTH-1 : 0]   gbuff_data_in      [0 : 3];
logic   [DATA_WIDTH-1 : 0]   gbuff_data_out     [0 : 3];
logic   [DATA_WIDTH-1 : 0]   gbuff_data_out_reg [0 : 3];

// weight sram signal
logic                        weight_wr_en   [0 : 3];
logic   [ADDR_BITS-1  : 0]   weight_index   [0 : 3];
logic   [DATA_WIDTH-1 : 0]   weight_in      [0 : 3];
logic   [DATA_WIDTH-1 : 0]   weight_out     [0 : 3];
logic   [DATA_WIDTH-1 : 0]   weight_out_reg [0 : 3];

// index sram signal
logic                        I_wr_en   [0 : 3];
logic   [ADDR_BITS-1  : 0]   I_index   [0 : 3];
logic   [DATA_WIDTH-1 : 0]   I_in      [0 : 3];
logic   [DATA_WIDTH-1 : 0]   I_out     [0 : 3];
logic   [DATA_WIDTH-1 : 0]   I_out_reg [0 : 3];

logic   [ADDR_BITS-1 : 0]   I_out_offset_index [0 : 3];
logic   [ADDR_BITS-1 : 0]   I_out_offset_out   [0 : 3];
logic   [ADDR_BITS-1 : 0]   I_out_offset_out_r [0 : 3];

// result sram signal
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

/*
 * mode: 0 -> conv
 * mode: 1 -> BatchNorm
 * mode: 2 -> skip add
 */
logic [1 : 0] mode;
logic   [DATA_WIDTH*4-1 : 0]  bn_fma_out_r      [0 : 3];
logic batchNormResultValid;

logic   [DATA_WIDTH*4-1 : 0]  fma_out_r;

logic   [ADDR_BITS-1    : 0]  sram_r_idx [0 : 3];
logic   [ADDR_BITS-1    : 0]  sram_w_idx;

logic                         relu_en;

logic   [ADDR_BITS-1    : 0]  send_cnt;
logic   [ADDR_BITS-1    : 0]  recv_cnt;
logic   [ADDR_BITS-1    : 0]  calc_len;

logic skipConnectionResultValid;

/*
 * AVERAGE POOLING SIGNAL (NEW)
 * 9 COMPARATER FOR MAX POOLING
 * 1 ACCUMULATOR FOR AVERAGE POOLING
 */
logic   [DATA_WIDTH-1   : 0] pooling_data_r [0 : 51];
logic   [ADDR_BITS-1    : 0] pooling_index;
logic   [DATA_WIDTH-1   : 0] pooling_result;
// logic   [DATA_WIDTH-1   : 0] acc_data_in , div_data_in;
// logic   [DATA_WIDTH-1   : 0] acc_data_out, div_data_out;
logic                        acc_data_in_valid, div_data_valid;
// logic                        acc_data_out_valid;
// logic                        acc_data_in_last, acc_data_out_last;
// logic   [DATA_WIDTH-1   : 0] avg_pooling_data;

// logic   [ADDR_BITS-1    : 0] acc_recv_cnt;

logic enable_avg_pooling;

// softmax signal
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

/*
 * pooling hardware signal
 */
logic                          max_pooling_valid;
logic   [DATA_WIDTH*4-1 : 0]   max_pooling_out [0 : 3];
logic                          pooling_busy;
logic   [ADDR_BITS-1 : 0]      pooling_index_r;


assign conv_end      = (row_acc + 4 >= M_reg && col_acc + 4 >= N_reg);
assign conv_next_row = (row_acc + 4 < M_reg);
assign conv_next_col = (col_acc + 4 < N_reg);

always_ff @( posedge clk_i ) begin
    if     ( is_tpu_cmd_valid_and_match(SET_CONV_MODE)    ) begin
        mode <= 0;
    end
    else if( is_tpu_cmd_valid_and_match(SET_FIX_MAC_MODE) ) begin
        mode <= 1;
    end
    else if( is_tpu_cmd_valid_and_match(SET_MODE)         ) begin
        mode <= tpu_param_1_in;
    end
end

always_ff @( posedge clk_i ) begin
    if( is_tpu_cmd_valid_and_match(SET_PRELOAD) ) begin
        preload <= tpu_param_1_in;
    end
    
end

// avg pooling
always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        enable_avg_pooling <= 0;
    end
    else if( is_tpu_cmd_valid_and_match(RESET) ) begin
        enable_avg_pooling <= 0;
    end
    else if( is_tpu_cmd_valid_and_match(SET_AVERAGE_POOLING) ) begin
        enable_avg_pooling <= 1;
    end
end

always_ff @(posedge clk_i) begin
    tpu_cmd_valid_reg <= tpu_cmd_valid;
    if(tpu_cmd_valid) begin
        tpu_cmd_reg       <= tpu_cmd;  
        param_1_in_reg    <= tpu_param_1_in;
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

// next state logic
always_comb begin
    case (curr_state)
    IDLE_S: if     ( is_tpu_cmd_valid_and_match(TRIGGER_CONV   ) ) 
                next_state = LOAD_IDX_S;
            else if( is_tpu_cmd_valid_and_match(SW_READ_DATA   ) ) 
                next_state = SW_READ_DATA_S;
            else if( is_tpu_cmd_valid_and_match(SET_SOFTMAX    ) ) 
                next_state = WAIT_SF_ACC_S;
            else if( is_tpu_cmd_valid_and_match(TRIGGER_SOFTMAX) ) 
                next_state = WAIT_SF_S;
            else if( is_tpu_cmd_valid_and_match(TRIGGER_ADD    ) ) 
                next_state = FMA_1_S;
            else         
                next_state = IDLE_S;
    LOAD_IDX_S    : next_state = PRELOAD_DATA_S;
    PRELOAD_DATA_S: next_state = READ_DATA_1_S;
    READ_DATA_1_S : next_state = READ_DATA_2_S;
    READ_DATA_2_S : next_state = READ_DATA_3_S;
    READ_DATA_3_S : next_state = READ_DATA_4_S;
    READ_DATA_4_S : next_state = READ_DATA_5_S;
    READ_DATA_5_S : next_state = TRIGGER_S;
    TRIGGER_S  : if(K_cnt == K_reg - 2) next_state = TRIGGER_LAST_S;
                 else                   next_state = TRIGGER_S;
    TRIGGER_LAST_S : next_state = FORWARD_S;
    FORWARD_S: if(sa_forward_cnt == 6)  next_state = WAIT_IDLE_S;
               else                     next_state = FORWARD_S;
    WAIT_IDLE_S: if   (mmu_busy) next_state = WAIT_IDLE_S;
                 else            next_state = START_BATCHNORM_S;
    START_BATCHNORM_S: next_state = WAIT_BATCHNORM_S;
    WAIT_BATCHNORM_S : if(batchNormResultValid) next_state = STORE_S;
                       else next_state = WAIT_BATCHNORM_S;
    STORE_S: if(conv_end)           next_state = IDLE_S;
             else if(conv_next_row) next_state = NEXT_ROW_S;
             else                   next_state = NEXT_COL_S;
    NEXT_ROW_S: next_state = PRELOAD_DATA_S;
    NEXT_COL_S: next_state = PRELOAD_DATA_S;
    FMA_1_S: next_state = FMA_2_S;
    FMA_2_S: next_state = FMA_3_S;
    FMA_3_S: if(send_cnt + 4 >= calc_len) 
                next_state = FMA_WAIT_IDLE_S;
             else
                next_state = FMA_3_S;
    FMA_WAIT_IDLE_S: if(recv_cnt + 4 >= calc_len && !enable_avg_pooling)
                        next_state = IDLE_S;
                     else if(recv_cnt + 4 >= calc_len && enable_avg_pooling)
                        next_state = AVG_POOLING_ACC_S;
                     else
                        next_state = FMA_WAIT_IDLE_S;
    AVG_POOLING_ACC_S:  if(pooling_index == 48)
                            next_state = WAIT_AVG_POOLING_DIV_S;
                        else 
                            next_state = AVG_POOLING_ACC_S;
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

assign tpu_busy = (curr_state != IDLE_S) || pooling_busy;

always_ff @( posedge clk_i ) begin
    if(curr_state == IDLE_S || curr_state == NEXT_ROW_S || curr_state == NEXT_COL_S) begin
        K_cnt <= 0;
    end
    else if( is_tpu_cmd_valid_and_match(RESET) ) begin
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

// set KMN value
always_ff @(posedge clk_i) begin
    if( is_tpu_cmd_valid_and_match(SET_KMN) ) begin
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


// read result
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

// send input to GeMM
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
    if( is_tpu_cmd_valid_and_match(SET_ROW_IDX) )begin
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
    if( is_tpu_cmd_valid_and_match(SET_COL_IDX) )begin
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
        if(curr_state == STORE_S) begin
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
        else if(curr_state == STORE_S) begin
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
//#      P DATA IN        #
//#########################
always_ff @(posedge clk_i) begin
    for (int i = 0; i < 4; i++) begin
        P_data_in[i] <= (P_status[i] == SW_READ   ) ? 0
                      : (P_status[i] == SW_WRITE  ) ? 0 //tpu_data[i] 
                      : (P_status[i] == TPU_READ  ) ? 0 
                      : (P_status[i] == TPU_WRITE && mode == 0 ) ? bn_fma_out_r[i]
                    //   : (P_status[i] == TPU_WRITE && mode == 1 ) ? bn_data_out[i]
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
    //    (tpu_cmd_valid_reg && tpu_cmd_reg == SET_MUL_VAL     ) || 
    //    (tpu_cmd_valid_reg && tpu_cmd_reg == SET_ADD_VAL     ) ||
       (tpu_cmd_valid_reg && tpu_cmd_reg == SET_CONV_MODE   ) || 
    //    (tpu_cmd_valid_reg && tpu_cmd_reg == SET_FIX_MAC_MODE) ||
       (curr_state == TRIGGER_S                             ) ||
       (curr_state == TRIGGER_LAST_S                        ) ||
       (curr_state == FORWARD_S                             ) ||
       (curr_state == READ_DATA_2_S && preload)               
        ) begin
        mmu_cmd_valid = 1; 
    end
    else begin
        mmu_cmd_valid = 0;
    end
end
//#########################
//#       MMU CMD         #
//#########################

always_comb begin
    if(curr_state == FORWARD_S) begin
        mmu_cmd         = FORWARD;
    end
    else if(curr_state == READ_DATA_2_S) begin
        mmu_cmd = SET_PE_VAL;
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
    .ACLEN(ACLEN),
    .DATA_WIDTH(DATA_WIDTH)
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

    .mmu_busy(mmu_busy)
);

//#########################
//       SOFTMAX
//#########################
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
        .s_axis_a_tvalid( is_tpu_cmd_valid_and_match(SET_SOFTMAX) ),
        .m_axis_result_tdata(exp_1_out),
        .m_axis_result_tvalid(exp_1_out_valid)
);
// exp 2
floating_point_exp exp_2(
        .aclk(clk_i),
        .s_axis_a_tdata(tpu_param_1_in),
        .s_axis_a_tvalid( is_tpu_cmd_valid_and_match(TRIGGER_SOFTMAX) ),
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

// store skip connection output
always_ff @( posedge clk_i ) begin
    if(skipConnectionResultValid) begin
        pooling_data_r[sram_w_idx*4  ] <= fma_out_r[(DATA_WIDTH*4-1)-:DATA_WIDTH];
        pooling_data_r[sram_w_idx*4+1] <= fma_out_r[(DATA_WIDTH*3-1)-:DATA_WIDTH];
        pooling_data_r[sram_w_idx*4+2] <= fma_out_r[(DATA_WIDTH*2-1)-:DATA_WIDTH];
        pooling_data_r[sram_w_idx*4+3] <= fma_out_r[(DATA_WIDTH*1-1)-:DATA_WIDTH];
    end
end

//#########################
//#      LINE BUFFER      #
//#########################

// first dual port sram control signal
always_comb begin
    // port 1
    gbuff_wr_en_0     = ((mode) ? 0             : gbuff_wr_en[0]);
    gbuff_index_0     = ((mode) ? sram_r_idx[0] : gbuff_index[0]);
    gbuff_data_in_0   = (gbuff_data_in[0]);
    gbuff_data_out[0] = gbuff_data_out_0;  
    // port 2
    gbuff_index_1     = ((mode) ? sram_r_idx[1] : gbuff_index[1]);
    gbuff_data_out[1] = gbuff_data_out_1;
end

// second dual port sram control signal
always_comb begin
    // port 1
    gbuff_wr_en_2     = ((mode) ? 0             : gbuff_wr_en[2]);
    gbuff_index_2     = ((mode) ? sram_r_idx[2] : gbuff_index[2]);
    gbuff_data_in_2   = (gbuff_data_in[2]);
    gbuff_data_out[2] = gbuff_data_out_2;  
    // port 2
    gbuff_index_3     = ((mode) ? sram_r_idx[3] : gbuff_index[3]);
    gbuff_data_out[3] = gbuff_data_out_3;
end

//#########################
//#     WEIGHT BUFFER     #
//#########################
global_buffer_dp #(
.ADDR_BITS(14),
.DATA_BITS(DATA_WIDTH)
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
.ADDR_BITS(14),
.DATA_BITS(DATA_WIDTH)
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

global_buffer #(
.ADDR_BITS(13),
.DATA_BITS(DATA_WIDTH)
)
index_1 (
    .clk_i   (clk_i),

    .wr_en   (I_wr_en[0]),
    .index   (I_index[0]),

    .data_in (I_in[0]),
    .data_out(I_out[0])
);

assign I_out[1] = I_out[0];
assign I_out[2] = I_out[0];
assign I_out[3] = I_out[0];

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
        .wr_en   ( (mode == 3) ? max_pooling_valid  :
                   (mode)      ? skipConnectionResultValid : P_wr_en[i]),

        .index   ( (mode == 3) ? pooling_index_r : 
                   (mode)      ? sram_w_idx      : P_index[i]),
        
        .data_in ( (mode == 3) ? max_pooling_out[i] : (mode) 
                               ? fma_out_r          : P_data_in[i]),
        
        .data_out(P_data_out[i])
    );
end
endgenerate

/*
 * skip connection ctrl signal
 */
always_ff @( posedge clk_i ) begin
    if( is_tpu_cmd_valid_and_match(SET_RELU) ) begin
        relu_en <= tpu_param_1_in;
    end
end
always_ff @( posedge clk_i ) begin
    if( is_tpu_cmd_valid_and_match(SET_MODE) ) begin
        calc_len <= tpu_param_2_in;
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
    else if(skipConnectionResultValid) begin
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
    else if(skipConnectionResultValid) begin
        sram_w_idx <= sram_w_idx + 1;
    end
end

SKIPCONNECTION #(
    .ACLEN(ACLEN),
    .ADDR_BITS(ADDR_BITS),
    .DATA_WIDTH(DATA_WIDTH)
) sc1
(
    .clk_i(clk_i), .rst_i(rst_i),
    .relu_en(relu_en),
    //
    .skipConnectionInputValid((curr_state == FMA_3_S)),
    .skipConnectionInput_1(gbuff_data_out_reg),
    .skipConnectionInput_2(weight_out_reg),
    // 
    .skipConnectionResultValid(skipConnectionResultValid),
    .skipConnectionResult_r(fma_out_r)
);

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

AVGPOOLING #(
    .ACLEN(ACLEN),
    .ADDR_BITS(ADDR_BITS),
    .DATA_WIDTH(DATA_WIDTH)
) avg1
(
    .clk_i(clk_i), .rst_i(rst_i),
    //
    .avgPoolingInputValid((curr_state == AVG_POOLING_ACC_S)),
    .avgPoolingInputLast((pooling_index == 48)),
    .avgPoolingInput(pooling_data_r[pooling_index]),
    // batchNorm Result
    .avgPoolingResultValid(div_data_valid),
    .avgPoolingResult_r(pooling_result)
);

// batchNorm
BATCHNORM #(
    .ACLEN(ACLEN),
    .ADDR_BITS(ADDR_BITS),
    .DATA_WIDTH(DATA_WIDTH)
) bn1
(
    .clk_i(clk_i), .rst_i(rst_i),
    // cmd
    .tpu_cmd_valid(tpu_cmd_valid),
    .tpu_cmd(tpu_cmd),
    .tpu_param_1_in(tpu_param_1_in), 
    .tpu_param_2_in(tpu_param_2_in),
    .relu_en(relu_en),
    .batchNormSramIdxRst(curr_state == IDLE_S),
    .batchNormSramIdxInc(curr_state == NEXT_COL_S),
    // convolution result
    .convolutionResultValid((curr_state == START_BATCHNORM_S)),
    .rdata_out(rdata_out),
    // batchNorm Result
    .batchNormResultValid(batchNormResultValid),
    .batchNormResult_r(bn_fma_out_r)
);

/*
 * max pooling
 */
always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        pooling_index_r <= 0;
    end
    else if( is_tpu_cmd_valid_and_match(RESET_POOLING_IDX) ) begin
        pooling_index_r <= 0;
    end
    else if(max_pooling_valid) begin
        pooling_index_r <= pooling_index_r + 1;
    end
end

POOLING p1
(
    .clk_i(clk_i), .rst_i(rst_i),

    .reset_lans   ( is_tpu_cmd_valid_and_match(RESET_LANS   ) ),
    .set_lans_idx ( is_tpu_cmd_valid_and_match(SET_LANS_IDX ) ),
    .sram_next    ( is_tpu_cmd_valid_and_match(SRAM_NEXT    ) ),
    .pooling_start( is_tpu_cmd_valid_and_match(POOLING_START) ),

    .bn_valid(curr_state == STORE_S),
    .bn_out_1(bn_fma_out_r[0]),
    .bn_out_2(bn_fma_out_r[1]),
    .bn_out_3(bn_fma_out_r[2]),
    .bn_out_4(bn_fma_out_r[3]),

    .max_pooling_valid(max_pooling_valid),
    .max_pooling_out_1(max_pooling_out[0]),
    .max_pooling_out_2(max_pooling_out[1]),
    .max_pooling_out_3(max_pooling_out[2]),
    .max_pooling_out_4(max_pooling_out[3]),

    .busy(pooling_busy)
);


endmodule