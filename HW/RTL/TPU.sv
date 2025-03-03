`timescale 1ns / 1ps
// =============================================================================
//  Program : TPU.sv
//  Author  : 
//  Date    : 
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// =============================================================================
// `include "config.vh"
module TPU
#(parameter ACLEN  = 8,
  parameter DATA_WIDTH = 32,
  parameter ADDR_BITS=16,
  parameter DATA_BITS=32 
)
(
    /////////// System signals   ///////////////////////////////////////////////
    input   logic                        clk_i, rst_i,
    /////////// MMU command ///////////////////////////////////////////////
    input   logic                        tpu_cmd_valid,     // tpu valid
    input   logic   [ACLEN-1 : 0]        tpu_cmd,           // tpu
    input   logic   [DATA_WIDTH-1 : 0]   tpu_param_1_in,    // data 1
    input   logic   [DATA_WIDTH-1 : 0]   tpu_param_2_in     // data 2

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
    // output  logic   [DATA_WIDTH*4-1 : 0] rdata_1_out,
    // output  logic   [DATA_WIDTH*4-1 : 0] rdata_2_out,
    // output  logic   [DATA_WIDTH*4-1 : 0] rdata_3_out,
    // output  logic   [DATA_WIDTH*4-1 : 0] rdata_4_out, 

    // output  logic                        tpu_busy     // 0->idle, 1->busy
);

//************************
//    TPU cmd table
//************************
parameter  RESET             = 0;  // whole
parameter  TRIGGER_CONV      = 1;  // whole
parameter  SW_WRITE_RAM_A    = 2;  // tpu_cmd   : 2
                                   // param_1_in: index
                                   // param_2_in: data
parameter  SW_WRITE_RAM_B    = 3;  // tpu_cmd   : 3
                                   // param_1_in: index
                                   // param_2_in: data
parameter  SW_WRITE_RAM_C    = 4;  // tpu_cmd   : 4
                                   // param_1_in: index
                                   // param_2_in: data
parameter  SW_WRITE_RAM_W    = 5;  // tpu_cmd   : 5
                                   // param_1_in: index
                                   // param_2_in: data
parameter  SW_READ_RAM       = 6;  // tpu_cmd   : 6
                                   // param_1_in: ram sel (0 -> A, 1 -> B, 2 -> C, 3 -> W)
                                   // param_2_in: index
parameter  SET_MUL_VAL       = 7;  // tpu_cmd   : 7
                                   // param_1_in: column number
                                   // param_2_in: multipled value
parameter  SET_ADD_VAL       = 8;  // partial
                                   // param_1_in: column number
                                   // param_2_in: added value

parameter  SET_PE_VAL_A      = 9;  // partial
                                   // param_1_in: PE number
                                   // param_2_in: index
parameter  SET_PE_VAL_B      = 10; // partial
                                   // param_1_in: PE number
                                   // param_2_in: index
parameter  SET_CONV_MODE     = 11; // whole
parameter  SET_FIX_MAC_MODE  = 12; // whole
parameter  SET_ROW_IDX       = 13; // partial
                                   // param_1_in: idx (0~4)
                                   // param_2_in: value
parameter  SET_KMN           = 14; // partial
                                   // param_1_in: idx (0:K, 1:M, 2:N)
                                   // param_2_in: value
parameter  SET_ST_IDX        = 14; // partial
                                   // param_1_in: idx (0~4)
                                   // param_2_in: value
parameter  IDLE              = 14; // whole

typedef enum {IDLE_S, HW_RESET_S,
              READ_R1_S, READ_R2_S, READ_R3_S, READ_R4_S, 
              READ_I1_S, READ_I2_S, READ_I3_S,
              TRIGGER_S, WAIT_IDLE_S, 
              STORE_VAL_1_S, STORE_VAL_2_S, STORE_VAL_3_S, STORE_VAL_4_S,
              SET_MUL_VAL_S, 
              SET_ADD_VAL_S, 
              SET_PE_VAL_S, 
              SET_CONV_MODE_S, 
              SET_FIX_MAC_MODE_S} state_t;
state_t curr_state, next_state;


typedef enum {SW_READ,   SW_WRITE,
              TPU_READ,  TPU_WRITE,
              DRAM_READ, DRAM_WRITE} gbuff_state_t;
gbuff_state_t gbuff_A_status, gbuff_B_status,
              gbuff_C_status, gbuff_W_status;

logic                        tpu_cmd_valid_reg; // tpu valid
logic   [ACLEN : 0]          tpu_cmd_reg;       // tpu
logic   [DATA_WIDTH-1 : 0]   param_1_in_reg;    // data 1
logic   [DATA_WIDTH-1 : 0]   param_2_in_reg;    // data 2

logic   [DATA_WIDTH*4-1 : 0] rdata_1_out;
logic   [DATA_WIDTH*4-1 : 0] rdata_2_out;
logic   [DATA_WIDTH*4-1 : 0] rdata_3_out;
logic   [DATA_WIDTH*4-1 : 0] rdata_4_out; 

logic                        tpu_busy;     // 0->idle, 1->busy

logic gbuff_sel; // 0-> TPU READ GBUFF A
                 //     SW / DRAM WRITE GBUFF B
                 // 1-> TPU READ GBUFF B
                 //     SW / DRAM WRITE GBUFF A


//************************
//    INDEX START REG
//************************
logic [ADDR_BITS-1 : 0] row_idx   [0 : 3];
logic [ADDR_BITS-1 : 0] col_idx   [0 : 3];
logic [DATA_BITS-1 : 0] row_input [0 : 3];
logic [DATA_BITS-1 : 0] col_input [0 : 3];

logic [ADDR_BITS-1 : 0] store_idx [0 : 3];

logic [ADDR_BITS-1 : 0] I_index;

logic [ADDR_BITS-1 : 0] K, M, N; // (M * K) (K * N)

logic                        mmu_cmd_valid;  // cmd
logic   [ACLEN : 0]          mmu_cmd;        // cmd
logic   [DATA_WIDTH-1 : 0]   mmu_param_1_in; // data 1
logic   [DATA_WIDTH-1 : 0]   mmu_param_2_in; // data 2

logic   [DATA_BITS-1  : 0] gbuff_data, gbuff_weight;

logic                        A_wr_en, B_wr_en, W_wr_en;
logic   [ADDR_BITS-1  : 0]   A_index, B_index, W_index;
logic   [DATA_WIDTH-1 : 0]   A_data_in, B_data_in, W_data_in;
logic   [DATA_WIDTH-1 : 0]   A_data_out, B_data_out, W_data_out, I_data_out;

// gbuff c signal (128 bits)
logic                        C_wr_en;
logic   [ADDR_BITS-1    : 0] C_index;
logic   [DATA_WIDTH*4-1 : 0] C_data_in;
logic   [DATA_WIDTH*4-1 : 0] C_data_out;


logic   [ADDR_BITS-1  : 0]   K_cnt;

always_ff @(posedge clk_i) begin
    if(tpu_cmd_valid) begin
        tpu_cmd_valid_reg <= tpu_cmd_valid;
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
//************************
//   NEXT STATE LOGIC
//************************
always_comb begin
    case (curr_state)
    IDLE_S: if     (tpu_cmd_valid && tpu_cmd == RESET           ) next_state = HW_RESET_S;
            else if(tpu_cmd_valid && tpu_cmd == TRIGGER_CONV    ) next_state = READ_R1_S;
            else if(tpu_cmd_valid && tpu_cmd == SET_MUL_VAL     ) next_state = SET_MUL_VAL_S;
            else if(tpu_cmd_valid && tpu_cmd == SET_ADD_VAL     ) next_state = SET_ADD_VAL_S;
            else if(tpu_cmd_valid && tpu_cmd == SET_PE_VAL_A      ) next_state = SET_PE_VAL_S;
            else if(tpu_cmd_valid && tpu_cmd == SET_CONV_MODE   ) next_state = SET_CONV_MODE_S;
            else if(tpu_cmd_valid && tpu_cmd == SET_FIX_MAC_MODE) next_state = SET_FIX_MAC_MODE_S;
            else         next_state = IDLE_S;
    HW_RESET_S : next_state = IDLE_S;
    READ_R1_S  : next_state = READ_R2_S;
    READ_R2_S  : next_state = READ_R3_S;
    READ_R3_S  : next_state = READ_R4_S;
    READ_R4_S  : next_state = READ_I1_S;
    READ_I1_S  : next_state = READ_I2_S;
    READ_I2_S  : next_state = READ_I3_S;
    READ_I3_S  : next_state = TRIGGER_S;
    TRIGGER_S  : next_state = WAIT_IDLE_S;
    WAIT_IDLE_S: if     (tpu_busy)   next_state = WAIT_IDLE_S;
                 else if(K_cnt != K) next_state = READ_R1_S;
                 else                next_state = STORE_VAL_1_S;
    STORE_VAL_1_S: next_state = STORE_VAL_2_S;
    STORE_VAL_2_S: next_state = STORE_VAL_3_S;
    STORE_VAL_3_S: next_state = STORE_VAL_4_S;
    STORE_VAL_4_S: next_state = IDLE_S;
    default: next_state = IDLE_S;
    endcase
end

assign tpu_busy = (curr_state != IDLE);

always_ff @( posedge clk_i ) begin
    if(tpu_cmd_valid && tpu_cmd) begin
        K_cnt <= 0;
    end
    else if(curr_state == TRIGGER_CONV) begin
        K_cnt <= K_cnt + 1;
    end
    else if(curr_state == IDLE) begin
        K_cnt <= 0;
    end
end

//************************
//       STORE KMN
//************************
always_ff @(posedge clk_i) begin
    if(tpu_cmd_valid && tpu_cmd == SET_KMN) begin
        if(tpu_param_1_in == 0) begin
            K <= tpu_param_2_in;
        end
        else if(tpu_param_1_in == 1) begin
            M <= tpu_param_2_in;
        end
        else if(tpu_param_1_in == 2) begin
            N <= tpu_param_2_in;
        end
    end
end

//************************
//      STORE IDX
//************************
always_ff @(posedge clk_i) begin
    if(rst_i) begin
        store_idx[0] <= 0;
        store_idx[1] <= 0;
        store_idx[2] <= 0;
        store_idx[3] <= 0;
    end
    else if(tpu_cmd_valid && tpu_cmd == SET_ST_IDX) begin
        store_idx[tpu_param_1_in] <= tpu_param_2_in;
    end
    else if(curr_state == STORE_VAL_4_S) begin
        store_idx[0] <= store_idx[0] + 1;
        store_idx[1] <= store_idx[1] + 1;
        store_idx[2] <= store_idx[2] + 1;
        store_idx[3] <= store_idx[3] + 1;
    end
    
end
//************************
//      GBUFF SEL
//************************
always_comb begin
    if(gbuff_sel == 0) begin
        gbuff_data   = A_data_out;
        gbuff_weight = W_data_out;
    end
    else begin
        gbuff_data   = B_data_out;
        gbuff_weight = W_data_out;
    end
end

always_ff @(posedge clk_i) begin
    if(curr_state == READ_R4_S) begin
        row_input [0] <= gbuff_data;
        col_input [0] <= gbuff_weight;
    end
    if(curr_state == READ_I1_S) begin
        row_input [1] <= gbuff_data;
        col_input [1] <= gbuff_weight;
    end
    if(curr_state == READ_I2_S) begin
        row_input [2] <= gbuff_data;
        col_input [2] <= gbuff_weight;
    end
    if(curr_state == READ_I3_S) begin
        row_input [3] <= gbuff_data;
        col_input [3] <= gbuff_weight;
    end
end

always_ff @(posedge clk_i) begin
    if(rst_i) begin
        row_idx[0] <= 0;
        row_idx[1] <= 0;
        row_idx[2] <= 0;
        row_idx[3] <= 0;
    end
    else if(tpu_cmd_valid && tpu_cmd == SET_ROW_IDX)begin
        row_idx[tpu_param_1_in] <= tpu_param_2_in;
    end
end

always_comb begin
    case (curr_state)
        READ_R1_S: I_index = row_idx[0];
        READ_R2_S: I_index = row_idx[1];
        READ_R3_S: I_index = row_idx[2];
        READ_R4_S: I_index = row_idx[3];
        default: I_index = 0;
    endcase
end

//************************
//     GBUFF A STATUS
//************************
always_ff @(posedge clk_i) begin
    if(rst_i) begin
        gbuff_A_status <= TPU_READ;
    end
    else if(tpu_cmd_valid && tpu_cmd == SW_WRITE_RAM_A) begin
        gbuff_A_status <= SW_WRITE;
    end
    else begin
        gbuff_A_status <= TPU_READ;
    end
    
end
//************************
//     GBUFF B STATUS
//************************
always_ff @(posedge clk_i) begin
    if(rst_i) begin
        gbuff_B_status <= TPU_READ;
    end
    else if(tpu_cmd_valid && tpu_cmd == SW_WRITE_RAM_B) begin
        gbuff_B_status <= SW_WRITE;
    end
    else begin
        gbuff_B_status <= TPU_READ;
    end
    
end

always_ff @(posedge clk_i) begin
    A_index <=    (gbuff_A_status == SW_READ   ) ? param_1_in_reg
                : (gbuff_A_status == SW_WRITE  ) ? param_1_in_reg 
                : (gbuff_A_status == TPU_READ  ) ? I_data_out 
                : (gbuff_A_status == TPU_WRITE ) ? 0 
                : (gbuff_A_status == DRAM_READ ) ? 0  
                : (gbuff_A_status == DRAM_WRITE) ? 0  
                : 0;
end

always_ff @(posedge clk_i) begin
    B_index <=    (gbuff_B_status == SW_READ   ) ? param_1_in_reg
                : (gbuff_B_status == SW_WRITE  ) ? param_1_in_reg 
                : (gbuff_B_status == TPU_READ  ) ? I_data_out 
                : (gbuff_B_status == TPU_WRITE ) ? 0 
                : (gbuff_B_status == DRAM_READ ) ? 0  
                : (gbuff_B_status == DRAM_WRITE) ? 0  
                : 0;
end

// store output
always_ff @(posedge clk_i) begin
    C_index <= (curr_state == STORE_VAL_1_S) ? store_idx[0]
             : (curr_state == STORE_VAL_2_S) ? store_idx[1]
             : (curr_state == STORE_VAL_3_S) ? store_idx[2]
             : (curr_state == STORE_VAL_4_S) ? store_idx[3]
             : 0;
end
// always_ff @(posedge clk_i) begin
//     A_index <=    (gbuff_A_status == SW_READ   ) ? param_1_in_reg
//                 : (gbuff_A_status == SW_WRITE  ) ? param_1_in_reg // 
//                 : (gbuff_A_status == TPU_READ  ) ? I_data_out //
//                 //  : (gbuff_A_status == TPU_READ_2) ? row_idx[1] // 
//                 //  : (gbuff_A_status == TPU_READ_3) ? row_idx[2] // 
//                 //  : (gbuff_A_status == TPU_READ_4) ? row_idx[3] // 
//                 : (gbuff_A_status == TPU_WRITE ) ? 0 // 
//                 : (gbuff_A_status == DRAM_READ ) ? 0 // 
//                 : (gbuff_A_status == DRAM_WRITE) ? 0 // 
//                 : 0;
// end

// to do : A_wr_en to be register
// assign A_wr_en =   (gbuff_A_status == SW_READ   ) ? 0
//                  : (gbuff_A_status == SW_WRITE  ) ? 1 // 
//                  : (gbuff_A_status == TPU_READ  ) ? 0 // 
//                  : (gbuff_A_status == TPU_WRITE ) ? 1 // 
//                  : (gbuff_A_status == DRAM_READ ) ? 0 // 
//                  : (gbuff_A_status == DRAM_WRITE) ? 1 // 
//                  : 0;

always_ff @(posedge clk_i) begin
    A_wr_en <=    (gbuff_A_status == SW_READ   ) ? 0
                : (gbuff_A_status == SW_WRITE  ) ? 1 
                : (gbuff_A_status == TPU_READ  ) ? 0 
                : (gbuff_A_status == TPU_WRITE ) ? 1 
                : (gbuff_A_status == DRAM_READ ) ? 0  
                : (gbuff_A_status == DRAM_WRITE) ? 1  
                : 0;
end

always_ff @(posedge clk_i) begin
    B_wr_en <=    (gbuff_B_status == SW_READ   ) ? 0
                : (gbuff_B_status == SW_WRITE  ) ? 1 
                : (gbuff_B_status == TPU_READ  ) ? 0 
                : (gbuff_B_status == TPU_WRITE ) ? 1 
                : (gbuff_B_status == DRAM_READ ) ? 0  
                : (gbuff_B_status == DRAM_WRITE) ? 1  
                : 0;
end

//************************
//    GBUFF_C enbale
//************************
always_ff @(posedge clk_i) begin
    C_wr_en <= (curr_state == STORE_VAL_1_S) ? 1
             : (curr_state == STORE_VAL_2_S) ? 1
             : (curr_state == STORE_VAL_3_S) ? 1
             : (curr_state == STORE_VAL_4_S) ? 1
             : 0;
end



//************************
//     MMU CMD VALID
//************************
always_comb begin
    if(curr_state == TRIGGER_S         || curr_state == SET_MUL_VAL_S ||
       curr_state == SET_ADD_VAL_S   || curr_state == SET_PE_VAL_S  ||
       curr_state == SET_CONV_MODE_S || curr_state == SET_FIX_MAC_MODE_S) begin
        mmu_cmd_valid = 1; 
    end
    else begin
        mmu_cmd_valid = 0;
    end
end
//************************
//       MMU CMD
//************************
always_comb begin
    mmu_cmd        = tpu_cmd_reg;
    mmu_param_1_in = param_1_in_reg;
    mmu_param_2_in = param_2_in_reg;
end
//************************
// MATRIX MULTIPLE ENGINE
//************************
MMU M1
//#(
//)
(
    .clk_i(clk_i), 
    .rst_i(rst_i),
    .mmu_cmd_valid(mmu_cmd_valid), 
    .mmu_cmd(mmu_cmd),
    .param_1_in(mmu_param_1_in),
    .param_2_in(mmu_param_2_in),
    .data_1_in(row_input[0]),
    .data_2_in(row_input[1]),
    .data_3_in(row_input[2]),
    .data_4_in(row_input[3]),
    .weight_1_in(col_input[0]),
    .weight_2_in(col_input[1]),
    .weight_3_in(col_input[2]),
    .weight_4_in(col_input[3]),
    .rdata_1_out(rdata_1_out),
    .rdata_2_out(rdata_2_out),
    .rdata_3_out(rdata_3_out),
    .rdata_4_out(rdata_4_out),
    .mmu_busy(mmu_busy)
);
//************************
//       SOFTMAX
//************************

//************************
//    AVG / MAX POOLING
//************************

//************************
//       gbuff_A
//************************
global_buffer #(
    .ADDR_BITS(16),
    .DATA_BITS(32)
)
gbuff_A(
    .clk_i(clk_i),
    .rst_i(rst_i),
    .wr_en(A_wr_en),
    .index(A_index),
    .data_in(A_data_in),
    .data_out(A_data_out)
);
//************************
//       gbuff_B
//************************
global_buffer #(
    .ADDR_BITS(16),
    .DATA_BITS(32)
)
gbuff_B(
    .clk_i(clk_i),
    .rst_i(rst_i),
    .wr_en(B_wr_en),
    .index(B_index),
    .data_in(B_data_in),
    .data_out(B_data_out)
);

//************************
//    gbuff_C (output)
//************************
global_buffer #(
    .ADDR_BITS(16),
    .DATA_BITS(128)
)
gbuff_C(
    .clk_i(clk_i),
    .rst_i(rst_i),
    .wr_en(C_wr_en),
    .index(C_index),
    .data_in(C_data_in),
    .data_out(C_data_out)
);

//************************
//       gbuff_Index
//************************
global_buffer #(
    .ADDR_BITS(16),
    .DATA_BITS(32)
)
gbuff_Index_Ram(
    .clk_i(clk_i),
    .rst_i(rst_i),
    .wr_en(I_wr_en),
    .index(I_index),
    .data_in(I_data_in),
    .data_out(I_data_out)
);

//************************
//     gbuff_weight
//************************
global_buffer #(
    .ADDR_BITS(16),
    .DATA_BITS(32)
)
gbuff_W(
    .clk_i(clk_i),
    .rst_i(rst_i),
    .wr_en(W_wr_en),
    .index(W_index),
    .data_in(W_data_in),
    .data_out(W_data_out)
);

endmodule
