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
    input   logic   [DATA_WIDTH-1 : 0]   tpu_param_2_in,     // data 2

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
// localparam  SET_COL_IDX        = 15;
localparam  SET_PRELOAD               = 15; // whole
// localparam  IDLE               = 16; // whole


typedef enum {IDLE_S, HW_RESET_S, 
              LOAD_IDX_S,
              READ_DATA_1_S, READ_DATA_2_S, 
              READ_DATA_3_S, READ_DATA_4_S,
              READ_DATA_5_S, //READ_DATA_6_S,
              TRIGGER_S, TRIGGER_LAST_S,
              FORWARD_S,
              WAIT_IDLE_S,
              STORE_S, 
              NEXT_ROW_S,
              NEXT_COL_S,
              PRELOAD_DATA_S,
              SET_MUL_VAL_S, 
              SET_ADD_VAL_S, 
              SET_PE_VAL_S, 
              SET_CONV_MODE_S, 
              SET_FIX_MAC_MODE_S,
              SW_READ_DATA_S,
              OUTPUT_1_S,
              OUTPUT_2_S,
              OUTPUT_3_S} state_t;
state_t curr_state, next_state;


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
// logic   [DATA_WIDTH*4-1 : 0] rdata_1_out;
// logic   [DATA_WIDTH*4-1 : 0] rdata_2_out;
// logic   [DATA_WIDTH*4-1 : 0] rdata_3_out;
// logic   [DATA_WIDTH*4-1 : 0] rdata_4_out; 

// logic                        tpu_busy;     // 0->idle, 1->busy
logic                        mmu_busy;     // 0->idle, 1->busy
// logic   [DATA_WIDTH*4-1 : 0] mmu_out [0 : 3]; 
// logic gbuff_sel; // 0-> TPU READ GBUFF A
//                  //     SW / DRAM WRITE GBUFF B
//                  // 1-> TPU READ GBUFF B
//                  //     SW / DRAM WRITE GBUFF A


//#########################
//#    INDEX START REG    #
//#########################
/*
row_idx: base address
*/
logic [ADDR_BITS-1 : 0] row_idx       [0 : 3];
logic [ADDR_BITS-1 : 0] row_acc;
logic [ADDR_BITS-1 : 0] row_offset;
logic [ADDR_BITS-1 : 0] row_idx_start [0 : 3];
logic [ADDR_BITS-1 : 0] col_idx       [0 : 3];
logic [ADDR_BITS-1 : 0] col_acc;
logic [ADDR_BITS-1 : 0] col_offset;
logic [ADDR_BITS-1 : 0] col_idx_start [0 : 3];
logic [DATA_BITS-1 : 0] row_input     [0 : 3];
logic [DATA_BITS-1 : 0] col_input     [0 : 3];

logic [ADDR_BITS-1 : 0] store_idx [0 : 3];

// logic [ADDR_BITS-1 : 0] I_index;

//#########################
//#    K     M     N      #
//#########################
logic [ADDR_BITS-1 : 0] K_reg, M_reg, N_reg; // (M * K) (K * N)

logic                          mmu_cmd_valid;  // cmd
logic   [ACLEN : 0]            mmu_cmd;        // cmd
logic   [DATA_WIDTH*4-1 : 0]   mmu_param_in [0 : 3]; // data 1
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

assign conv_end      = (row_acc + 4 >= M_reg && col_acc + 4 >= N_reg);
assign conv_next_row = (row_acc + 4 < M_reg);
assign conv_next_col = (col_acc + 4 < N_reg);
// logic   [ADDR_BITS-1  : 0]   conv_len_cnt;

always_ff @( posedge clk_i ) begin
    if(tpu_cmd_valid && tpu_cmd == SET_PRELOAD) begin
        preload <= tpu_param_1_in;
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
            else if(tpu_cmd_valid && tpu_cmd == SET_MUL_VAL     ) next_state = SET_MUL_VAL_S;
            else if(tpu_cmd_valid && tpu_cmd == SET_ADD_VAL     ) next_state = SET_ADD_VAL_S;
            else if(tpu_cmd_valid && tpu_cmd == SET_PE_VAL    ) next_state = SET_PE_VAL_S;
            // else if(tpu_cmd_valid && tpu_cmd == SET_CONV_MODE   ) next_state = SET_CONV_MODE_S;
            // else if(tpu_cmd_valid && tpu_cmd == SET_FIX_MAC_MODE) next_state = SET_FIX_MAC_MODE_S;
            else if(tpu_cmd_valid && tpu_cmd == SW_READ_DATA    ) next_state = SW_READ_DATA_S;
            else         next_state = IDLE_S;
    HW_RESET_S    : next_state = IDLE_S;
    LOAD_IDX_S    : next_state = READ_DATA_1_S;
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
                 else            next_state = STORE_S;
    STORE_S: if(conv_end)           next_state = IDLE_S;
             else if(conv_next_row) next_state = NEXT_ROW_S;
             else                   next_state = NEXT_COL_S;
    NEXT_ROW_S: next_state = PRELOAD_DATA_S;
    NEXT_COL_S: next_state = PRELOAD_DATA_S;
    PRELOAD_DATA_S: next_state = READ_DATA_1_S;
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
//#       STORE IDX       #
//#########################
// always_ff @(posedge clk_i) begin
//     if(rst_i) begin
//         store_idx[0] <= 0;
//         store_idx[1] <= 0;
//         store_idx[2] <= 0;
//         store_idx[3] <= 0;
//     end
//     else if(tpu_cmd_valid && tpu_cmd == SET_ST_IDX) begin
//         store_idx[tpu_param_1_in] <= tpu_param_2_in;
//     end
//     else if(curr_state == STORE_VAL_4_S) begin
//         store_idx[0] <= store_idx[0] + 1;
//         store_idx[1] <= store_idx[1] + 1;
//         store_idx[2] <= store_idx[2] + 1;
//         store_idx[3] <= store_idx[3] + 1;
//     end
// end

//#########################
//#      SEND OUTPUT      #
//#########################
always_comb begin
    if(curr_state == OUTPUT_3_S) begin
        ret_valid = 1;
        case (param_1_in_reg)
            0 : ret_data_out = P_data_out_reg[0][127 : 96];
            1 : ret_data_out = P_data_out_reg[0][95  : 64];
            2 : ret_data_out = P_data_out_reg[0][63  : 32];
            3 : ret_data_out = P_data_out_reg[0][31  :  0];
            4 : ret_data_out = P_data_out_reg[1][127 : 96];
            5 : ret_data_out = P_data_out_reg[1][95  : 64];
            6 : ret_data_out = P_data_out_reg[1][63  : 32];
            7 : ret_data_out = P_data_out_reg[1][31  :  0];
            8 : ret_data_out = P_data_out_reg[2][127 : 96];
            9 : ret_data_out = P_data_out_reg[2][95  : 64];
            10: ret_data_out = P_data_out_reg[2][63  : 32];
            11: ret_data_out = P_data_out_reg[2][31  :  0];
            12: ret_data_out = P_data_out_reg[3][127 : 96];
            13: ret_data_out = P_data_out_reg[3][95  : 64];
            14: ret_data_out = P_data_out_reg[3][63  : 32];
            15: ret_data_out = P_data_out_reg[3][31  :  0];
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
    if(next_state == LOAD_IDX_S) begin
        for(int i = 0; i < 4; i++) begin
            row_idx[i] <= row_idx_start[i];
        end
    end
    else if(curr_state == NEXT_ROW_S) begin
        for(int i = 0; i < 4; i++) begin
            row_idx[i] <= row_idx[i] + K_cnt * 4;
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
    else if(curr_state == LOAD_IDX_S    ||
            curr_state == PRELOAD_DATA_S||
            curr_state == READ_DATA_1_S || 
            curr_state == READ_DATA_2_S || 
            curr_state == READ_DATA_3_S || 
            curr_state == READ_DATA_4_S ||
            curr_state == READ_DATA_5_S || 
            // curr_state == READ_DATA_6_S ||
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
    if(tpu_cmd_valid && tpu_cmd == SET_ROW_IDX)begin
        col_idx_start[tpu_param_1_in] <= tpu_param_2_in;
    end
end

//#########################
//#       COL INDEX       #
//#########################
always_ff @(posedge clk_i) begin
    if(next_state == LOAD_IDX_S) begin
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
                        : (gbuff_status[i] == TPU_READ  ) ? I_out_reg[i] 
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
        P_data_in[i] <= rdata_out[i];
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
       (curr_state == PRELOAD_DATA_S && preload)) begin
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
    else begin
        mmu_cmd         = tpu_cmd_reg;
    end
    mmu_param_in[0] = param_1_in_reg;
    mmu_param_in[1] = param_2_in_reg;
    mmu_param_in[2] = P_data_out_reg[2];
    mmu_param_in[3] = P_data_out_reg[3];
    
end
//##########################
//# MATRIX MULTIPLE ENGINE #
//##########################
MMU M1
//#(
//)
(
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

//#########################
//    AVG / MAX POOLING
//#########################



//#########################
//#      LINE BUFFER      #
//#########################
generate
for (genvar i = 0; i < 4; i++) begin
    global_buffer #(
    .ADDR_BITS(14),
    .DATA_BITS(32)
    )
    gbuff (
        .clk_i(clk_i),
        .rst_i(rst_i),
        .wr_en(gbuff_wr_en[i]),
        .index(gbuff_index[i]),
        .data_in(gbuff_data_in[i]),
        .data_out(gbuff_data_out[i])
    );
end
endgenerate

//#########################
//#     WEIGHT BUFFER     #
//#########################
generate
for (genvar i = 0; i < 4; i++) begin
    global_buffer #(
    .ADDR_BITS(14),
    .DATA_BITS(32)
    )
    weight (
        .clk_i(clk_i),
        .rst_i(rst_i),
        .wr_en(weight_wr_en[i]),
        .index(weight_index[i]),
        .data_in(weight_in[i]),
        .data_out(weight_out[i])
    );
end
endgenerate

//#########################
//#       I BUFFER        #
//#########################
generate
for (genvar i = 0; i < 4; i++) begin
    global_buffer #(
    .ADDR_BITS(14),
    .DATA_BITS(32)
    )
    I_gbuff (
        .clk_i(clk_i),
        .rst_i(rst_i),
        .wr_en(I_wr_en[i]),
        .index(I_index[i]),
        .data_in(I_in[i]),
        .data_out(I_out[i])
    );
end
endgenerate

//#########################
//#   PARTIAL SUM BUFFER  #
//#########################
generate
for (genvar i = 0; i < 4; i++) begin
    global_buffer #(
    .ADDR_BITS(12),
    .DATA_BITS(128)
    )
    P_gbuff (
        .clk_i(clk_i),
        .rst_i(rst_i),
        .wr_en(P_wr_en[i]),
        .index(P_index[i]),
        .data_in(P_data_in[i]),
        .data_out(P_data_out[i])
    );
end
endgenerate
endmodule