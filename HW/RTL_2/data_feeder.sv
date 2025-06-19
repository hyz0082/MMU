module data_feeder 
#( parameter XLEN = 32, 
   parameter BUF_ADDR_LEN = 32, 
   parameter ACLEN  = 8,
   parameter ADDR_BITS=15,
   parameter DATA_WIDTH = 16,
   parameter GEMM_NUM   = 4)
(
    input                           clk_i,
    input                           rst_i,

    input                           S_DEVICE_strobe_i, 
    input [BUF_ADDR_LEN-1 : 0]      S_DEVICE_addr_i,
    input                           S_DEVICE_rw_i,
    input [XLEN/8-1 : 0]            S_DEVICE_byte_enable_i,
    input [XLEN-1 : 0]              S_DEVICE_data_i,

    // to aquila
    output logic                      S_DEVICE_ready_o,
    output logic [XLEN-1 : 0]         S_DEVICE_data_o,

    // to cdc
    input  logic              fifo_addr_full_i,
                            input  logic       dram_rw_full_i,
                            input  logic       dram_write_data_h_full_i,
                            input  logic       dram_write_data_l_full_i,
                            input  logic       dram_str_idx_full_i,
                            input  logic       dram_end_idx_full_i,

output logic              dram_addr_valid_o,
output logic [XLEN-1 : 0] dram_addr_o,
output logic              rw_o,
output logic [511 : 0]    dram_data_o,
output logic [4 : 0] data_start_idx_o,
output logic [4 : 0] data_end_idx_o,

(* mark_debug="true" *)     input  logic          fifo_data_empty_i,
    output logic          fifo_data_rd_en_o,
    input logic[255 : 0] dram_read_data_h_i,
    input logic[255 : 0] dram_read_data_l_i,

    input logic dram_write_done_fifo_empty_i,
    input logic dram_write_done_i
);
`include "config.svh"
// `include "interface.svh"
/*
 *  DRAM ACCESS FSM
 */
typedef enum {IDLE_S,
              WAIT_FIFO_ADDR_S,
              WAIT_FIFO_DATA_S,
              SEND_REQ_S,
              READ_S,
              WAIT_READ_S,
              WRITE_INPUT_S,
              WRITE_WEIGHT_S,
              READ_NEXT_DATA_S,
              WRITE_S,
              COLLECT_OUTPUT_S,
              WAIT_GEMM_DATA_S,
              WRITE_DRAM_DATA_S,
              WAIT_GEMM_IDLE_S,
              DUMMY_1_S,
              WAIT_WRITE_DONE_S,
              READ_NEXT_ROUND_S,
              RECV_NEXT_ROUND_S,
              RESET_SRAM_S
              } state_t;
state_t send_req_curr_state;
state_t send_req_next_state;
state_t write_data_curr_state;
state_t write_data_next_state;
state_t write_dram_curr_state;
state_t write_dram_next_state;

// 0xC4000000
logic [GEMM_NUM : 0] gemm_core_sel; // 0: gemm core 0, 1: gemm core 1
(* mark_debug="true" *) logic                        tpu_cmd_valid;     // tpu valid
(* mark_debug="true" *) logic   [ACLEN-1 : 0]        tpu_cmd;
logic  [DATA_WIDTH-1 : 0] S_DEVICE_data_i_t;
assign S_DEVICE_data_i_t = S_DEVICE_data_i;
// 0xC4000004
(* mark_debug="true" *) logic   [DATA_WIDTH-1 : 0]   tpu_param_1_in;    // data 1
// 0xC4000008
(* mark_debug="true" *) logic   [DATA_WIDTH-1 : 0]   tpu_param_2_in;     // data 2

logic   [GEMM_NUM-1 : 0]   ret_valid;
logic   [DATA_WIDTH-1 : 0] ret_data_out    [0 : GEMM_NUM-1];
logic   [DATA_WIDTH-1 : 0] ret_max_pooling [0 : GEMM_NUM-1];
logic   [DATA_WIDTH-1 : 0] ret_avg_pooling [0 : GEMM_NUM-1];
logic   [DATA_WIDTH-1 : 0] ret_softmax_result [0 : GEMM_NUM-1];
// 0xC400000A
(* mark_debug="true" *) logic                      tpu_busy [0 : GEMM_NUM-1 + 4];     // 0->idle, 1->busy
// 0xC4000010
logic   [DATA_WIDTH-1 : 0] ret_data_out_reg;

logic   [DATA_WIDTH*4-1 : 0]   tpu_data_1_in;
logic   [DATA_WIDTH*4-1 : 0]   tpu_data_2_in;
logic   [DATA_WIDTH*4-1 : 0]   tpu_data_3_in;
logic   [DATA_WIDTH*4-1 : 0]   tpu_data_4_in;

//############################
//# MEMORY ARBITER INTERFACE #
//############################
/*
    signal connect to cdc
    addr   : C4002024
    length : C4002028
    rw_r   : C400202C
*/
logic [XLEN-1 : 0] addr; // + 64 each times (512 bits)
logic rw_r;  // 3'b000 for read, 3'b001 for write
logic [511 : 0] dram_data_read;
logic [511 : 0] dram_data_write;
logic [63:0] byte_mask_r;

logic [XLEN-1 : 0] got_addr;

/*
 * MULTIPLE READ SIGNAL
 */
logic [XLEN-1 : 0] read_offset;
logic [15 : 0]     read_rounds;

logic [XLEN-1 : 0] read_addr_base;
logic [15 : 0]     read_rounds_cnt;

logic [XLEN-1 : 0] recv_addr_base;
logic [15 : 0]     recv_rounds_cnt;

logic [6 : 0] dram_write_addr_offset;

logic [255 : 0] dram_read_data_h_r;
logic [255 : 0] dram_read_data_l_r;

logic [2 : 0] write_data_type; // 0: input, 1: weight, 2: batchNorm mul 
                               // 2: batchNorm add
logic [11 : 0] batchNormIndex;
logic [1  : 0] batchNormOffset;

/*
    get 512 bits data per dram read
    512 / 16 = 32 inputs
*/
logic [DATA_WIDTH-1 : 0] data_in [0 : 31];
logic [ADDR_BITS-1 : 0] req_len_cnt;
logic [5 : 0]           req_len; // max req len = 32
logic [ADDR_BITS-1 : 0] got_len_cnt;
logic [5 : 0]           got_len_expect; // max req len = 32
logic [ADDR_BITS-1 : 0] send_cnt;
logic [ADDR_BITS-1 : 0] length;
logic [3 : 0] word_size;
(* mark_debug="true" *) logic [ADDR_BITS-1 : 0] sram_offset;
logic [3 : 0] data_type;
logic [6 : 0] addr_offset;


(* mark_debug="true" *) logic [9 : 0]           currInputType;

(* mark_debug="true" *) logic [ADDR_BITS-1 : 0] inputLength;
(* mark_debug="true" *) logic [ADDR_BITS-1 : 0] sramWriteCount;

(* mark_debug="true" *) logic [ADDR_BITS-1 : 0] inputHeight;
(* mark_debug="true" *) logic [ADDR_BITS-1 : 0] inputHeightCount;

logic [ADDR_BITS-1 : 0] paddingSize;
logic [ADDR_BITS-1 : 0] boundaryPaddingSize;

logic [ADDR_BITS-1 : 0] sramResetIndex;

/*
 * DRAM WRITE SIGNAL
 *  
 */
logic [ADDR_BITS-1  : 0] output_recv_cnt;
logic [ADDR_BITS-1  : 0] dram_addr_offset;
logic [DATA_WIDTH-1 : 0] dram_data_r [0 : 150];

logic [DATA_WIDTH-1 : 0] dram_data_reorder_r [0 : 69];
logic [XLEN-1 : 0]       dram_write_addr [0 : 3];
logic [ADDR_BITS-1  : 0] dram_write_length;
logic [ADDR_BITS-1  : 0] dram_write_length_cnt;
logic [ADDR_BITS-1  : 0] target_idx;
logic [7 : 0] num_lans;//, send_lans_cnt;
logic [DATA_WIDTH*4-1 : 0] P_data_out_r [0 : 3];
logic [DATA_WIDTH*4-1 : 0] P_data_out_1 [0 : GEMM_NUM-1];//  [0 : 3];
logic [DATA_WIDTH*4-1 : 0] P_data_out_2 [0 : GEMM_NUM-1];//[0 : 3];
logic [DATA_WIDTH*4-1 : 0] P_data_out_3 [0 : GEMM_NUM-1];//[0 : 3];
logic [DATA_WIDTH*4-1 : 0] P_data_out_4 [0 : GEMM_NUM-1];//[0 : 3];

logic [ADDR_BITS-1 : 0] GeMMOutputSel;

/*
 * SOFTWARE READ / WRITE DATA
 *  
 */
logic [DATA_WIDTH-1 : 0] sw_data_r;
logic [DATA_WIDTH-1 : 0] sw_write_data_r;
logic sw_write_dram_mode;
logic rw_to_gemm;

logic                     gbuff_wr_en   [0 : 3];
logic  [ADDR_BITS-1  : 0] gbuff_index   [0 : 3];
logic  [DATA_WIDTH-1 : 0] gbuff_data_in [0 : 3];
logic  [DATA_WIDTH-1 : 0] gbuff_data_out[0 : 3];

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        batchNormOffset <= 0;
        batchNormIndex  <= 0;
    end
    else if(write_data_curr_state == IDLE_S) begin
        batchNormOffset <= 0;
        batchNormIndex  <= 0;
    end
    else if(write_data_curr_state == WRITE_INPUT_S) begin
        batchNormOffset <= batchNormOffset + 1;
        if(batchNormOffset == 3) begin
            batchNormIndex <= batchNormIndex + 1;
        end
    end
end


always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        gemm_core_sel <= 1;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == GEMM_CORE_SEL_ADDR) begin
        gemm_core_sel <= S_DEVICE_data_i;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        rw_to_gemm <= 0;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == DRAM_TR || 
            S_DEVICE_strobe_i && S_DEVICE_addr_i == TR_DRAM_W) begin
        rw_to_gemm <= S_DEVICE_data_i;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        sw_write_dram_mode <= 0;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == SW_WRITE_DRAM_MODE_ADDR) begin
        sw_write_dram_mode <= S_DEVICE_data_i;
    end
end


always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        write_data_type <= 0;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == WRITE_DATA_TYPE_ADDR) begin
        write_data_type <= S_DEVICE_data_i;
    end
end

always_ff @( posedge clk_i ) begin
    if(ret_valid[0]) begin
        ret_data_out_reg <= ret_data_out[0];
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        S_DEVICE_data_o  <= 0;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == BUSY_ADDR) begin
        S_DEVICE_data_o  <= tpu_busy[0] || 
                            (send_req_curr_state   != IDLE_S) ||
                            (write_data_curr_state != IDLE_S) ||
                            (write_dram_curr_state != IDLE_S);
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == BUSY_ADDR_2) begin
        S_DEVICE_data_o  <= tpu_busy[1] || 
                            (send_req_curr_state   != IDLE_S) ||
                            (write_data_curr_state != IDLE_S) ||
                            (write_dram_curr_state != IDLE_S);
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == BUSY_ADDR_3) begin
        S_DEVICE_data_o  <= tpu_busy[2] || 
                            (send_req_curr_state   != IDLE_S) ||
                            (write_data_curr_state != IDLE_S) ||
                            (write_dram_curr_state != IDLE_S);
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == BUSY_ADDR_4) begin
        S_DEVICE_data_o  <= tpu_busy[3] || 
                            (send_req_curr_state   != IDLE_S) ||
                            (write_data_curr_state != IDLE_S) ||
                            (write_dram_curr_state != IDLE_S);
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == RET_ADDR) begin
        S_DEVICE_data_o  <= {16'h0, ret_data_out_reg};
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == RET_MAX_POOLING_ADDR) begin
        S_DEVICE_data_o  <= ret_max_pooling[0];
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == RET_AVG_POOLING_ADDR) begin
        S_DEVICE_data_o  <= ret_avg_pooling[0];
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == RET_SOFTMAX_ADDR) begin
        S_DEVICE_data_o  <= ret_softmax_result[0];
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == SW_DATA_ADDR) begin
        S_DEVICE_data_o  <= sw_data_r;
    end
end


always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        S_DEVICE_ready_o <= 0;
    end
    else if(S_DEVICE_strobe_i) begin
        S_DEVICE_ready_o <= 1;
    end
    else begin
        S_DEVICE_ready_o <= 0;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        tpu_cmd_valid  <= 0;
        tpu_cmd        <= 0;
        tpu_param_1_in <= 0;
        tpu_param_2_in <= 0;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == TPU_CMD_ADDR) begin
        tpu_cmd_valid <= 1;
        tpu_cmd  <= S_DEVICE_data_i;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == PARAM_1_ADDR) begin
        tpu_param_1_in <= S_DEVICE_data_i;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == PARAM_2_ADDR) begin
        tpu_param_2_in <= S_DEVICE_data_i;
    end
    else if(write_data_curr_state == WRITE_INPUT_S && write_data_type == 0) begin
        tpu_cmd_valid  <= 1;
        tpu_cmd        <= 9;
        tpu_param_1_in <= data_in[got_addr[5:1]];// data_in
        tpu_param_2_in <= sram_offset;// index
    end
    else if(write_data_curr_state == WRITE_INPUT_S && write_data_type == 1) begin
        tpu_cmd_valid  <= 1;
        tpu_cmd        <= 10;
        tpu_param_1_in <= data_in[got_addr[5:1]];// data_in
        tpu_param_2_in <= sram_offset;// index
    end
    else if(write_data_curr_state == RESET_SRAM_S) begin
        tpu_cmd_valid  <= 1;
        tpu_cmd        <= 9;
        tpu_param_1_in <= 0;// data_in
        tpu_param_2_in <= sram_offset;// index
    end
    else if(write_data_curr_state == WRITE_INPUT_S && write_data_type == 2) begin
        tpu_cmd_valid  <= 1;
        if     (batchNormOffset == 0) tpu_cmd <= 26;
        else if(batchNormOffset == 1) tpu_cmd <= 27;
        else if(batchNormOffset == 2) tpu_cmd <= 28;
        else                          tpu_cmd <= 29;
        tpu_param_1_in <= data_in[got_addr[5:1]];
        tpu_param_2_in <= batchNormIndex;
    end
    else if(write_data_curr_state == WRITE_INPUT_S && write_data_type == 3) begin
        tpu_cmd_valid  <= 1;
        if     (batchNormOffset == 0) tpu_cmd <= 30;
        else if(batchNormOffset == 1) tpu_cmd <= 31;
        else if(batchNormOffset == 2) tpu_cmd <= 32;
        else                          tpu_cmd <= 33;
        tpu_param_1_in <= data_in[got_addr[5:1]];
        tpu_param_2_in <= batchNormIndex;
    end
    else if(write_dram_curr_state ==  COLLECT_OUTPUT_S) begin
        tpu_cmd_valid  <= 1;
        tpu_cmd        <= 14;
        // tpu_param_1_in <= 0;// data_in
        tpu_param_2_in <= output_recv_cnt;// index
    end
    else begin
        tpu_cmd_valid <= 0;
    end
end


global_buffer_dp #(
.ADDR_BITS(ADDR_BITS), // ADDR_BITS
.DATA_BITS(DATA_WIDTH)  // DATA_WIDTH
)
gbuff_1 (
    .clk_i   (clk_i),

    .wr_en_1   (gbuff_wr_en[0]),
    .index_1   (gbuff_index[0]),
    .data_in_1 (gbuff_data_in[0]),
    .data_out_1(gbuff_data_out[0]),

    .index_2(gbuff_index[1]),
    .data_in_2(gbuff_data_in[1]),
    .data_out_2(gbuff_data_out[1])
);


global_buffer_dp #(
.ADDR_BITS(ADDR_BITS), // ADDR_BITS
.DATA_BITS(DATA_WIDTH)  // DATA_WIDTH
)
gbuff_2 (
    .clk_i   (clk_i),

    .wr_en_1   (gbuff_wr_en[2]),
    .index_1   (gbuff_index[2]),
    .data_in_1 (gbuff_data_in[2]),
    .data_out_1(gbuff_data_out[2]),

    .index_2(gbuff_index[3]),
    .data_in_2(gbuff_data_in[3]),
    .data_out_2(gbuff_data_out[3])
);


TPU #(
    .ACLEN(ACLEN),
    .ADDR_BITS(ADDR_BITS),
    .DATA_WIDTH(DATA_WIDTH)
) 
t1 (
    .clk_i(clk_i), .rst_i(rst_i),
    .tpu_cmd_valid(tpu_cmd_valid && gemm_core_sel[0]),     // tpu valid
    .tpu_cmd(tpu_cmd),           // tpu
    .tpu_param_1_in(tpu_param_1_in),    // data 1
    .tpu_param_2_in(tpu_param_2_in),     // data 2
    
    .tpu_data_1_in(tpu_data_1_in),
    .tpu_data_2_in(tpu_data_2_in),
    .tpu_data_3_in(tpu_data_3_in),
    .tpu_data_4_in(tpu_data_4_in),

    .tpu_data_1_out(P_data_out_1[0]),
    .tpu_data_2_out(P_data_out_2[0]),
    .tpu_data_3_out(P_data_out_3[0]),
    .tpu_data_4_out(P_data_out_4[0]),
    
    .ret_valid(ret_valid[0]),
    .ret_data_out(ret_data_out[0]),
    .ret_max_pooling(ret_max_pooling[0]),
    .ret_avg_pooling(ret_avg_pooling[0]),
    .ret_softmax_result(ret_softmax_result[0]),

    // first dual port sram control signal
    .gbuff_wr_en_0(gbuff_wr_en[0]),
    .gbuff_index_0(gbuff_index[0]),
    .gbuff_data_in_0(gbuff_data_in[0]),
    .gbuff_data_out_0(gbuff_data_out[0]),

    .gbuff_index_1(gbuff_index[1]),
    .gbuff_data_out_1(gbuff_data_out[1]),

    // second dual port sram control signal
    .gbuff_wr_en_2(gbuff_wr_en[2]),
    .gbuff_index_2(gbuff_index[2]),
    .gbuff_data_in_2(gbuff_data_in[2]),
    .gbuff_data_out_2(gbuff_data_out[2]),

    .gbuff_index_3(gbuff_index[3]),
    .gbuff_data_out_3(gbuff_data_out[3]),

    .tpu_busy(tpu_busy[0])     
);

TPU #(
    .ACLEN(ACLEN),
    .ADDR_BITS(ADDR_BITS),
    .DATA_WIDTH(DATA_WIDTH)
) 
t2 (
    .clk_i(clk_i), .rst_i(rst_i),
    .tpu_cmd_valid(tpu_cmd_valid && gemm_core_sel[1]),     // tpu valid
    .tpu_cmd(tpu_cmd),           // tpu
    .tpu_param_1_in(tpu_param_1_in),    // data 1
    .tpu_param_2_in(tpu_param_2_in),     // data 2
    
    .tpu_data_1_in(tpu_data_1_in),
    .tpu_data_2_in(tpu_data_2_in),
    .tpu_data_3_in(tpu_data_3_in),
    .tpu_data_4_in(tpu_data_4_in),

    .tpu_data_1_out(P_data_out_1[1]),
    .tpu_data_2_out(P_data_out_2[1]),
    .tpu_data_3_out(P_data_out_3[1]),
    .tpu_data_4_out(P_data_out_4[1]),
    
    .ret_valid(ret_valid[1]),
    .ret_data_out(ret_data_out[1]),
    .ret_max_pooling(ret_max_pooling[1]),
    .ret_avg_pooling(ret_avg_pooling[1]),
    .ret_softmax_result(ret_softmax_result[1]),

    // first dual port sram control signal
    .gbuff_data_out_0(gbuff_data_out[0]),
    .gbuff_data_out_1(gbuff_data_out[1]),

    // second dual port sram control signal
    .gbuff_data_out_2(gbuff_data_out[2]),
    .gbuff_data_out_3(gbuff_data_out[3]),

    .tpu_busy(tpu_busy[1])     
);

TPU #(
    .ACLEN(ACLEN),
    .ADDR_BITS(ADDR_BITS),
    .DATA_WIDTH(DATA_WIDTH)
) 
t3 (
    .clk_i(clk_i), .rst_i(rst_i),
    .tpu_cmd_valid(tpu_cmd_valid && gemm_core_sel[2]),     // tpu valid
    .tpu_cmd(tpu_cmd),           // tpu
    .tpu_param_1_in(tpu_param_1_in),    // data 1
    .tpu_param_2_in(tpu_param_2_in),     // data 2
    
    .tpu_data_1_in(tpu_data_1_in),
    .tpu_data_2_in(tpu_data_2_in),
    .tpu_data_3_in(tpu_data_3_in),
    .tpu_data_4_in(tpu_data_4_in),

    .tpu_data_1_out(P_data_out_1[2]),
    .tpu_data_2_out(P_data_out_2[2]),
    .tpu_data_3_out(P_data_out_3[2]),
    .tpu_data_4_out(P_data_out_4[2]),
    
    .ret_valid(ret_valid[2]),
    .ret_data_out(ret_data_out[2]),
    .ret_max_pooling(ret_max_pooling[2]),
    .ret_avg_pooling(ret_avg_pooling[2]),
    .ret_softmax_result(ret_softmax_result[2]),

    // first dual port sram control signal
    .gbuff_data_out_0(gbuff_data_out[0]),
    .gbuff_data_out_1(gbuff_data_out[1]),

    // second dual port sram control signal
    .gbuff_data_out_2(gbuff_data_out[2]),
    .gbuff_data_out_3(gbuff_data_out[3]),

    .tpu_busy(tpu_busy[2])     
);

TPU #(
    .ACLEN(ACLEN),
    .ADDR_BITS(ADDR_BITS),
    .DATA_WIDTH(DATA_WIDTH)
) 
t4 (
    .clk_i(clk_i), .rst_i(rst_i),
    .tpu_cmd_valid(tpu_cmd_valid && gemm_core_sel[3]),     // tpu valid
    .tpu_cmd(tpu_cmd),           // tpu
    .tpu_param_1_in(tpu_param_1_in),    // data 1
    .tpu_param_2_in(tpu_param_2_in),     // data 2
    
    .tpu_data_1_in(tpu_data_1_in),
    .tpu_data_2_in(tpu_data_2_in),
    .tpu_data_3_in(tpu_data_3_in),
    .tpu_data_4_in(tpu_data_4_in),

    .tpu_data_1_out(P_data_out_1[3]),
    .tpu_data_2_out(P_data_out_2[3]),
    .tpu_data_3_out(P_data_out_3[3]),
    .tpu_data_4_out(P_data_out_4[3]),
    
    .ret_valid(ret_valid[3]),
    .ret_data_out(ret_data_out[3]),
    .ret_max_pooling(ret_max_pooling[3]),
    .ret_avg_pooling(ret_avg_pooling[3]),
    .ret_softmax_result(ret_softmax_result[3]),

    // first dual port sram control signal
    .gbuff_data_out_0(gbuff_data_out[0]),
    .gbuff_data_out_1(gbuff_data_out[1]),

    // second dual port sram control signal
    .gbuff_data_out_2(gbuff_data_out[2]),
    .gbuff_data_out_3(gbuff_data_out[3]),

    .tpu_busy(tpu_busy[3])     
);

/*
 * MULTIPLE READ 
 */

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        read_offset  <= 0;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == READ_OFFSET) begin
        read_offset  <= S_DEVICE_data_i;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        read_rounds  <= 1;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == READ_ROUNDS) begin
        read_rounds  <= S_DEVICE_data_i;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        read_rounds_cnt <= 0;
    end
    else if(send_req_curr_state == IDLE_S) begin
        read_rounds_cnt <= 0;
    end
    else if(send_req_curr_state == READ_NEXT_ROUND_S) begin
        read_rounds_cnt <= read_rounds_cnt + 1;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        recv_rounds_cnt <= 0;
    end
    else if(write_data_curr_state == IDLE_S) begin
        recv_rounds_cnt <= 0;
    end
    else if(write_data_curr_state == RECV_NEXT_ROUND_S) begin
        recv_rounds_cnt <= recv_rounds_cnt + 1;
    end
end


//############################
//# MEMORY ARBITER INTERFACE #
//############################
/*
 * FSM (DRAM REQ)
 */
always_ff @(posedge clk_i) begin
    if(rst_i) begin
        send_req_curr_state <= IDLE_S;
    end
    else begin
        send_req_curr_state <= send_req_next_state;
    end
end

always_comb begin
    case (send_req_curr_state)
    IDLE_S: if(S_DEVICE_strobe_i && S_DEVICE_addr_i == DRAM_TR) 
                send_req_next_state = WAIT_FIFO_ADDR_S;
            else
                send_req_next_state = IDLE_S;
    WAIT_FIFO_ADDR_S: if(!fifo_addr_full_i && 
                         !dram_rw_full_i) 
                        send_req_next_state = SEND_REQ_S;
                      else 
                        send_req_next_state = WAIT_FIFO_ADDR_S;
    SEND_REQ_S: if(req_len_cnt + (addr_offset >> 1) >= length) //32
                    send_req_next_state = READ_NEXT_ROUND_S;
                else 
                    send_req_next_state = WAIT_FIFO_ADDR_S;
    READ_NEXT_ROUND_S:  if(read_rounds_cnt + 1 == read_rounds)
                            send_req_next_state = IDLE_S;
                        else
                            send_req_next_state = WAIT_FIFO_ADDR_S;
    default: send_req_next_state = IDLE_S;
    endcase
end

always_comb begin
    dram_addr_o = (send_req_curr_state == SEND_REQ_S) ? addr
                                                      : dram_write_addr[num_lans];
    dram_addr_valid_o = (send_req_curr_state == SEND_REQ_S) || 
                        (write_dram_curr_state == SEND_REQ_S);
    //b000 for read command, 3'b001 for write command
    rw_o = (send_req_curr_state == SEND_REQ_S) ? 0
                                               : 1;
end

always_comb begin
    /*
     * bits 255:0
     */
    for (int i = 0; i < 16; i+=2) begin
        {data_in[i+1], data_in[i]} = dram_data_read[(DATA_WIDTH*(16-i)-1)-:(DATA_WIDTH*2)];
    end
    /*
     * bits 511:256
     */
    for (int i = 16; i < 32; i+=2) begin
        {data_in[i+1], data_in[i]} = dram_data_read[(DATA_WIDTH*(32-(i-16))-1)-:(DATA_WIDTH*2)];
    end
    
end

always_comb begin
    addr_offset = 64 - addr[5:0];
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        addr <= 0;
        rw_r <= 1;
        byte_mask_r <= 64'hFFFF_FFFF_FFFF_FFFF;
        // need additional reset
        req_len_cnt <= 0;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == DRAM_R_ADDR) begin
        addr <= S_DEVICE_data_i;

        read_addr_base <= S_DEVICE_data_i;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == DRAM_R_LENGTH) begin
        length <= S_DEVICE_data_i;
    end
    else if(send_req_curr_state == SEND_REQ_S) begin
        addr <= addr + addr_offset;
        rw_r <= 1;
        byte_mask_r <= 64'hFFFF_FFFF_FFFF_FFFF;
        req_len_cnt += addr_offset >> 1;
    end
    else if(send_req_curr_state == IDLE_S) begin
        req_len_cnt <= 0;
    end
    else if(send_req_curr_state == READ_NEXT_ROUND_S) begin
        addr           <= read_addr_base + read_offset;
        read_addr_base <= read_addr_base + read_offset;
        req_len_cnt <= 0;
    end

end

assign dram_data_read = {dram_read_data_h_r, dram_read_data_l_r};


/*
 * FSM (WRITE DATA)
 */
always_ff @(posedge clk_i) begin
    if(rst_i) begin
        write_data_curr_state <= IDLE_S;
    end
    else begin
        write_data_curr_state <= write_data_next_state;
    end
end

always_comb begin
    case (write_data_curr_state)
    IDLE_S: if(S_DEVICE_strobe_i && S_DEVICE_addr_i == DRAM_TR) 
                write_data_next_state = WAIT_FIFO_DATA_S;
            else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == RESET_SRAM) 
                write_data_next_state = RESET_SRAM_S;
            else
                write_data_next_state = IDLE_S;
    WAIT_FIFO_DATA_S: if(!fifo_data_empty_i && rw_to_gemm) 
                        write_data_next_state = WRITE_INPUT_S;
                      else if(!fifo_data_empty_i && !rw_to_gemm) 
                        write_data_next_state = DUMMY_1_S;
                      else 
                        write_data_next_state = WAIT_FIFO_DATA_S;
    WRITE_INPUT_S: if(send_cnt == length - 1)
                        write_data_next_state = READ_NEXT_DATA_S;
                   else if(got_addr[5:0] == 6'b111110)
                        write_data_next_state = READ_NEXT_DATA_S;
                   else write_data_next_state = WRITE_INPUT_S;
    READ_NEXT_DATA_S: if(send_cnt == length)
                            write_data_next_state = RECV_NEXT_ROUND_S;
                      else
                            write_data_next_state = WAIT_FIFO_DATA_S;
    DUMMY_1_S: write_data_next_state = IDLE_S;
    RECV_NEXT_ROUND_S:  if(recv_rounds_cnt + 1 == read_rounds) 
                            write_data_next_state = IDLE_S;
                        else write_data_next_state = WAIT_FIFO_DATA_S;
    RESET_SRAM_S:   if(sram_offset == 32767)
                        write_data_next_state = IDLE_S;
                    else    
                        write_data_next_state = RESET_SRAM_S;
    default: write_data_next_state = IDLE_S;
    endcase
end

always_comb begin
    fifo_data_rd_en_o = (write_data_curr_state == READ_NEXT_DATA_S) ||
                        (write_data_curr_state == DUMMY_1_S);
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        got_addr <= 0;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == DRAM_R_ADDR) begin
        got_addr <= S_DEVICE_data_i;

        recv_addr_base <= S_DEVICE_data_i;
    end
    else if(write_data_curr_state == WRITE_INPUT_S) begin
        got_addr <= got_addr + 2;
    end
    else if(write_data_curr_state == RECV_NEXT_ROUND_S) begin
        got_addr       <= recv_addr_base + read_offset;
        recv_addr_base <= recv_addr_base + read_offset;
    end
end

/*
 * SW READ TEST SIGNAL
 * sw_data_r
 */
always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        sw_data_r <= 0;
    end
    else if(write_data_curr_state == DUMMY_1_S) begin
        sw_data_r <= data_in[got_addr[5:1]];
    end
end


always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        send_cnt <= 0;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == SRAM_OFFSET) begin
        sram_offset <= S_DEVICE_data_i;
    end
    else if(write_data_curr_state == WRITE_INPUT_S) begin
        send_cnt <= send_cnt + 1;
        if(sramWriteCount   + 1 == inputLength && 
           inputHeightCount + 1 == inputHeight && 
           write_data_type == 0) begin
            sram_offset <= sram_offset + boundaryPaddingSize + 1;
            // if     (currInputType == 0) begin
            //     sram_offset <= sram_offset + paddingSize*4 + inputLength;
            // end
            // else if(currInputType == 1) begin
            //     sram_offset <= sram_offset + paddingSize*2;
            // end
            // else if(currInputType == 2) begin
            //     sram_offset <= sram_offset + paddingSize*4 + inputLength;
            // end
            // else if(currInputType == 3) begin
            //     sram_offset <= sram_offset + paddingSize*6 + inputLength*2;
            // end
        end
        else if(sramWriteCount + 1 == inputLength && 
                write_data_type == 0) begin
            sram_offset <= sram_offset + paddingSize*2 + 1;
        end
        else begin
            sram_offset <= sram_offset + 1;
        end
    end
    else if(write_data_curr_state == IDLE_S) begin
        send_cnt <= 0;
    end
    else if(write_data_curr_state == RECV_NEXT_ROUND_S) begin
        send_cnt <= 0;
    end
    else if(write_data_curr_state == RESET_SRAM_S) begin
        sram_offset <= sram_offset + 1;
    end
end

/*
 *  type 0: top
 *  *********
 *  *       *
 *  type 1: mid
 *  *       *
 *  *       *
 *  type 2: bottom
 *  *       *
 *  *********
 *  type 3: all
 *  *********
 *  *       *
 *  *********
 *
 */
always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        currInputType <= 0;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == CURRINPUTTYPE) begin
        currInputType <= S_DEVICE_data_i;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        inputLength <= 0;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == INPUTLENGTH) begin
        inputLength <= S_DEVICE_data_i;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        paddingSize <= 0;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == PADDINGSIZE) begin
        paddingSize <= S_DEVICE_data_i;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        boundaryPaddingSize <= 0;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == BOUNDARYPADDINGSIZE) begin
        boundaryPaddingSize <= S_DEVICE_data_i;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        inputHeight <= 0;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == INPUTHEIGHT) begin
        inputHeight <= S_DEVICE_data_i;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        sramWriteCount <= 0;
    end
    else if(write_data_curr_state == WRITE_INPUT_S) begin
        if(sramWriteCount + 1 == inputLength) begin
            sramWriteCount <= 0;
        end
        else begin
            sramWriteCount <= sramWriteCount + 1;
        end
    end
    else if(write_data_curr_state == IDLE_S) begin
        sramWriteCount <= 0;
    end
    else if(write_data_curr_state == RECV_NEXT_ROUND_S) begin
        sramWriteCount <= 0;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        inputHeightCount <= 0;
    end
    else if(sramWriteCount + 1 == inputLength && write_data_curr_state ==  WRITE_INPUT_S) begin
        inputHeightCount <= inputHeightCount + 1;
    end
    else if(write_data_curr_state == IDLE_S) begin
        inputHeightCount <= 0;
    end
    else if(write_data_curr_state == RECV_NEXT_ROUND_S) begin
        inputHeightCount <= 0;
    end
   
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        dram_read_data_h_r <= 0;
    end
    else if(!fifo_data_empty_i) begin
        dram_read_data_h_r <= dram_read_data_h_i;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        dram_read_data_l_r <= 0;
    end
    else if(!fifo_data_empty_i) begin
        dram_read_data_l_r <= dram_read_data_l_i;
    end
end



/*
 * DRAM WRITE 
 * 1. collect data from output buffer
 * 2. write data to dram 
 */
always_ff @(posedge clk_i) begin
    if(rst_i) begin
        write_dram_curr_state <= IDLE_S;
    end
    else begin
        write_dram_curr_state <= write_dram_next_state;
    end
end

always_comb begin
    case (write_dram_curr_state)
    IDLE_S: if(S_DEVICE_strobe_i && S_DEVICE_addr_i == TR_DRAM_W)
                write_dram_next_state = COLLECT_OUTPUT_S;
            else 
                write_dram_next_state = IDLE_S;
    COLLECT_OUTPUT_S: write_dram_next_state = WAIT_GEMM_DATA_S;
    WAIT_GEMM_DATA_S: if(( |ret_valid ) && (dram_addr_offset*4 + 4) >= dram_write_length)
                        write_dram_next_state = WAIT_FIFO_ADDR_S;
                      else if(( |ret_valid ))
                        write_dram_next_state = COLLECT_OUTPUT_S;
                      else 
                        write_dram_next_state = WAIT_GEMM_DATA_S;
    WAIT_FIFO_ADDR_S: if(!fifo_addr_full_i         && 
                         !dram_rw_full_i           &&
                         !dram_write_data_h_full_i && 
                         !dram_write_data_l_full_i &&
                         !dram_str_idx_full_i      && 
                         !dram_end_idx_full_i) 
                        write_dram_next_state = SEND_REQ_S;
                      else
                        write_dram_next_state = WAIT_FIFO_ADDR_S;
    // SEND_REQ_S: write_dram_next_state = WAIT_WRITE_DONE_S;
    SEND_REQ_S: if(dram_write_length_cnt + (dram_write_addr_offset >> 1) >= dram_write_length)
                    write_dram_next_state = IDLE_S;
                else
                    write_dram_next_state = WAIT_FIFO_ADDR_S;
    // WAIT_WRITE_DONE_S:  if(!dram_write_done_fifo_empty_i && 
    //                         dram_write_length_cnt >= dram_write_length)
    //                             write_dram_next_state = IDLE_S;
    //                     else if(!dram_write_done_fifo_empty_i)
    //                             write_dram_next_state = WAIT_FIFO_ADDR_S;
    //                    else write_dram_next_state = WAIT_WRITE_DONE_S;
    default: write_dram_next_state = IDLE_S;
    endcase
end

// write len may less than 32

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        output_recv_cnt <= 0;
    end
    else if(write_dram_curr_state == WAIT_GEMM_DATA_S && ( |ret_valid )) begin
        output_recv_cnt <= output_recv_cnt + 1;
    end
    else if (S_DEVICE_strobe_i && S_DEVICE_addr_i == OUTPUT_RECV_CNT_ADDR ) begin
        output_recv_cnt <= S_DEVICE_data_i;
    end
end

// dram_addr_offset
always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        dram_addr_offset <= 0;
    end
    else if(write_dram_curr_state == IDLE_S) begin
        dram_addr_offset <= 0;
    end
    else if(write_dram_curr_state == WAIT_GEMM_DATA_S && ( |ret_valid )) begin
        dram_addr_offset <= dram_addr_offset + 1;
    end
end

always_comb begin
    dram_write_addr_offset = 64 - dram_write_addr[num_lans][5:0];
end

always_comb begin
    data_start_idx_o = dram_write_addr[num_lans][5:1];

    if((dram_write_length - dram_write_length_cnt) < (dram_write_addr_offset >> 1)) begin
        data_end_idx_o = data_start_idx_o + (dram_write_length - dram_write_length_cnt - 1);
    end
    else begin
        data_end_idx_o = 31;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        for(int i = 0; i < 4; i++) begin
            dram_write_addr[i] <= 0;
        end
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == DRAM_WRITE_ADDR[0]) begin
        dram_write_addr[0] <= S_DEVICE_data_i;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == DRAM_WRITE_ADDR[1]) begin
        dram_write_addr[1] <= S_DEVICE_data_i;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == DRAM_WRITE_ADDR[2]) begin
        dram_write_addr[2] <= S_DEVICE_data_i;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == DRAM_WRITE_ADDR[3]) begin
        dram_write_addr[3] <= S_DEVICE_data_i;
    end
    else if (write_dram_curr_state == SEND_REQ_S) begin
        dram_write_addr[num_lans] <= dram_write_addr[num_lans] + dram_write_addr_offset; 
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        num_lans <= 0;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == NUM_LANS_ADDR) begin
        num_lans <= S_DEVICE_data_i;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        dram_write_length <= 0;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == DRAM_WRITE_LEN) begin
        dram_write_length <= S_DEVICE_data_i;
    end
end

// ERROR
//dram_write_length_cnt NEED RESET

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        // send_lans_cnt <= 0;
        dram_write_length_cnt <= 0;
    end
    else if(write_dram_curr_state == IDLE_S) begin
        // send_lans_cnt <= 0;
        dram_write_length_cnt <= 0;
    end
    else if (write_dram_curr_state == SEND_REQ_S) begin   
        // if(dram_write_length_cnt + (dram_write_addr_offset >> 1) >= dram_write_length) begin
        //     dram_write_length_cnt <= 0;
        //     // send_lans_cnt <= send_lans_cnt + 1;
        // end
        // else begin
            // ??
            dram_write_length_cnt <= dram_write_length_cnt + 
                                 (dram_write_addr_offset >> 1);
        // end
        
    end
end

always_comb begin
    if     (gemm_core_sel[0]) GeMMOutputSel = 0;
    else if(gemm_core_sel[1]) GeMMOutputSel = 1;
    else if(gemm_core_sel[2]) GeMMOutputSel = 2;
    else if(gemm_core_sel[3]) GeMMOutputSel = 3;
    else                      GeMMOutputSel = 0;
end


always_ff @( posedge clk_i ) begin
    // for(int i = 0; i < 4; i++) begin
        for(int j = 0; j < 4; j++) begin
            if(write_dram_curr_state == WAIT_GEMM_DATA_S && ( |ret_valid )) begin
                // dram_data_r[i][output_recv_cnt*4 + j] <= P_data_out[i][(DATA_WIDTH*(4-j)-1)-:DATA_WIDTH];
                if     (num_lans == 0) begin
                    dram_data_r[dram_addr_offset*4 + j] <= P_data_out_1[GeMMOutputSel][(DATA_WIDTH*(4-j)-1)-:DATA_WIDTH];
                end
                else if(num_lans == 1) begin
                    dram_data_r[dram_addr_offset*4 + j] <= P_data_out_2[GeMMOutputSel][(DATA_WIDTH*(4-j)-1)-:DATA_WIDTH];
                end
                else if(num_lans == 2) begin
                    dram_data_r[dram_addr_offset*4 + j] <= P_data_out_3[GeMMOutputSel][(DATA_WIDTH*(4-j)-1)-:DATA_WIDTH];
                end
                else if(num_lans == 3) begin
                    dram_data_r[dram_addr_offset*4 + j] <= P_data_out_4[GeMMOutputSel][(DATA_WIDTH*(4-j)-1)-:DATA_WIDTH];
                end
                // if(gemm_core_sel[0]) begin
                //     dram_data_r[dram_addr_offset*4 + j] <= P_data_out_1[num_lans][(DATA_WIDTH*(4-j)-1)-:DATA_WIDTH];
                // end
                // else if(gemm_core_sel[1]) begin
                //     dram_data_r[dram_addr_offset*4 + j] <= P_data_out_2[num_lans][(DATA_WIDTH*(4-j)-1)-:DATA_WIDTH];
                // end
                // else if(gemm_core_sel[2]) begin
                //     dram_data_r[dram_addr_offset*4 + j] <= P_data_out_3[num_lans][(DATA_WIDTH*(4-j)-1)-:DATA_WIDTH];
                // end
                // else if(gemm_core_sel[3]) begin
                //     dram_data_r[dram_addr_offset*4 + j] <= P_data_out_4[num_lans][(DATA_WIDTH*(4-j)-1)-:DATA_WIDTH];
                // end
            end
        end
    // end
end

logic [4:0] v = dram_write_addr[num_lans][5:1];


always_ff @( posedge clk_i ) begin
    for(int i = 0; i < 32; i++) begin
        if(sw_write_dram_mode) begin
            dram_data_reorder_r[i] <= tpu_param_1_in;
        end
        else begin
            dram_data_reorder_r[v + i] <= dram_data_r[dram_write_length_cnt + i];
        end
    end
end
always_comb begin
    dram_data_o = {dram_data_reorder_r[17], dram_data_reorder_r[16],
                   dram_data_reorder_r[19], dram_data_reorder_r[18],
                   dram_data_reorder_r[21], dram_data_reorder_r[20],
                   dram_data_reorder_r[23], dram_data_reorder_r[22],
                   dram_data_reorder_r[25], dram_data_reorder_r[24],
                   dram_data_reorder_r[27], dram_data_reorder_r[26],
                   dram_data_reorder_r[29], dram_data_reorder_r[28],
                   dram_data_reorder_r[31], dram_data_reorder_r[30],
        
                   dram_data_reorder_r[ 1], dram_data_reorder_r[ 0],
                   dram_data_reorder_r[ 3], dram_data_reorder_r[ 2],
                   dram_data_reorder_r[ 5], dram_data_reorder_r[ 4],
                   dram_data_reorder_r[ 7], dram_data_reorder_r[ 6],
                   dram_data_reorder_r[ 9], dram_data_reorder_r[ 8],
                   dram_data_reorder_r[11], dram_data_reorder_r[10],
                   dram_data_reorder_r[13], dram_data_reorder_r[12],
                   dram_data_reorder_r[15], dram_data_reorder_r[14]};
end

endmodule
