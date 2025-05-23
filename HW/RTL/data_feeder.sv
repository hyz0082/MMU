module data_feeder 
#( parameter XLEN = 32, 
   parameter BUF_ADDR_LEN = 32, 
   parameter ACLEN  = 8,
   parameter ADDR_BITS=15,
   parameter DATA_WIDTH = 16)
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
(* mark_debug="true" *)     input  logic              fifo_addr_full_i,
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

localparam TPU_CMD_ADDR  = 32'hC4000000;
localparam PARAM_1_ADDR  = 32'hC4000004;
localparam PARAM_2_ADDR  = 32'hC4000008;
localparam BUSY_ADDR     = 32'hC4000020;
localparam RET_ADDR      = 32'hC4000010;
localparam RET_MAX_POOLING_ADDR      = 32'hC4002010;
localparam RET_SOFTMAX_ADDR          = 32'hC4002020;

localparam DRAM_R_ADDR   = 32'hC4002024;
localparam DRAM_R_LENGTH = 32'hC4002028;
localparam DRAM_RW          = 32'hC400202C;
localparam DRAM_TR   = 32'hC4002030;
localparam SRAM_OFFSET   = 32'hC4002034;
localparam WRITE_DATA_TYPE_ADDR = 32'hC4002038;
localparam [31:0] DRAM_WRITE_ADDR [0:3] = {32'hC400203C,
                                           32'hC4002040,
                                           32'hC4002044,
                                           32'hC4002048};
localparam NUM_LANS_ADDR     = 32'hC400204C;
localparam DRAM_WRITE_LEN    = 32'hC4002050;
localparam TR_DRAM_W         = 32'hC4002054;
localparam OUTPUT_RECV_CNT_ADDR = 32'hC4002058;
localparam SW_DATA_ADDR =  32'hC400205C;
localparam SW_WRITE_DRAM_MODE_ADDR = 32'hC4002060;
localparam RET_AVG_POOLING_ADDR    = 32'hC4002064;


localparam [31:0] TPU_DATA_ADDR [0:15] = {32'hC4001000, 32'hC4001100, 32'hC4001200, 32'hC4001300,
                                          32'hC4001400, 32'hC4001500, 32'hC4001600, 32'hC4001700,
                                          32'hC4001800, 32'hC4001900, 32'hC4001A00, 32'hC4001B00,
                                          32'hC4001C00, 32'hC4001D00, 32'hC4001E00, 32'hC4001F00};

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
              WAIT_WRITE_DONE_S
              } state_t;
state_t send_req_curr_state;
state_t send_req_next_state;
state_t write_data_curr_state;
state_t write_data_next_state;
state_t write_dram_curr_state;
state_t write_dram_next_state;

// 0xC4000000
(* mark_debug="true" *) logic                        tpu_cmd_valid;     // tpu valid
(* mark_debug="true" *) logic   [ACLEN-1 : 0]        tpu_cmd;
logic  [DATA_WIDTH-1 : 0] S_DEVICE_data_i_t;
assign S_DEVICE_data_i_t = S_DEVICE_data_i;
// 0xC4000004
(* mark_debug="true" *) logic   [DATA_WIDTH-1 : 0]   tpu_param_1_in;    // data 1
// 0xC4000008
(* mark_debug="true" *) logic   [DATA_WIDTH-1 : 0]   tpu_param_2_in;     // data 2

logic                      ret_valid;
logic   [DATA_WIDTH-1 : 0] ret_data_out;
logic   [DATA_WIDTH-1 : 0] ret_max_pooling;
logic   [DATA_WIDTH-1 : 0] ret_avg_pooling;
logic   [DATA_WIDTH-1 : 0] ret_softmax_result;
// 0xC400000A
(* mark_debug="true" *) logic                      tpu_busy;     // 0->idle, 1->busy
// 0xC4000010
logic   [DATA_WIDTH-1 : 0] ret_data_out_reg;

logic   [DATA_WIDTH-1 : 0] tpu_data_reg [0 : 15];

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


logic [6 : 0] dram_write_addr_offset;

// (* mark_debug="true" *) logic           dram_read_data_vaild_h_r;
logic [255 : 0] dram_read_data_h_r;
// (* mark_debug="true" *) logic           dram_read_data_vaild_l_r;
logic [255 : 0] dram_read_data_l_r;

logic write_data_type; // 0: input, 1: weight

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
logic [ADDR_BITS-1 : 0] sram_offset;
logic [3 : 0] data_type;
logic [6 : 0] addr_offset;


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
logic [DATA_WIDTH*4-1 : 0] P_data_out [0 : 3];

/*
 * SOFTWARE READ / WRITE DATA
 *  
 */
logic [DATA_WIDTH-1 : 0] sw_data_r;
logic [DATA_WIDTH-1 : 0] sw_write_data_r;
logic sw_write_dram_mode;
logic rw_to_gemm;

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
    if(ret_valid) begin
        ret_data_out_reg <= ret_data_out;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        S_DEVICE_data_o  <= 0;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == BUSY_ADDR) begin
        S_DEVICE_data_o  <= tpu_busy || 
                            (send_req_curr_state   != IDLE_S) ||
                            (write_data_curr_state != IDLE_S) ||
                            (write_dram_curr_state != IDLE_S);
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == RET_ADDR) begin
        S_DEVICE_data_o  <= {16'h0, ret_data_out_reg};
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == RET_MAX_POOLING_ADDR) begin
        S_DEVICE_data_o  <= ret_max_pooling;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == RET_AVG_POOLING_ADDR) begin
        S_DEVICE_data_o  <= ret_avg_pooling;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == RET_SOFTMAX_ADDR) begin
        S_DEVICE_data_o  <= ret_softmax_result;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == SW_DATA_ADDR) begin
        S_DEVICE_data_o  <= sw_data_r;
    end
end

always_ff @( posedge clk_i ) begin
    for(int i = 0; i < 16; i++) begin
        if(rst_i) begin
            tpu_data_reg[i] <= 0;
        end
        else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == TPU_DATA_ADDR[i]) begin
            tpu_data_reg[i]  <= S_DEVICE_data_i;
        end
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

always_comb begin
    tpu_data_1_in = {tpu_data_reg[0], tpu_data_reg[4], tpu_data_reg[8] , tpu_data_reg[12]};
    tpu_data_2_in = {tpu_data_reg[1], tpu_data_reg[5], tpu_data_reg[9] , tpu_data_reg[13]};
    tpu_data_3_in = {tpu_data_reg[2], tpu_data_reg[6], tpu_data_reg[10], tpu_data_reg[14]};
    tpu_data_4_in = {tpu_data_reg[3], tpu_data_reg[7], tpu_data_reg[11], tpu_data_reg[15]};
    
end

TPU #(
    .ACLEN(ACLEN),
    .ADDR_BITS(ADDR_BITS),
    .DATA_WIDTH(DATA_WIDTH)
) 
t1 (
    .clk_i(clk_i), .rst_i(rst_i),
    .tpu_cmd_valid(tpu_cmd_valid),     // tpu valid
    .tpu_cmd(tpu_cmd),           // tpu
    .tpu_param_1_in(tpu_param_1_in),    // data 1
    .tpu_param_2_in(tpu_param_2_in),     // data 2
    
    .tpu_data_1_in(tpu_data_1_in),
    .tpu_data_2_in(tpu_data_2_in),
    .tpu_data_3_in(tpu_data_3_in),
    .tpu_data_4_in(tpu_data_4_in),

    .tpu_data_1_out(P_data_out[0]),
    .tpu_data_2_out(P_data_out[1]),
    .tpu_data_3_out(P_data_out[2]),
    .tpu_data_4_out(P_data_out[3]),
    
    .ret_valid(ret_valid),
    .ret_data_out(ret_data_out),
    .ret_max_pooling(ret_max_pooling),
    .ret_avg_pooling(ret_avg_pooling),
    .ret_softmax_result(ret_softmax_result),
    .tpu_busy(tpu_busy)     
);

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
        // dram_data <= 0;
        byte_mask_r <= 64'hFFFF_FFFF_FFFF_FFFF;
        // need additional reset
        req_len_cnt <= 0;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == DRAM_R_ADDR) begin
        addr <= S_DEVICE_data_i;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == DRAM_R_LENGTH) begin
        length <= S_DEVICE_data_i;
    end
    else if(send_req_curr_state == SEND_REQ_S) begin
        addr <= addr + addr_offset;
        rw_r <= 1;
        // dram_data <= 0;
        byte_mask_r <= 64'hFFFF_FFFF_FFFF_FFFF;
        req_len_cnt += addr_offset >> 1;
    end
    else if(send_req_curr_state == IDLE_S) begin
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
                            write_data_next_state = IDLE_S;
                      else
                            write_data_next_state = WAIT_FIFO_DATA_S;
    DUMMY_1_S: write_data_next_state = IDLE_S;
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
    end
    else if(write_data_curr_state == WRITE_INPUT_S) begin
        got_addr <= got_addr + 2;
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
        sram_offset <= 0;
    end
    else if(write_data_curr_state == WRITE_INPUT_S) begin
        send_cnt <= send_cnt + 1;
        sram_offset <= sram_offset + 1;
    end
    else if(write_data_curr_state == IDLE_S) begin
        send_cnt <= 0;
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
    WAIT_GEMM_DATA_S: if(ret_valid && (dram_addr_offset*4 + 4) >= dram_write_length)
                        write_dram_next_state = WAIT_FIFO_ADDR_S;
                      else if(ret_valid)
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
    else if(write_dram_curr_state == WAIT_GEMM_DATA_S && ret_valid) begin
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
    else if(write_dram_curr_state == WAIT_GEMM_DATA_S && ret_valid) begin
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

always_ff @( posedge clk_i ) begin
    // for(int i = 0; i < 4; i++) begin
        for(int j = 0; j < 4; j++) begin
            if(write_dram_curr_state == WAIT_GEMM_DATA_S && ret_valid) begin
                // dram_data_r[i][output_recv_cnt*4 + j] <= P_data_out[i][(DATA_WIDTH*(4-j)-1)-:DATA_WIDTH];
                dram_data_r[dram_addr_offset*4 + j] <= P_data_out[num_lans][(DATA_WIDTH*(4-j)-1)-:DATA_WIDTH];
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
