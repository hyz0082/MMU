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
(* mark_debug="true" *)         input  logic              fifo_addr_full_i,
(* mark_debug="true" *)     output logic              dram_addr_valid_o,
    output logic [XLEN-1 : 0] dram_addr_o,

(* mark_debug="true" *)     input  logic          fifo_data_empty_i,
    output logic          fifo_data_rd_en_o,
    // input logic          dram_read_data_vaild_h_i,
    input logic[255 : 0] dram_read_data_h_i,
    // input logic          dram_read_data_vaild_l_i,
    input logic[255 : 0] dram_read_data_l_i
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

localparam [31:0] TPU_DATA_ADDR [0:15] = {32'hC4001000, 32'hC4001100, 32'hC4001200, 32'hC4001300,
                                          32'hC4001400, 32'hC4001500, 32'hC4001600, 32'hC4001700,
                                          32'hC4001800, 32'hC4001900, 32'hC4001A00, 32'hC4001B00,
                                          32'hC4001C00, 32'hC4001D00, 32'hC4001E00, 32'hC4001F00};

/*
    DRAM ACCESS FSM
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
              WRITE_S 
              } state_t;
(* mark_debug="true" *)state_t send_req_curr_state, send_req_next_state;
(* mark_debug="true" *)state_t write_data_curr_state, write_data_next_state;

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
logic   [DATA_WIDTH-1 : 0] ret_softmax_result;
// 0xC400000A
logic                      tpu_busy;     // 0->idle, 1->busy
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
(* mark_debug="true" *) logic [XLEN-1 : 0] addr; // + 64 each times (512 bits)
logic rw_r;  // 3'b000 for write command, 3'b001 for read command
logic [511 : 0] dram_data_read;
logic [511 : 0] dram_data_write;
logic [63:0] byte_mask_r;

(* mark_debug="true" *) logic [XLEN-1 : 0] got_addr;
logic [6 : 0] got_addr_offset;

// (* mark_debug="true" *) logic           dram_read_data_vaild_h_r;
(* mark_debug="true" *) logic [255 : 0] dram_read_data_h_r;
// (* mark_debug="true" *) logic           dram_read_data_vaild_l_r;
(* mark_debug="true" *) logic [255 : 0] dram_read_data_l_r;

/*
    get 512 bits data per dram read
    512 / 16 = 32 inputs
*/
logic [DATA_WIDTH-1 : 0] data_in [0 : 31];
(* mark_debug="true" *) logic [ADDR_BITS-1 : 0] req_len_cnt;
logic [5 : 0]           req_len; // max req len = 32
logic [ADDR_BITS-1 : 0] got_len_cnt;
logic [5 : 0]           got_len_expect; // max req len = 32
logic [ADDR_BITS-1 : 0] send_cnt;
(* mark_debug="true" *) logic [ADDR_BITS-1 : 0] length;
logic [3 : 0] word_size;
logic [ADDR_BITS-1 : 0] sram_offset;
logic [3 : 0] data_type;
logic [6 : 0] addr_offset;



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
                            (write_data_curr_state != IDLE_S);
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == RET_ADDR) begin
        S_DEVICE_data_o  <= {16'h0, ret_data_out_reg};
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == RET_MAX_POOLING_ADDR) begin
        S_DEVICE_data_o  <= ret_max_pooling;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == RET_SOFTMAX_ADDR) begin
        S_DEVICE_data_o  <= ret_softmax_result;
    end
    
    // RET_MAX_POOLING_ADDR
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
    else if(write_data_curr_state == WRITE_INPUT_S) begin
        tpu_cmd_valid  <= 1;
        tpu_cmd        <= 9;
        tpu_param_1_in <= data_in[got_addr[5:1]];// data_in
        tpu_param_2_in <= sram_offset;// index
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
    
    .ret_valid(ret_valid),
    .ret_data_out(ret_data_out),
    .ret_max_pooling(ret_max_pooling),
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
    WAIT_FIFO_ADDR_S: if(!fifo_addr_full_i) 
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
    dram_addr_o = addr;
    dram_addr_valid_o = (send_req_curr_state == SEND_REQ_S);
end

always_comb begin
    // for (int i = 0; i < 32; i++) begin
    //     data_in[i] = dram_data_read[(DATA_WIDTH*(i+1)-1)-:DATA_WIDTH];
    // end
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
        // {data_in[i+1], data_in[i]} = dram_read_data_h_r
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
    WAIT_FIFO_DATA_S: if(!fifo_data_empty_i) 
                        write_data_next_state = WRITE_INPUT_S;
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
    default: write_data_next_state = IDLE_S;
    endcase
end

always_comb begin
    fifo_data_rd_en_o = (write_data_curr_state == READ_NEXT_DATA_S);
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

/*
    cdc response
*/
// always_ff @( posedge clk_i ) begin
//     if(rst_i) begin
//         dram_read_data_vaild_h_r <= 0;
//     end
//     else if(!fifo_data_empty_i) begin
//         dram_read_data_vaild_h_r <= 1;
//     end
//     else if(curr_state == WRITE_INPUT_S) begin
//         dram_read_data_vaild_h_r <= 0;
//     end
// end

// always_ff @( posedge clk_i ) begin
//     if(rst_i) begin
//         dram_read_data_vaild_l_r <= 0;
//     end
//     else if(dram_read_data_vaild_l_i) begin
//         dram_read_data_vaild_l_r <= 1;
//     end
//     else if(curr_state == WRITE_INPUT_S) begin
//         dram_read_data_vaild_l_r <= 0;
//     end
// end

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

endmodule