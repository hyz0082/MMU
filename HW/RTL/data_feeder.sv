module data_feeder 
#( parameter XLEN = 32, 
   parameter BUF_ADDR_LEN = 32, 
   parameter ACLEN  = 8,
   parameter ADDR_BITS=15,
   parameter DATA_WIDTH = 32)
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
    output logic [XLEN-1 : 0]             S_DEVICE_data_o
);

localparam TPU_CMD_ADDR  = 32'hC4000000;
localparam PARAM_1_ADDR  = 32'hC4000004;
localparam PARAM_2_ADDR  = 32'hC4000008;
localparam BUSY_ADDR     = 32'hC4000020;
localparam RET_ADDR      = 32'hC4000010;
localparam RET_MAX_POOLING_ADDR      = 32'hC4002010;
localparam RET_SOFTMAX_ADDR          = 32'hC4002020;

localparam [31:0] TPU_DATA_ADDR [0:15] = {32'hC4001000, 32'hC4001100, 32'hC4001200, 32'hC4001300,
                                          32'hC4001400, 32'hC4001500, 32'hC4001600, 32'hC4001700,
                                          32'hC4001800, 32'hC4001900, 32'hC4001A00, 32'hC4001B00,
                                          32'hC4001C00, 32'hC4001D00, 32'hC4001E00, 32'hC4001F00};

// 0xC4000000
(* mark_debug="true" *)logic                        tpu_cmd_valid;     // tpu valid
(* mark_debug="true" *)logic   [ACLEN-1 : 0]        tpu_cmd;
(* mark_debug="true" *)logic  [DATA_WIDTH-1 : 0] S_DEVICE_data_i_t;
assign S_DEVICE_data_i_t = S_DEVICE_data_i;
// 0xC4000004
(* mark_debug="true" *)logic   [DATA_WIDTH-1 : 0]   tpu_param_1_in;    // data 1
// 0xC4000008
(* mark_debug="true" *)logic   [DATA_WIDTH-1 : 0]   tpu_param_2_in;     // data 2

(* mark_debug="true" *)logic                      ret_valid;
(* mark_debug="true" *)logic   [DATA_WIDTH-1 : 0] ret_data_out;
(* mark_debug="true" *)logic   [DATA_WIDTH-1 : 0] ret_max_pooling;
(* mark_debug="true" *)logic   [DATA_WIDTH-1 : 0] ret_softmax_result;
// 0xC400000A
(* mark_debug="true" *) logic                      tpu_busy;     // 0->idle, 1->busy
// 0xC4000010
logic   [DATA_WIDTH-1 : 0] ret_data_out_reg;

logic   [DATA_WIDTH-1 : 0] tpu_data_reg [0 : 15];

logic   [DATA_WIDTH*4-1 : 0]   tpu_data_1_in;
logic   [DATA_WIDTH*4-1 : 0]   tpu_data_2_in;
logic   [DATA_WIDTH*4-1 : 0]   tpu_data_3_in;
logic   [DATA_WIDTH*4-1 : 0]   tpu_data_4_in;

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
        S_DEVICE_data_o  <= tpu_busy;
    end
    else if(S_DEVICE_strobe_i && S_DEVICE_addr_i == RET_ADDR) begin
        S_DEVICE_data_o  <= ret_data_out_reg;
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

endmodule