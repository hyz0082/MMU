`timescale 1ns / 1ps
// =============================================================================
//  Program : PE.v
//  Author  : 
//  Date    : 
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// =============================================================================
`include "config.vh"

module PE
#(parameter ACLEN  = 4,
  parameter DATA_WIDTH = 32
//   parameter CLSIZE = `CLP
)
(
    /////////// System signals   ///////////////////////////////////////////////
    input                         clk_i, rst_i,
    ///////////   PE command    ///////////////////////////////////////////////
    input                         pe_cmd_valid, // pe_cmd valid
    input      [ACLEN : 0]        pe_cmd,     // pe_cmd
    input      [DATA_WIDTH-1 : 0] param_1_in, // act ctrl
    input      [DATA_WIDTH-1 : 0] param_2_in, // data

    /////////// PE input   ///////////////////////////////////////////////
    input      [DATA_WIDTH-1 : 0] data_in,
    input      [DATA_WIDTH-1 : 0] weight_in,
    /////////// PE outpupt ///////////////////////////////////////////////
    output reg [DATA_WIDTH-1 : 0] data_out, 
    output reg [DATA_WIDTH-1 : 0] weight_out,
    output     [DATA_WIDTH-1 : 0] mac_value,

    output reg                    busy     // 0 for idle, 1 for busy
);


// pe_cmd table
parameter  RESET             = 0;
parameter  TRIGGER           = 1;
parameter  SET_MUL_VAL       = 2;
parameter  SET_ADD_VAL       = 3;
parameter  LOAD_DATA         = 4;
parameter  SET_CONV_MODE     = 5;
parameter  SET_FIX_MAC_MODE  = 6;
parameter  IDLE              = 7;
// parameter  NEW_ROUND  = 5;

// mode
parameter  CONV_MODE          = 0;
parameter  FIX_MAC_MODE       = 1;

reg mode; // 0 for normal mac, 1 for fix value mac

reg [DATA_WIDTH-1 : 0] mac_reg;
reg [DATA_WIDTH-1 : 0] mul_val_reg;
reg [DATA_WIDTH-1 : 0] add_val_reg;

wire [DATA_WIDTH-1 : 0] a_data, b_data, c_data;
wire                    t_valid;
wire [DATA_WIDTH-1 : 0] r_data;
wire                    r_valid;

always@(posedge clk_i) begin
    if(rst_i) begin
        weight_out <= 0;
        data_out   <= 0;
    end
    else if(pe_cmd_valid && pe_cmd == TRIGGER)begin
        weight_out <= weight_in;
        data_out   <= data_in;
    end
end

// mul_val and add_val
always@(posedge clk_i) begin
    if(rst_i) begin
        mul_val_reg <= 0;
        add_val_reg <= 0;
    end
    else if(pe_cmd_valid && pe_cmd == SET_MUL_VAL) begin
        mul_val_reg <= param_2_in;
    end
    else if(pe_cmd_valid && pe_cmd == SET_ADD_VAL) begin
        add_val_reg <= param_2_in;
    end
end

// 
always@(posedge clk_i) begin
    if(rst_i) begin
        mode <= 0;
    end
    else if(pe_cmd_valid && pe_cmd == SET_CONV_MODE) begin
        mode <= CONV_MODE;
    end
    else if(pe_cmd_valid && pe_cmd == SET_FIX_MAC_MODE) begin
        mode <= FIX_MAC_MODE;
    end
end

// a_data, b_data, c_data;
assign a_data = (mode == CONV_MODE) ? data_in   : mul_val_reg;
assign b_data = (mode == CONV_MODE) ? weight_in : mac_reg;
assign c_data = (mode == CONV_MODE) ? mac_reg  : add_val_reg;

assign t_valid = pe_cmd_valid && pe_cmd == TRIGGER;

always@(posedge clk_i) begin
    if(rst_i) begin
        mac_reg <= 0;
    end
    else if(pe_cmd_valid && pe_cmd == RESET) begin
        mac_reg <= 0;
    end 
    else if(pe_cmd_valid && pe_cmd == LOAD_DATA) begin
        mac_reg <= param_2_in;
    end 
    else if(r_valid) begin
        mac_reg <= r_data;
    end
end

always@(posedge clk_i) begin
    if(rst_i) begin
        busy <= 0;
    end
    else if(pe_cmd_valid && pe_cmd == TRIGGER) begin
        busy <= 1;
    end
    else if(r_valid) begin
        busy <= 0;
    end
end

assign mac_value = mac_reg;


`ifdef VIVADO_ENV
floating_point_0 FP(

    .aclk(clk_i),

    .s_axis_a_tdata(a_data),
    .s_axis_a_tvalid(t_valid),
    //.s_axis_a_tready(a_ready),

    .s_axis_b_tdata(b_data),
    .s_axis_b_tvalid(t_valid),
    //.s_axis_b_tready(b_ready),

    .s_axis_c_tdata(c_data),
    .s_axis_c_tvalid(t_valid),
    //.s_axis_c_tready(c_ready),

    .m_axis_result_tdata(r_data),
    .m_axis_result_tvalid(r_valid)
    //.m_axis_result_tready(r_ready)
);
`elsif
    logic signed [WIDTH-1:0] a_reg, b_reg;
    logic signed [WIDTH-1:0] result_reg;

    always_ff @(posedge clk) begin
        if (rst) begin
        a_reg <= 0;
        b_reg <= 0;
        result_reg <= 0;
        end else begin
        a_reg <= a;
        b_reg <= b;

        // Fake floating-point multiplication using shortreal approximation.
        //  This is a simplified approach and does NOT accurately reflect 
        //  true floating-point behavior.  It's for demonstration only.

        // 1. Convert to shortreal (approximation)
        shortreal a_short = $itor(a_reg);
        shortreal b_short = $itor(b_reg);

        // 2. Multiply the shortreals. This is where the "floating-point-like"
        //    behavior is simulated.  The shortreal type handles some of the
        //    fractional part.
        shortreal result_short = a_short * b_short;

        // 3. Convert back to integer, handling potential overflow/underflow
        //    This is a VERY crude approximation.  Real FP has exponents, etc.
        if (result_short > $pow(2, WIDTH-1) -1 ) begin
            result_reg <= $pow(2, WIDTH-1) -1; //Saturate positive overflow
        end else if (result_short < -$pow(2, WIDTH-1)) begin
            result_reg <= -$pow(2, WIDTH-1); // Saturate negative overflow
        end else begin
            result_reg <= $rtoi(result_short); // Round to integer
        end

        end
    end

  assign result = result_reg;
 
`endif
 


endmodule
