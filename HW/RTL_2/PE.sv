`timescale 1ns / 1ps
// =============================================================================
//  Program : PE.v
//  Author  : 
//  Date    : 
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// =============================================================================

`define VIVADO_ENV

module PE
#(parameter ACLEN  = 8,
  parameter DATA_WIDTH = 32
)
(
    input                         clk_i, rst_i,
    
    input                         pe_cmd_valid,
    input      [ACLEN : 0]        pe_cmd,
    input      [DATA_WIDTH-1 : 0] param_1_in,
    input      [DATA_WIDTH-1 : 0] param_2_in,
    input      [DATA_WIDTH-1 : 0] preload_data_in, 

    /////////// PE input   ///////////////////////////////////////////////
    input      [DATA_WIDTH-1 : 0] data_in,
    input      [DATA_WIDTH-1 : 0] weight_in,
    /////////// PE outpupt ///////////////////////////////////////////////
    output reg [DATA_WIDTH-1 : 0] data_out,
    output reg [DATA_WIDTH-1 : 0] weight_out,
    output     [DATA_WIDTH-1 : 0] mac_value,

    output reg                    busy      // 0 for idle, 1 for busy
);

//#########################
//#     PE_CMD TABLE      #
//#########################
localparam  RESET             = 0;
localparam  TRIGGER           = 1; // 
                                  // param_1_in: 
                                  // param_2_in:
localparam  TRIGGER_LAST      = 2; // 
                                  // param_1_in: 
                                  // param_2_in:
// localparam  SET_MUL_VAL       = 3; // 
//                                   // param_1_in: 
//                                   // param_2_in:
// localparam  SET_ADD_VAL       = 4; // 
//                                   // param_1_in: 
//                                   // param_2_in:
localparam  LOAD_DATA         = 5; // 
                                  // param_1_in: 
                                  // param_2_in:
localparam  SET_CONV_MODE     = 6; // 
                                  // param_1_in: conv_len
                                  // param_2_in: 
localparam  SET_FIX_MAC_MODE  = 7; // 
                                  // param_1_in: 
                                  // param_2_in:
localparam  FORWARD              = 8;
localparam  TRIGGER_BN   = 17; // whole

//#########################
//#         MODE          #
//#########################
localparam  CONV_MODE          = 0;
localparam  FIX_MAC_MODE       = 1;

// reg mode; // 0 for normal mac, 1 for fix value mac

reg [DATA_WIDTH-1 : 0] mac_reg;
// reg [DATA_WIDTH-1 : 0] mul_val_reg;
// reg [DATA_WIDTH-1 : 0] add_val_reg;

wire [DATA_WIDTH-1 : 0] a_data, b_data, c_data;
wire                    t_valid;
wire [DATA_WIDTH-1 : 0] r_data;
wire                    r_valid;

wire [DATA_WIDTH-1 : 0] acc_data;
wire                    acc_valid;
wire                    acc_last;

reg  [DATA_WIDTH-1 : 0] r_data_reg;
reg                     r_last_reg;
reg                     r_valid_reg;

reg  [DATA_WIDTH-1 : 0] last_cnt;
reg  [DATA_WIDTH-1 : 0] conv_len;


//#########################
//#     BYPASS INPUT      #
//#########################
always@(posedge clk_i) begin
    if(rst_i) begin
        weight_out <= 0;
        data_out   <= 0;
    end
    else if(pe_cmd_valid && pe_cmd == TRIGGER || 
            pe_cmd_valid && pe_cmd == TRIGGER_LAST || 
            pe_cmd_valid && pe_cmd == FORWARD)begin
        weight_out <= weight_in;
        data_out   <= data_in;
    end
end


//#########################
//#    SET MUL/ADD VAL    #
//#########################
// always@(posedge clk_i) begin
//     if(rst_i) begin
//         mul_val_reg <= 0;
//         add_val_reg <= 0;
//     end
//     else if(pe_cmd_valid && pe_cmd == SET_MUL_VAL) begin
//         mul_val_reg <= param_2_in;
//     end
//     else if(pe_cmd_valid && pe_cmd == SET_ADD_VAL) begin
//         add_val_reg <= param_2_in;
//     end
// end

//#########################
//#   SET MODE(CONV/BN)   #
//#########################
// always@(posedge clk_i) begin
//     if(rst_i) begin
//         mode <= 0;
//     end
//     else if(pe_cmd_valid && pe_cmd == SET_CONV_MODE) begin
//         mode <= CONV_MODE;
//     end
//     else if(pe_cmd_valid && pe_cmd == SET_FIX_MAC_MODE) begin
//         mode <= FIX_MAC_MODE;
//     end
// end

assign a_data = data_in;
assign b_data = weight_in;
assign c_data = 0;

assign t_valid = pe_cmd_valid && pe_cmd == TRIGGER      ||
                 pe_cmd_valid && pe_cmd == TRIGGER_LAST ||
                 pe_cmd_valid && pe_cmd == FORWARD      ||
                 pe_cmd_valid && pe_cmd == TRIGGER_BN;


//#########################
//#        MAC REG        #
//#########################
always@(posedge clk_i) begin
    if(rst_i) begin
        mac_reg <= 0;
    end
    else if(pe_cmd_valid && pe_cmd == RESET) begin
        mac_reg <= 0;
    end 
    // else if(pe_cmd_valid && pe_cmd == LOAD_DATA) begin
    //     mac_reg <= param_2_in;
    // end
    else if(acc_last) begin
        mac_reg <= acc_data;
    end
end

assign mac_value = mac_reg;

//#########################
//#         BUSY          #
//#########################
always@(posedge clk_i) begin
    if(rst_i) begin
        busy <= 0;
    end
    else if(pe_cmd_valid && pe_cmd == RESET) begin
        busy <= 0;
    end
    else if(pe_cmd_valid && pe_cmd == TRIGGER) begin
        busy <= 1;
    end
    else if(acc_last) begin
        busy <= 0;
    end
end

//#########################
//#     STORE R_DATA      #
//#########################
always_ff @( posedge clk_i ) begin
    if(pe_cmd_valid && pe_cmd == RESET) begin
        r_valid_reg <= 1;
    end
    else if(r_valid) begin
        r_data_reg  <= r_data;
        r_valid_reg <= r_valid;
    end
    else if(pe_cmd_valid && pe_cmd == LOAD_DATA) begin
        r_data_reg  <= preload_data_in;
        r_valid_reg <= 1;
    end
    else begin
        r_valid_reg <= 0;
    end
end

//#########################
//#      R_LAST_REG       #
//#########################
always_ff @( posedge clk_i ) begin
    if(last_cnt == conv_len - 1) begin
        r_last_reg <= 1;
    end
    else if(pe_cmd_valid && pe_cmd == RESET) begin
        // reset accumulator
        r_last_reg <= 1;
    end
    else begin
        r_last_reg <= 0;
    end    
end

//#########################
//#       LAST_CNT        #
//#########################
always_ff @(posedge clk_i) begin
    if(rst_i) begin
        last_cnt <= 0;
    end
    else if(pe_cmd_valid && pe_cmd == RESET) begin
        last_cnt <= 0;
    end
    else if(r_valid) begin
        if(last_cnt == conv_len - 1) begin
            last_cnt <= 0;
        end
        else begin
            last_cnt <= last_cnt + 1;
        end
    end
end

//#########################
//#     SET CONV_LEN      #
//#########################
always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        conv_len <= 0;
    end
    else if(pe_cmd_valid && pe_cmd == RESET) begin
        conv_len <= 0;
    end
    else if(pe_cmd_valid && pe_cmd == SET_CONV_MODE) begin
        conv_len <= param_1_in + 7;
    end
end



`ifdef VIVADO_ENV
//#########################
//#        FP MAC         #
//#########################
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

//#########################
//#        FP ACC         #
//#########################
floating_point_acc ACC(

    .aclk(clk_i),

    .s_axis_a_tdata(r_data_reg),
    .s_axis_a_tlast(r_last_reg),
    .s_axis_a_tvalid(r_valid_reg),

    .m_axis_result_tdata(acc_data),
    .m_axis_result_tlast(acc_last),
    .m_axis_result_tvalid(acc_valid)
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
