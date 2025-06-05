`timescale 1ns / 1ps
// =============================================================================
//  Program : GeMM.sv
//  Author  : 
//  Date    : 
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// =============================================================================
// `include "config.vh"
module MMU
#(parameter ACLEN  = 8,
  parameter DATA_WIDTH = 32
//   parameter CLSIZE = `CLP
)
(
    /////////// System signals   ///////////////////////////////////////////////
    input   logic                        clk_i, rst_i,
    /////////// MMU command ///////////////////////////////////////////////
    input   logic                        mmu_cmd_valid, // cmd valid
    input   logic   [ACLEN : 0]          mmu_cmd,       // cmd
    input   logic   [DATA_WIDTH*4-1 : 0]   param_1_in,    // cmd ctrl
    input   logic   [DATA_WIDTH*4-1 : 0]   param_2_in,    // data
    input   logic   [DATA_WIDTH*4-1 : 0]   param_3_in,    // data
    input   logic   [DATA_WIDTH*4-1 : 0]   param_4_in,    // data

    /////////// MMU input   ///////////////////////////////////////////////
    input   logic   [DATA_WIDTH-1 : 0]   data_1_in,
    input   logic   [DATA_WIDTH-1 : 0]   data_2_in,
    input   logic   [DATA_WIDTH-1 : 0]   data_3_in,
    input   logic   [DATA_WIDTH-1 : 0]   data_4_in,
    input   logic   [DATA_WIDTH-1 : 0]   weight_1_in,
    input   logic   [DATA_WIDTH-1 : 0]   weight_2_in,
    input   logic   [DATA_WIDTH-1 : 0]   weight_3_in,
    input   logic   [DATA_WIDTH-1 : 0]   weight_4_in,
    /////////// MMU outpupt ///////////////////////////////////////////////
    output  logic   [DATA_WIDTH*4-1 : 0] rdata_1_out,
    output  logic   [DATA_WIDTH*4-1 : 0] rdata_2_out,
    output  logic   [DATA_WIDTH*4-1 : 0] rdata_3_out,
    output  logic   [DATA_WIDTH*4-1 : 0] rdata_4_out,

    input   logic   [DATA_WIDTH*4-1 : 0] bn_data_1_in,
    input   logic   [DATA_WIDTH*4-1 : 0] bn_data_2_in,
    input   logic   [DATA_WIDTH*4-1 : 0] bn_data_3_in,
    input   logic   [DATA_WIDTH*4-1 : 0] bn_data_4_in,

    output  logic   [DATA_WIDTH*4-1 : 0] bn_data_1_out,
    output  logic   [DATA_WIDTH*4-1 : 0] bn_data_2_out,
    output  logic   [DATA_WIDTH*4-1 : 0] bn_data_3_out,
    output  logic   [DATA_WIDTH*4-1 : 0] bn_data_4_out,
    output  logic   bn_valid,

    output  logic                        mmu_busy     // 0 for idle, 1 for busy
);


// mmu_cmd table
localparam  RESET             = 0; // whole
localparam  TRIGGER           = 1; // whole
localparam  TRIGGER_LAST      = 2; // 
                                  // param_1_in: 
                                  // param_2_in:
localparam  SET_MUL_VAL       = 3; // partial
                                  // param_1_in: column number
                                  // param_2_in: multipled value

localparam  SET_ADD_VAL       = 4; // partial
                                  // param_1_in: column number
                                  // param_2_in: added value

localparam  SET_PE_VAL        = 5; // partial
                                  // param_1_in: PE number
                                  // param_2_in: initialized value
localparam  SET_CONV_MODE     = 6; // whole
localparam  SET_FIX_MAC_MODE  = 7; // whole
localparam  FORWARD              = 8; // whole
localparam  TRIGGER_BN   = 17; // whole

logic [DATA_WIDTH-1 : 0] row_1_reg;
logic [DATA_WIDTH-1 : 0] row_2_reg [0:1];
logic [DATA_WIDTH-1 : 0] row_3_reg [0:2];

logic [DATA_WIDTH-1 : 0] col_1_reg;
logic [DATA_WIDTH-1 : 0] col_2_reg [0:1];
logic [DATA_WIDTH-1 : 0] col_3_reg [0:2];

//#########################
//    4 X 4 PE SIGNAL
//#########################
logic          [15 : 0]     pe_busy   ;
logic [DATA_WIDTH-1 : 0] data_in   [0:3];
logic [DATA_WIDTH-1 : 0] weight_in [0:3];

logic [DATA_WIDTH-1 : 0] data_out   [0:16];
logic [DATA_WIDTH-1 : 0] weight_out [0:16];

logic [DATA_WIDTH-1 : 0] mac_value  [0:15];

/// PE CMD ENABLE SIGNAL /////
logic pe_cmd_valid [0:15];
logic whole_pe_select;
logic column_select [0:3];

logic   [DATA_WIDTH*4-1 : 0] params [0 : 3];

logic   [DATA_WIDTH*4-1 : 0] bn_in  [0 : 3];
logic   [DATA_WIDTH-1   : 0] bn_out [0 : 15];
logic                        pe_bn_valid [0 : 15];


assign params[0] = param_1_in;
assign params[1] = param_2_in;
assign params[2] = param_3_in;
assign params[3] = param_4_in;

// assign bn_in[0] = bn_data_1_in;
// assign bn_in[1] = bn_data_2_in;
// assign bn_in[2] = bn_data_3_in;
// assign bn_in[3] = bn_data_4_in;

// assign bn_data_1_out = {bn_out[0], bn_out[4], bn_out[8] , bn_out[12]};
// assign bn_data_2_out = {bn_out[1], bn_out[5], bn_out[9] , bn_out[13]};
// assign bn_data_3_out = {bn_out[2], bn_out[6], bn_out[10], bn_out[14]};
// assign bn_data_4_out = {bn_out[3], bn_out[7], bn_out[11], bn_out[15]};

assign bn_valid = pe_bn_valid[0];

assign whole_pe_select = (mmu_cmd == RESET)            || 
                         (mmu_cmd == TRIGGER)          ||
                         (mmu_cmd == TRIGGER_LAST)     ||
                         (mmu_cmd == SET_CONV_MODE)    || 
                         (mmu_cmd == SET_FIX_MAC_MODE) || 
                         (mmu_cmd == FORWARD)          ||
                         (mmu_cmd == SET_PE_VAL)       ||
                         (mmu_cmd == TRIGGER_BN );

assign column_select[0] = (mmu_cmd == SET_MUL_VAL) && (param_1_in == 0) ||
                          (mmu_cmd == SET_ADD_VAL) && (param_1_in == 0);

assign column_select[1] = (mmu_cmd == SET_MUL_VAL) && (param_1_in == 1) ||
                          (mmu_cmd == SET_ADD_VAL) && (param_1_in == 1);

assign column_select[2] = (mmu_cmd == SET_MUL_VAL) && (param_1_in == 2) ||
                          (mmu_cmd == SET_ADD_VAL) && (param_1_in == 2);

assign column_select[3] = (mmu_cmd == SET_MUL_VAL) && (param_1_in == 3) ||
                          (mmu_cmd == SET_ADD_VAL) && (param_1_in == 3);
always_comb begin : PE_VALID
       // column 0
       pe_cmd_valid[0]  = (mmu_cmd_valid && whole_pe_select)     || 
                          (mmu_cmd_valid && column_select[0]);
       pe_cmd_valid[4]  = (mmu_cmd_valid && whole_pe_select)     || 
                          (mmu_cmd_valid && column_select[0]);
       pe_cmd_valid[8]  = (mmu_cmd_valid && whole_pe_select)     || 
                          (mmu_cmd_valid && column_select[0]);
       pe_cmd_valid[12] = (mmu_cmd_valid && whole_pe_select)     || 
                          (mmu_cmd_valid && column_select[0]);
       // column 1
       pe_cmd_valid[1]  = (mmu_cmd_valid && whole_pe_select)     || 
                          (mmu_cmd_valid && column_select[1]);
       pe_cmd_valid[5]  = (mmu_cmd_valid && whole_pe_select)     || 
                          (mmu_cmd_valid && column_select[1]);
       pe_cmd_valid[9]  = (mmu_cmd_valid && whole_pe_select)     || 
                          (mmu_cmd_valid && column_select[1]);
       pe_cmd_valid[13] = (mmu_cmd_valid && whole_pe_select)     || 
                          (mmu_cmd_valid && column_select[1]);
       // column 2
       pe_cmd_valid[2]  = (mmu_cmd_valid && whole_pe_select)     || 
                          (mmu_cmd_valid && column_select[2]);
       pe_cmd_valid[6]  = (mmu_cmd_valid && whole_pe_select)     || 
                          (mmu_cmd_valid && column_select[2]);
       pe_cmd_valid[10] = (mmu_cmd_valid && whole_pe_select)     || 
                          (mmu_cmd_valid && column_select[2]);
       pe_cmd_valid[14] = (mmu_cmd_valid && whole_pe_select)     || 
                          (mmu_cmd_valid && column_select[2]);
       // column 3
       pe_cmd_valid[3]  = (mmu_cmd_valid && whole_pe_select)     || 
                          (mmu_cmd_valid && column_select[3]);
       pe_cmd_valid[7]  = (mmu_cmd_valid && whole_pe_select)     || 
                          (mmu_cmd_valid && column_select[3]);
       pe_cmd_valid[11] = (mmu_cmd_valid && whole_pe_select)     || 
                          (mmu_cmd_valid && column_select[3]);
       pe_cmd_valid[15] = (mmu_cmd_valid && whole_pe_select)     || 
                          (mmu_cmd_valid && column_select[3]);  
                         
end

// logic [DATA_WIDTH-1 : 0] data_in   [0:3];
always_ff @( posedge clk_i) begin
       if(mmu_cmd_valid && mmu_cmd == RESET) begin
              // ROW 1
              row_1_reg    <= 0;
              // ROW 2
              row_2_reg[1] <= 0;
              row_2_reg[0] <= 0;
              // ROW 3
              row_3_reg[2] <= 0;
              row_3_reg[1] <= 0;
              row_3_reg[0] <= 0;
              // COL 1
              col_1_reg    <= 0;
              // COL 2
              col_2_reg[1] <= 0;
              col_2_reg[0] <= 0;
              // COL 3
              col_3_reg[2] <= 0;
              col_3_reg[1] <= 0; 
              col_3_reg[0] <= 0;
       end
       if(mmu_cmd_valid && mmu_cmd == TRIGGER ||
          mmu_cmd_valid && mmu_cmd == TRIGGER_LAST ||
          mmu_cmd_valid && mmu_cmd == FORWARD) begin
              // ROW 1
              row_1_reg    <= data_2_in;
              // ROW 2
              row_2_reg[1] <= data_3_in;
              row_2_reg[0] <= row_2_reg[1];
              // ROW 3
              row_3_reg[2] <= data_4_in;
              row_3_reg[1] <= row_3_reg[2];
              row_3_reg[0] <= row_3_reg[1];
              // COL 1
              col_1_reg    <= weight_2_in;
              // COL 2
              col_2_reg[1] <= weight_3_in;
              col_2_reg[0] <= col_2_reg[1];
              // COL 3
              col_3_reg[2] <= weight_4_in;
              col_3_reg[1] <= col_3_reg[2]; 
              col_3_reg[0] <= col_3_reg[1]; 
       end 
end

always_comb begin
       data_in[0]   = data_1_in;
       data_in[1]   = row_1_reg;
       data_in[2]   = row_2_reg[0];
       data_in[3]   = row_3_reg[0];

       weight_in[0] = weight_1_in;
       weight_in[1] = col_1_reg;
       weight_in[2] = col_2_reg[0];
       weight_in[3] = col_3_reg[0];
end

//#########################
//       4 X 4 PE
//#########################
genvar i;

generate
       for(i = 0; i < 16; i++) begin
              if(i == 0) begin
                     PE #(
                         .ACLEN(ACLEN), // ADDR_BITS
                         .DATA_WIDTH(DATA_WIDTH)  // DATA_WIDTH
                     ) 
                     pe0 (
                         .clk_i(clk_i), 
                         .rst_i(rst_i),
                         .pe_cmd_valid(pe_cmd_valid[0]),
                         .pe_cmd(mmu_cmd),
                         .param_1_in(param_1_in),
                         .param_2_in(param_2_in),
                         .preload_data_in(params[0][(DATA_WIDTH*4-1)-:DATA_WIDTH]),
                         .data_in(data_in[0]),
                         .weight_in(weight_in[0]),
                         .data_out(data_out[0]), 
                         .weight_out(weight_out[0]),
                         .mac_value(mac_value[0]),
                         .busy(pe_busy[0])
                     );
              end
              else if (i == 1 || i == 2 || i == 3) begin
                     PE #(
                         .ACLEN(ACLEN), // ADDR_BITS
                         .DATA_WIDTH(DATA_WIDTH)  // DATA_WIDTH
                     )
                     pe1 (
                         .clk_i(clk_i), 
                         .rst_i(rst_i),
                         .pe_cmd_valid(pe_cmd_valid[i]),
                         .pe_cmd(mmu_cmd),
                         .param_1_in(param_1_in),
                         .param_2_in(param_2_in),
                         .preload_data_in(params[i][(DATA_WIDTH*4-1)-:DATA_WIDTH]),
                         .data_in(data_out[i-1]),
                         .weight_in(weight_in[i]),
                         .data_out(data_out[i]), 
                         .weight_out(weight_out[i]),
                         .mac_value(mac_value[i]),
                         .busy(pe_busy[i])
                     );
              end
              else if (i == 4 || i == 8 || i == 12) begin
                     PE #(
                         .ACLEN(ACLEN), // ADDR_BITS
                         .DATA_WIDTH(DATA_WIDTH)  // DATA_WIDTH
                     )
                     pe2 (
                         .clk_i(clk_i), 
                         .rst_i(rst_i),
                         .pe_cmd_valid(pe_cmd_valid[i]),
                         .pe_cmd(mmu_cmd),
                         .param_1_in(param_1_in),
                         .param_2_in(param_2_in),
                         .preload_data_in(params[0][(DATA_WIDTH*4-1)-(DATA_WIDTH*(i/4)) : (DATA_WIDTH*3)-(DATA_WIDTH*(i/4))]),
                         .data_in(data_in[i/4]),
                         .weight_in(weight_out[i-4]),
                         .data_out(data_out[i]), 
                         .weight_out(weight_out[i]),
                         .mac_value(mac_value[i]),
                         .busy(pe_busy[i])
                     );
              end
              else begin
                     PE #(
                         .ACLEN(ACLEN), // ADDR_BITS
                         .DATA_WIDTH(DATA_WIDTH)  // DATA_WIDTH
                     )
                     pe (
                         .clk_i(clk_i), 
                         .rst_i(rst_i),
                         .pe_cmd_valid(pe_cmd_valid[i]),
                         .pe_cmd(mmu_cmd),
                         .param_1_in(param_1_in),
                         .param_2_in(param_2_in),
                         .preload_data_in(params[i%4][(DATA_WIDTH*4-1)-(DATA_WIDTH*(i/4)) : (DATA_WIDTH*3)-(DATA_WIDTH*(i/4))]),
                         .data_in(data_out[i-1]),
                         .weight_in(weight_out[i-4]),
                         .data_out(data_out[i]), 
                         .weight_out(weight_out[i]),
                         .mac_value(mac_value[i]),
                         .busy(pe_busy[i])
                     );
              end
       end
       
endgenerate

assign rdata_1_out = {mac_value[0], mac_value[4], mac_value[ 8], mac_value[12]};
assign rdata_2_out = {mac_value[1], mac_value[5], mac_value[ 9], mac_value[13]};
assign rdata_3_out = {mac_value[2], mac_value[6], mac_value[10], mac_value[14]};
assign rdata_4_out = {mac_value[3], mac_value[7], mac_value[11], mac_value[15]};

always_comb begin
       mmu_busy = |pe_busy;
end


endmodule
