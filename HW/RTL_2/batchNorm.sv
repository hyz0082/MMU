`timescale 1ns / 1ps

// perform max pooling operation (3x3 stride=2, padding=1)
// read 112 data and replace one lane each time
// send 56 results back tp tpu

module BATCHNORM
#(parameter ACLEN=8,
  parameter ADDR_BITS=16,
  parameter DATA_WIDTH = 16
)
(
    input  logic                         clk_i, rst_i,
    // cmd
    input   logic                        tpu_cmd_valid,
    input   logic   [ACLEN-1 : 0]        tpu_cmd,
    input   logic   [DATA_WIDTH-1 : 0]   tpu_param_1_in, 
    input   logic   [DATA_WIDTH-1 : 0]   tpu_param_2_in,
    intput  logic                        relu_en,
    input   logic                        batchNormSramIdxRst,
    input   logic                        batchNormSramIdxInc,
    // convolution result
    intput logic                        convolutionResultValid,
    input  logic   [DATA_WIDTH*4-1 : 0] rdata_out [0 : 3],
    // batchNorm Result
    output logic                        batchNormResultValid,
    output logic   [DATA_WIDTH*4-1 : 0] batchNormResult_r [0 : 3]
);
`include "tpu_cmd.svh"
logic   [DATA_WIDTH-1   : 0]  bn_fma_a_data    [0 : 15];
logic                         bn_fma_a_valid   [0 : 15];
logic   [DATA_WIDTH-1   : 0]  bn_fma_b_data    [0 : 15];
logic                         bn_fma_b_valid   [0 : 15];
logic   [DATA_WIDTH-1   : 0]  bn_fma_c_data    [0 : 15];
logic                         bn_fma_c_valid   [0 : 15];
logic   [DATA_WIDTH-1   : 0]  bn_fma_out       [0 : 15];
logic   [DATA_WIDTH-1   : 0]  bn_fma_out_relu  [0 : 15];
logic                         bn_fma_out_valid [0 : 15];

logic                         bn_fma_out_valid_r [0 : 15];
logic   [DATA_WIDTH*4-1 : 0]  bn_fma_out_r      [0 : 3];

/*
 * BatchNorm mul & add sram signal
 */
logic [3 : 0]       bn_mul_wren;
logic [3 : 0]       bn_add_wren;
logic                         bn_source         [0 : 3];
logic   [ADDR_BITS-1  : 0]    bn_sram_idx       [0 : 3];
logic   [DATA_WIDTH-1 : 0]    bn_mul_sram_in    [0 : 3];
logic   [DATA_WIDTH-1 : 0]    bn_add_sram_in    [0 : 3];
logic   [DATA_WIDTH-1 : 0]    bn_mul_sram_out   [0 : 3];
logic   [DATA_WIDTH-1 : 0]    bn_mul_sram_out_r [0 : 3];
logic   [DATA_WIDTH-1 : 0]    bn_add_sram_out   [0 : 3];
logic   [DATA_WIDTH-1 : 0]    bn_add_sram_out_r [0 : 3];

logic   [DATA_WIDTH-1 : 0]    bn_mul_val_r [0 : 3];
logic   [DATA_WIDTH-1 : 0]    bn_add_val_r [0 : 3];



/*
 * BATCHNORM HARDWARE
 */
// a
always_comb begin
    bn_fma_a_data[ 0] = bn_mul_sram_out_r[0];
    bn_fma_a_data[ 4] = bn_mul_sram_out_r[0];
    bn_fma_a_data[ 8] = bn_mul_sram_out_r[0];
    bn_fma_a_data[12] = bn_mul_sram_out_r[0];

    bn_fma_a_data[ 1] = bn_mul_sram_out_r[1];
    bn_fma_a_data[ 5] = bn_mul_sram_out_r[1];
    bn_fma_a_data[ 9] = bn_mul_sram_out_r[1];
    bn_fma_a_data[13] = bn_mul_sram_out_r[1];

    bn_fma_a_data[ 2] = bn_mul_sram_out_r[2];
    bn_fma_a_data[ 6] = bn_mul_sram_out_r[2];
    bn_fma_a_data[10] = bn_mul_sram_out_r[2];
    bn_fma_a_data[14] = bn_mul_sram_out_r[2];

    bn_fma_a_data[ 3] = bn_mul_sram_out_r[3];
    bn_fma_a_data[ 7] = bn_mul_sram_out_r[3];
    bn_fma_a_data[11] = bn_mul_sram_out_r[3];
    bn_fma_a_data[15] = bn_mul_sram_out_r[3];
end
// b
always_comb begin
    {bn_fma_b_data[ 0], bn_fma_b_data[ 4], 
     bn_fma_b_data[ 8], bn_fma_b_data[12]} = rdata_out[0];

    {bn_fma_b_data[ 1], bn_fma_b_data[ 5], 
     bn_fma_b_data[ 9], bn_fma_b_data[13]} = rdata_out[1];

    {bn_fma_b_data[ 2], bn_fma_b_data[ 6], 
     bn_fma_b_data[10], bn_fma_b_data[14]} = rdata_out[2];

    {bn_fma_b_data[ 3], bn_fma_b_data[ 7], 
     bn_fma_b_data[11], bn_fma_b_data[15]} = rdata_out[3];
end
// c
always_comb begin
   bn_fma_c_data[ 0] = bn_add_sram_out_r[0];
   bn_fma_c_data[ 4] = bn_add_sram_out_r[0];
   bn_fma_c_data[ 8] = bn_add_sram_out_r[0];
   bn_fma_c_data[12] = bn_add_sram_out_r[0];

   bn_fma_c_data[ 1] = bn_add_sram_out_r[1];
   bn_fma_c_data[ 5] = bn_add_sram_out_r[1];
   bn_fma_c_data[ 9] = bn_add_sram_out_r[1];
   bn_fma_c_data[13] = bn_add_sram_out_r[1];

   bn_fma_c_data[ 2] = bn_add_sram_out_r[2];
   bn_fma_c_data[ 6] = bn_add_sram_out_r[2];
   bn_fma_c_data[10] = bn_add_sram_out_r[2];
   bn_fma_c_data[14] = bn_add_sram_out_r[2];

   bn_fma_c_data[ 3] = bn_add_sram_out_r[3];
   bn_fma_c_data[ 7] = bn_add_sram_out_r[3];
   bn_fma_c_data[11] = bn_add_sram_out_r[3];
   bn_fma_c_data[15] = bn_add_sram_out_r[3];
end

generate
    always_comb begin 
        for(int i = 0; i < 16; i++) begin
            bn_fma_a_valid[i] = conv_valid;//(curr_state == START_BATCHNORM_S);
            bn_fma_b_valid[i] = conv_valid;//(curr_state == START_BATCHNORM_S);
            bn_fma_c_valid[i] = conv_valid;//(curr_state == START_BATCHNORM_S);
        end
    end
endgenerate

/*
 * 4x4 FMA
 */
generate
for (genvar i = 0; i < 16; i++) begin
    floating_point_0 FP(

        .aclk(clk_i),

        .s_axis_a_tdata(bn_fma_a_data[i]),
        .s_axis_a_tvalid(bn_fma_a_valid[i]),

        .s_axis_b_tdata(bn_fma_b_data[i]),
        .s_axis_b_tvalid(bn_fma_b_valid[i]),

        .s_axis_c_tdata(bn_fma_c_data[i]),
        .s_axis_c_tvalid(bn_fma_c_valid[i]),

        .m_axis_result_tdata(bn_fma_out[i]),
        .m_axis_result_tvalid(bn_fma_out_valid[i])
    );
end
endgenerate

/*
 * relu
 */
generate
    always_comb begin
        for (int i = 0; i < 16; i++) begin
            if(bn_fma_out[i][15]) begin
                bn_fma_out_relu[i] = 0;
            end
            else begin
                bn_fma_out_relu[i] = bn_fma_out[i];
            end
        end
    end
endgenerate

generate
    always_ff @( posedge clk_i ) begin
        if(bn_fma_out_valid[0] && !relu_en) begin
            bn_fma_out_r[0] <= {bn_fma_out[ 0], bn_fma_out[ 4], 
                                bn_fma_out[ 8], bn_fma_out[12]};
            bn_fma_out_r[1] <= {bn_fma_out[ 1], bn_fma_out[ 5], 
                                bn_fma_out[ 9], bn_fma_out[13]};
            bn_fma_out_r[2] <= {bn_fma_out[ 2], bn_fma_out[ 6], 
                                bn_fma_out[10], bn_fma_out[14]};
            bn_fma_out_r[3] <= {bn_fma_out[ 3], bn_fma_out[ 7], 
                                bn_fma_out[11], bn_fma_out[15]};
        end
        else if(bn_fma_out_valid[0] && relu_en) begin
            bn_fma_out_r[0] <= {bn_fma_out_relu[ 0], bn_fma_out_relu[ 4], 
                                bn_fma_out_relu[ 8], bn_fma_out_relu[12]};
            bn_fma_out_r[1] <= {bn_fma_out_relu[ 1], bn_fma_out_relu[ 5], 
                                bn_fma_out_relu[ 9], bn_fma_out_relu[13]};
            bn_fma_out_r[2] <= {bn_fma_out_relu[ 2], bn_fma_out_relu[ 6], 
                                bn_fma_out_relu[10], bn_fma_out_relu[14]};
            bn_fma_out_r[3] <= {bn_fma_out_relu[ 3], bn_fma_out_relu[ 7], 
                                bn_fma_out_relu[11], bn_fma_out_relu[15]};
        end
    end
endgenerate

/*
 * BatchNorm MUL ADD SRAM
 */
always_comb begin
    bn_mul_wren[0] = is_tpu_cmd_valid_and_match(SET_BN_MUL_SRAM_0);
    bn_mul_wren[1] = is_tpu_cmd_valid_and_match(SET_BN_MUL_SRAM_1);
    bn_mul_wren[2] = is_tpu_cmd_valid_and_match(SET_BN_MUL_SRAM_2);
    bn_mul_wren[3] = is_tpu_cmd_valid_and_match(SET_BN_MUL_SRAM_3);

    bn_add_wren[0] = is_tpu_cmd_valid_and_match(SET_BN_ADD_SRAM_0);
    bn_add_wren[1] = is_tpu_cmd_valid_and_match(SET_BN_ADD_SRAM_1);
    bn_add_wren[2] = is_tpu_cmd_valid_and_match(SET_BN_ADD_SRAM_2);
    bn_add_wren[3] = is_tpu_cmd_valid_and_match(SET_BN_ADD_SRAM_3);
end

generate
    always_ff @( posedge clk_i ) begin
        for(int i = 0; i < 4; i++) begin
            if(batchNormSramIdxRst/*curr_state == IDLE_S*/) begin
                bn_sram_idx[i] <= 0;
            end
            else if(batchNormSramIdxInc/*curr_state == NEXT_COL_S*/) begin
                bn_sram_idx[i] <= bn_sram_idx[i] + 1;
            end
        end
    end
endgenerate

generate
for (genvar i = 0; i < 4; i++) begin
    global_buffer #(
    .ADDR_BITS(8),
    .DATA_BITS(DATA_WIDTH)
    )
    BN_mul_gbuff (
        .clk_i   (clk_i),
        .rst_i   (rst_i),
        .wr_en   (bn_mul_wren[i]),
        .index   ((bn_mul_wren[i]) ? tpu_param_2_in : bn_sram_idx[i]),
        .data_in (tpu_param_1_in),
        .data_out(bn_mul_sram_out[i])
    );
end
endgenerate

generate
for (genvar i = 0; i < 4; i++) begin
    global_buffer #(
    .ADDR_BITS(8),
    .DATA_BITS(DATA_WIDTH)
    )
    BN_add_gbuff (
        .clk_i   (clk_i),
        .rst_i   (rst_i),
        .wr_en   (bn_add_wren[i]),
        .index   ((bn_add_wren[i]) ? tpu_param_2_in : bn_sram_idx[i]),
        .data_in (tpu_param_1_in),
        .data_out(bn_add_sram_out[i])
    );
end
endgenerate

always_ff @( posedge clk_i ) begin
    for(int i = 0; i < 4; i++) begin
        bn_mul_sram_out_r[i] <= bn_mul_sram_out[i];
        bn_add_sram_out_r[i] <= bn_add_sram_out[i];
    end
end


endmodule