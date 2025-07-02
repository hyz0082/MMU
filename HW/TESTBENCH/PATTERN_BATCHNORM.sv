`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

`define CYCLE_TIME 20.0
`include "tpu_cmd.svh"

module tb_batchnorm
#(parameter ACLEN  = 8,
  parameter DATA_WIDTH = 16
) ();

logic                      clk_i = 0;
logic                      rst_i;

real CYCLE;
initial CYCLE = `CYCLE_TIME;
always #(CYCLE/2.0) clk_i = ~clk_i;

integer cycles;
integer total_cycles;
real         converted_float;

// cmd
logic                        tpu_cmd_valid;
logic   [ACLEN-1 : 0]        tpu_cmd;
logic   [DATA_WIDTH-1 : 0]   tpu_param_1_in; 
logic   [DATA_WIDTH-1 : 0]   tpu_param_2_in;
logic                        relu_en;
logic                        batchNormSramIdxRst;
logic                        batchNormSramIdxInc;
// convolution result
logic                        convolutionResultValid;
logic   [DATA_WIDTH*4-1 : 0] rdata_out [0 : 3];
// batchNorm Result
logic                        batchNormResultValid;
logic   [DATA_WIDTH*4-1 : 0] batchNormResult_r [0 : 3];

function real float16_to_real(logic [15:0] float16_val);
    logic sign_bit;
    logic [4:0] exponent_bits;
    logic [9:0] mantissa_bits;

    real sign;
    real exponent_val; // Renamed to avoid conflict with 'exponent' for power calculation
    real mantissa_val;
    real result;

    sign_bit = float16_val[15];
    exponent_bits = float16_val[14:10];
    mantissa_bits = float16_val[9:0];

    // Determine sign
    sign = (sign_bit == 1) ? -1.0 : 1.0;

    // Handle special cases
    if (exponent_bits == 5'b11111) begin // Exponent all ones
      if (mantissa_bits == 10'b0) begin
        // Infinity
        result = (sign_bit == 1) ? (-1.0 / 0.0) : (1.0 / 0.0);
      end else begin
        // NaN (Not a Number)
        result = 0.0 / 0.0;
      end
    end else if (exponent_bits == 5'b00000) begin // Exponent all zeros
      if (mantissa_bits == 10'b0) begin
        // Zero
        result = 0.0;
      end else begin
        // Denormalized number
        mantissa_val = 0.0;
        for (int i = 0; i < 10; i++) begin
          if (mantissa_bits[i] == 1) begin
            mantissa_val = mantissa_val + (1.0 / (2.0 ** (10 - i)));
          end
        end
        result = sign * mantissa_val * (2.0 ** (-14)); // Exponent is fixed at -14 for denormalized
      end
    end else begin
      // Normalized number
      mantissa_val = 1.0; // Implied leading '1'
      for (int i = 0; i < 10; i++) begin
        if (mantissa_bits[i] == 1) begin
          mantissa_val = mantissa_val + (1.0 / (2.0 ** (10 - i)));
        end
      end
      exponent_val = exponent_bits - 15;
      result = sign * mantissa_val * (2.0 ** exponent_val);
    end

    return result;
endfunction


BATCHNORM b1 (
    clk_i, rst_i,
    // cmd
    tpu_cmd_valid,
    tpu_cmd,
    tpu_param_1_in, 
    tpu_param_2_in,
    relu_en,
    batchNormSramIdxRst,
    batchNormSramIdxInc,
    // convolution result
    convolutionResultValid,
    rdata_out,
    // batchNorm Result
    batchNormResultValid,
    batchNormResult_r
);


initial begin

    rst_i = 1'b1;
    cycles = 0;
    total_cycles = 0;

    reset_task;
    
    send_batchNorm_weight;
    batchNorm_calc;
    get_batchNorm_result;
    $finish;

end

task reset_task ; begin
    batchNormSramIdxRst = 0;
    batchNormSramIdxInc = 0;
    convolutionResultValid = 0;
    relu_en = 0;
    #(3*`CYCLE_TIME); rst_i = 1;
    #(3*`CYCLE_TIME); rst_i = 0;
    @(negedge clk_i);
end endtask

task send_batchNorm_weight; begin

    @(negedge clk_i);
    tpu_cmd_valid  = 1;
    tpu_cmd        = SET_BN_MUL_SRAM_0;
    tpu_param_1_in = 16'h4200;
    tpu_param_2_in = 0;
    @(negedge clk_i);
    tpu_cmd_valid  = 1;
    tpu_cmd        = SET_BN_MUL_SRAM_1;
    tpu_param_1_in = 16'h4200;
    tpu_param_2_in = 0;
    @(negedge clk_i);
    tpu_cmd_valid  = 1;
    tpu_cmd        = SET_BN_MUL_SRAM_2;
    tpu_param_1_in = 16'h4200;
    tpu_param_2_in = 0;
    @(negedge clk_i);
    tpu_cmd_valid  = 1;
    tpu_cmd        = SET_BN_MUL_SRAM_3;
    tpu_param_1_in = 16'h4200;
    tpu_param_2_in = 0;
    @(negedge clk_i);
    tpu_cmd_valid  = 1;
    tpu_cmd        = SET_BN_ADD_SRAM_0;
    tpu_param_1_in = 16'h4200;
    tpu_param_2_in = 0;
    @(negedge clk_i);
    tpu_cmd_valid  = 1;
    tpu_cmd        = SET_BN_ADD_SRAM_1;
    tpu_param_1_in = 16'h4200;
    tpu_param_2_in = 0;
    @(negedge clk_i);
    tpu_cmd_valid  = 1;
    tpu_cmd        = SET_BN_ADD_SRAM_2;
    tpu_param_1_in = 16'h4200;
    tpu_param_2_in = 0;
    @(negedge clk_i);
    tpu_cmd_valid  = 1;
    tpu_cmd        = SET_BN_ADD_SRAM_3;
    tpu_param_1_in = 16'h4200;
    tpu_param_2_in = 0;
    @(negedge clk_i);
    tpu_cmd_valid  = 0;
    @(negedge clk_i);

    $display("weight 1: {%f, %f}", float16_to_real(16'h4200), float16_to_real(16'h4200));
end endtask

task batchNorm_calc; begin

    @(negedge clk_i);
    batchNormSramIdxRst = 1;
    @(negedge clk_i);
    batchNormSramIdxRst = 0;
    repeat(3) @(negedge clk_i);
    
    rdata_out[0] = {16'h0000, 16'h3c00, 16'h4000, 16'h4200};
    rdata_out[1] = {16'h4400, 16'h4500, 16'h4600, 16'h4700};
    rdata_out[2] = {16'h4800, 16'h4880, 16'h4900, 16'h4980};
    rdata_out[3] = {16'h4a00, 16'h4a80, 16'h4b00, 16'h4b80};
    convolutionResultValid = 1;
    @(negedge clk_i);
    convolutionResultValid = 0;

    $display("ipnut data");
    for(int i = 0; i < 4; i++) begin
        $write("{");
        for(int j = 3; j >= 0; j--) begin
            converted_float = float16_to_real(rdata_out[i][(DATA_WIDTH*(j+1)-1)-:DATA_WIDTH]);
            $write("%f ", converted_float);
        end
        $write("}\n");  
    end


end endtask

task get_batchNorm_result; begin

    // wait(batchNormResultValid);
    $display("batchNorm result");
    while(batchNormResultValid !== 1) begin
        #(`CYCLE_TIME);
    end
    repeat(3) @(negedge clk_i);
    for(int i = 0; i < 4; i++) begin
        $write("{");
        for(int j = 3; j >= 0; j--) begin
            converted_float = float16_to_real(batchNormResult_r[i][(DATA_WIDTH*(j+1)-1)-:DATA_WIDTH]);
            $write("%f ", converted_float);
        end
        $write("}\n");  
    end
end endtask




endmodule
