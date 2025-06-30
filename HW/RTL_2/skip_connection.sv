`timescale 1ns / 1ps

module SKIPCONNECTION
#(parameter ACLEN=8,
  parameter ADDR_BITS=16,
  parameter DATA_WIDTH = 16
)
(
    input  logic                         clk_i, rst_i,
    input   logic                        relu_en,
    // 
    input logic                        skipConnectionInputValid,
    input logic   [DATA_WIDTH-1 : 0]   skipConnectionInput_1 [0 : 3],
    input logic   [DATA_WIDTH-1 : 0]   skipConnectionInput_2 [0 : 3],
    // batchNorm Result
    output logic                        skipConnectionResultValid,
    output logic   [DATA_WIDTH*4-1 : 0] skipConnectionResult_r
);

/*
 * skip connection signal
 */
logic   [DATA_WIDTH-1   : 0]  fma_a_data  [0 : 3];
logic                         fma_a_valid [0 : 3];
logic   [DATA_WIDTH-1   : 0]  fma_b_data  [0 : 3];
logic                         fma_b_valid [0 : 3];
logic   [DATA_WIDTH-1   : 0]  fma_c_data  [0 : 3];
logic                         fma_c_valid [0 : 3];
logic   [DATA_WIDTH-1   : 0]  fma_out        [0 : 3];
logic   [DATA_WIDTH-1   : 0]  fma_out_relu   [0 : 3];
logic                         fma_out_valid   [0 : 3];
logic                         fma_out_valid_r [0 : 3];
logic   [DATA_WIDTH*4-1 : 0]  fma_out_r;


generate
    always_ff @( posedge clk_i ) begin
        // for (int i = 0; i < 4; i++) begin
            skipConnectionResultValid <= fma_out_valid[0];
        // end
    end
endgenerate

// assign skipConnectionResultValid <= fma_out_valid[0];

generate
    always_comb begin
        for (int i = 0; i < 4; i++) begin
            if(fma_out[i][15]) begin
                fma_out_relu[i] = 0;
            end
            else begin
                fma_out_relu[i] = fma_out[i];
            end
        end
    end
endgenerate

generate
    always_ff @( posedge clk_i ) begin
        if(fma_out_valid[0] && !relu_en) begin
            skipConnectionResult_r <= {fma_out[0], fma_out[1], 
                          fma_out[2], fma_out[3]};
        end
        else if(fma_out_valid[0] && relu_en) begin
            skipConnectionResult_r <= {fma_out_relu[0], fma_out_relu[1], 
                          fma_out_relu[2], fma_out_relu[3]};
        end
    end
endgenerate
/*
 * skip connection : a * 1 + c
 */
always_comb begin
    for (int i = 0; i < 4; i++) begin
        fma_a_data[i] = skipConnectionInput_1[i];
        fma_b_data[i] = 16'h3c00;
        fma_c_data[i] = skipConnectionInput_2[i];

        fma_a_valid[i] = skipConnectionInputValid;//(curr_state == FMA_3_S);
        fma_b_valid[i] = skipConnectionInputValid;//(curr_state == FMA_3_S);
        fma_c_valid[i] = skipConnectionInputValid;//(curr_state == FMA_3_S);
    end     
end

/*
 * skip connection ip
 */
generate
for (genvar i = 0; i < 4; i++) begin
    floating_point_0 FP(

        .aclk(clk_i),

        .s_axis_a_tdata(fma_a_data[i]),
        .s_axis_a_tvalid(fma_a_valid[i]),

        .s_axis_b_tdata(fma_b_data[i]),
        .s_axis_b_tvalid(fma_b_valid[i]),

        .s_axis_c_tdata(fma_c_data[i]),
        .s_axis_c_tvalid(fma_c_valid[i]),

        .m_axis_result_tdata(fma_out[i]),
        .m_axis_result_tvalid(fma_out_valid[i])
    );
end
endgenerate


endmodule