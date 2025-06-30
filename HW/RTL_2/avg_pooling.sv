`timescale 1ns / 1ps

module AVGPOOLING
#(parameter ACLEN=8,
  parameter ADDR_BITS=16,
  parameter DATA_WIDTH = 16
)
(
    input  logic                     clk_i, rst_i,
    //
    input                            avgPoolingInputValid,
    input                            avgPoolingInputLast,
    input  logic[DATA_WIDTH-1   : 0] avgPoolingInput,
    // batchNorm Result
    output logic                        avgPoolingResultValid,
    output logic   [DATA_WIDTH-1   : 0] avgPoolingResult_r
);


typedef enum {IDLE_S, 
              WAIT_AVG_POOLING_ACC_S,
              AVG_POOLING_DIV_S,
              WAIT_AVG_POOLING_DIV_S
              } state_t;


logic   [DATA_WIDTH-1   : 0] pooling_data_r [0 : 51];
logic   [ADDR_BITS-1    : 0] pooling_index;
logic   [DATA_WIDTH-1   : 0] pooling_result;
logic   [DATA_WIDTH-1   : 0] acc_data_in , div_data_in;
logic   [DATA_WIDTH-1   : 0] acc_data_out, div_data_out;
logic                        acc_data_in_valid, div_data_valid;
logic                        acc_data_out_valid;
logic                        acc_data_in_last, acc_data_out_last;

logic   [ADDR_BITS-1    : 0] acc_recv_cnt;



state_t curr_state, next_state;

always_ff @(posedge clk_i) begin
    if(rst_i) begin
        curr_state <= IDLE_S;
    end
    else begin
        curr_state <= next_state;
    end
end

// next state logic
always_comb begin
    case (curr_state)
    IDLE_S: if     (avgPoolingInputValid) 
                next_state = WAIT_AVG_POOLING_ACC_S;
            else         
                next_state = IDLE_S;
    WAIT_AVG_POOLING_ACC_S: if(acc_recv_cnt == 49)
                                next_state = AVG_POOLING_DIV_S;
                            else
                                next_state = WAIT_AVG_POOLING_ACC_S;
    AVG_POOLING_DIV_S: next_state = WAIT_AVG_POOLING_DIV_S;
    WAIT_AVG_POOLING_DIV_S: if(avgPoolingResultValid) next_state = IDLE_S;
                            else next_state = WAIT_AVG_POOLING_DIV_S;
    default: next_state = IDLE_S;
    endcase
end

always_ff @( posedge clk_i ) begin
    if(rst_i)                     acc_recv_cnt <= 0;
    else if(curr_state == IDLE_S) acc_recv_cnt <= 0;
    else if(acc_data_out_valid)   acc_recv_cnt <= acc_recv_cnt + 1;
end

always_ff @( posedge clk_i ) begin
    if(acc_data_out_valid)  avgPoolingResult_r <= acc_data_out;
    else if(avgPoolingResultValid) avgPoolingResult_r <= div_data_out;
end

floating_point_acc ACC2(

    .aclk(clk_i),

    .s_axis_a_tdata(avgPoolingInput),
    .s_axis_a_tlast(avgPoolingInputLast),
    .s_axis_a_tvalid(avgPoolingInputValid),

    .m_axis_result_tdata(acc_data_out),
    .m_axis_result_tlast(acc_data_out_last),
    .m_axis_result_tvalid(acc_data_out_valid)
);
/*
 * div ip
 */
floating_point_div div2(
        .aclk(clk_i),
        .s_axis_a_tdata(avgPoolingResult_r),
        .s_axis_a_tvalid((curr_state == AVG_POOLING_DIV_S)),
        .s_axis_b_tdata(16'h5220), // 49
        .s_axis_b_tvalid(1),
        .m_axis_result_tdata(div_data_out),
        .m_axis_result_tvalid(avgPoolingResultValid)
    );



endmodule