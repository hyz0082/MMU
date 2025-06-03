`timescale 1ns / 1ps

// perform max pooling operation (3x3 stride=2, padding=1)
// read 112 data and replace one lane each time
// send 56 results back tp tpu

module POOLING
#(parameter ACLEN=8,
  parameter ADDR_BITS=16,
  parameter DATA_WIDTH = 16
)
(
    /////////// System signals   ///////////////////////////////////////////////
    input   logic                          clk_i, rst_i,

    /// CMD ////
    input  logic                           reset_lans,
    input  logic                           set_lans_idx,
    input  logic                           sram_next,
    input  logic                           pooling_start,

    // batchNorm output
(* mark_debug="true" *)    input   logic                          bn_valid,
(* mark_debug="true" *)    input   logic   [DATA_WIDTH*4-1 : 0]   bn_out_1,
    input   logic   [DATA_WIDTH*4-1 : 0]   bn_out_2,
    input   logic   [DATA_WIDTH*4-1 : 0]   bn_out_3,
    input   logic   [DATA_WIDTH*4-1 : 0]   bn_out_4,

    // max pooling result
(* mark_debug="true" *)    output  logic                          max_pooling_valid,
(* mark_debug="true" *)    output  logic   [DATA_WIDTH*4-1 : 0]   max_pooling_out_1,
    output  logic   [DATA_WIDTH*4-1 : 0]   max_pooling_out_2,
    output  logic   [DATA_WIDTH*4-1 : 0]   max_pooling_out_3,
    output  logic   [DATA_WIDTH*4-1 : 0]   max_pooling_out_4,
    
    output  logic                          busy     // 0->idle, 1->busy
);

typedef enum {IDLE_S,
              RST_S,
              FILL_DATA_S,
              READ_DATA_S,
              START_MAX_POOLING_S,
              WAIT_MAX_POOLING_DONE_S,
              WRITE_DATA_S,

              DUMMY_S
              } state_t;

(* mark_debug="true" *) state_t curr_state, next_state;

/*
 * 
 */
(* mark_debug="true" *) logic [6 : 0] lans_1_idx;
logic [6 : 0] lans_2_idx;
logic [6 : 0] lans_3_idx;
logic [6 : 0] lans_4_idx;

logic   [DATA_WIDTH-1 : 0]   lans_1_in [0 : 3];
logic   [DATA_WIDTH-1 : 0]   lans_2_in [0 : 3];
logic   [DATA_WIDTH-1 : 0]   lans_3_in [0 : 3];
logic   [DATA_WIDTH-1 : 0]   lans_4_in [0 : 3];

logic [DATA_WIDTH-1   : 0] lans_1_out   [0 : 2];
logic [DATA_WIDTH-1   : 0] lans_2_out   [0 : 2];
logic [DATA_WIDTH-1   : 0] lans_3_out   [0 : 2];
logic [DATA_WIDTH-1   : 0] lans_4_out   [0 : 2];

logic [DATA_WIDTH-1   : 0] lans_1_data   [0 : 19];
logic [DATA_WIDTH-1   : 0] lans_2_data   [0 : 19];
logic [DATA_WIDTH-1   : 0] lans_3_data   [0 : 19];
logic [DATA_WIDTH-1   : 0] lans_4_data   [0 : 19];

/*
 * MAX POOLING (FOLLOW BY BATCHNORM)
 * depth 128 sram: 3x4
 */
logic [3 : 0] fill_data_cnt;
logic [3 : 0] read_out_cnt;
logic [2 : 0] max_pooling_sram_write_sel;
logic [1 : 0] pooling_result_cnt;

logic                        lans_1_cmp_in_valid         [0 : 20];
logic                        lans_1_cmp_result_valid_r   [0 : 20];
logic [7 : 0]                lans_1_cmp_result           [0 : 20];
logic                        lans_1_cmp_result_valid     [0 : 20];

logic                        lans_2_cmp_in_valid         [0 : 20];
logic                        lans_2_cmp_result_valid_r   [0 : 20];
logic [7 : 0]                lans_2_cmp_result [0 : 20];
logic                        lans_2_cmp_result_valid     [0 : 20];

logic                        lans_3_cmp_in_valid         [0 : 20];
logic                        lans_3_cmp_result_valid_r   [0 : 20];
logic [7 : 0]                lans_3_cmp_result [0 : 20];
logic                        lans_3_cmp_result_valid     [0 : 20];

logic                        lans_4_cmp_in_valid         [0 : 20];
logic                        lans_4_cmp_result_valid_r   [0 : 20];
logic [7 : 0]                lans_4_cmp_result [0 : 20];
logic                        lans_4_cmp_result_valid     [0 : 20];

logic   [DATA_WIDTH-1 : 0] lans_1_max_val [0 : 3];
logic   [DATA_WIDTH-1 : 0] lans_2_max_val [0 : 3];
logic   [DATA_WIDTH-1 : 0] lans_3_max_val [0 : 3];
logic   [DATA_WIDTH-1 : 0] lans_4_max_val [0 : 3];

always_comb begin
    busy = (curr_state != IDLE_S);
end

always_ff @(posedge clk_i) begin
    if(rst_i) begin
        curr_state <= IDLE_S;
    end
    else begin
        curr_state <= next_state;
    end
end
/*
 * NEXT STATE LOGIC
 */
always_comb begin
    case (curr_state)
    IDLE_S: if(bn_valid) next_state = FILL_DATA_S;
            else if(pooling_start)    next_state = READ_DATA_S;
            else if(reset_lans)       next_state = RST_S;
            else         next_state = IDLE_S;
    RST_S:  if(lans_1_idx == 120) next_state = IDLE_S;
            else                  next_state = RST_S;
    FILL_DATA_S: if(fill_data_cnt == 3) next_state = IDLE_S;
                 else                   next_state = FILL_DATA_S;
    READ_DATA_S: if(read_out_cnt == 3)  next_state = START_MAX_POOLING_S;
                 else                   next_state = READ_DATA_S;
    START_MAX_POOLING_S: next_state = WAIT_MAX_POOLING_DONE_S;
    WAIT_MAX_POOLING_DONE_S: if(lans_2_cmp_result_valid_r[7]) next_state = WRITE_DATA_S;
                             else next_state = WAIT_MAX_POOLING_DONE_S;
    WRITE_DATA_S:            if(lans_1_idx == 112) next_state = IDLE_S;
                             else next_state = READ_DATA_S;
    default: next_state = IDLE_S;
    endcase
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        max_pooling_valid <= 0;
    end
    else if(lans_1_cmp_result_valid_r[7] && pooling_result_cnt == 3) begin
        max_pooling_valid <= 1;
    end
    else begin
        max_pooling_valid <= 0;
    end
end

always_comb begin
    max_pooling_out_1 = {lans_1_max_val[0], lans_1_max_val[1], lans_1_max_val[2], lans_1_max_val[3]};
    max_pooling_out_2 = {lans_2_max_val[0], lans_2_max_val[1], lans_2_max_val[2], lans_2_max_val[3]};
    max_pooling_out_3 = {lans_3_max_val[0], lans_3_max_val[1], lans_3_max_val[2], lans_3_max_val[3]};
    max_pooling_out_4 = {lans_4_max_val[0], lans_4_max_val[1], lans_4_max_val[2], lans_4_max_val[3]};
end


always_ff @( posedge clk_i ) begin
    if(bn_valid) begin
        {lans_1_in[0], lans_1_in[1], lans_1_in[2], lans_1_in[3]} <= bn_out_1;
        {lans_2_in[0], lans_2_in[1], lans_2_in[2], lans_2_in[3]} <= bn_out_2;
        {lans_3_in[0], lans_3_in[1], lans_3_in[2], lans_3_in[3]} <= bn_out_3;
        {lans_4_in[0], lans_4_in[1], lans_4_in[2], lans_4_in[3]} <= bn_out_4;
    end
end

always_ff @( posedge clk_i ) begin
    if(curr_state == IDLE_S) begin
        fill_data_cnt <= 0;
    end else if(curr_state == FILL_DATA_S) begin
        fill_data_cnt <= fill_data_cnt + 1;
    end
end


always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        lans_1_idx <= 1;
        lans_2_idx <= 1;
        lans_3_idx <= 1;
        lans_4_idx <= 1;
    end
    else if(set_lans_idx) begin
        lans_1_idx <= 1;
        lans_2_idx <= 1;
        lans_3_idx <= 1;
        lans_4_idx <= 1;
    end
    else if(curr_state == FILL_DATA_S) begin
        lans_1_idx <= lans_1_idx + 1;
        lans_2_idx <= lans_2_idx + 1;
        lans_3_idx <= lans_3_idx + 1;
        lans_4_idx <= lans_4_idx + 1;
    end
    else if(pooling_start) begin
        lans_1_idx <= 0;
        lans_2_idx <= 0;
        lans_3_idx <= 0;
        lans_4_idx <= 0;
    end
    else if(curr_state == READ_DATA_S && next_state == READ_DATA_S) begin
        lans_1_idx <= lans_1_idx + 1;
        lans_2_idx <= lans_2_idx + 1;
        lans_3_idx <= lans_3_idx + 1;
        lans_4_idx <= lans_4_idx + 1;
    end
    else if(curr_state == READ_DATA_S && next_state != READ_DATA_S) begin
        lans_1_idx <= lans_1_idx - 1;
        lans_2_idx <= lans_2_idx - 1;
        lans_3_idx <= lans_3_idx - 1;
        lans_4_idx <= lans_4_idx - 1;
    end
    else if(curr_state != RST_S && next_state == RST_S) begin
        lans_1_idx <= 0;
        lans_2_idx <= 0;
        lans_3_idx <= 0;
        lans_4_idx <= 0;
    end
    else if(curr_state == RST_S) begin
        lans_1_idx <= lans_1_idx + 1;
        lans_2_idx <= lans_2_idx + 1;
        lans_3_idx <= lans_3_idx + 1;
        lans_4_idx <= lans_4_idx + 1;
    end
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        max_pooling_sram_write_sel <= 0;
    end
    else if(sram_next) begin
        if(max_pooling_sram_write_sel == 2) begin
            max_pooling_sram_write_sel <= 0;
        end
        else begin
           max_pooling_sram_write_sel <= max_pooling_sram_write_sel + 1; 
        end
    end
end

generate
for (genvar i = 0; i < 3; i++) begin
    global_buffer #(
    .ADDR_BITS(7),
    .DATA_BITS(DATA_WIDTH)
    )
    lans_1_gbuff (
        .clk_i   (clk_i),
        .rst_i   (rst_i),
        .wr_en   ( ((curr_state == FILL_DATA_S) && max_pooling_sram_write_sel == i) || (curr_state == RST_S) ),
        .index   (lans_1_idx),
        .data_in ( (curr_state == RST_S) ? 0 : lans_1_in[fill_data_cnt] ),
        .data_out(lans_1_out[i])
    );
end
endgenerate

generate
for (genvar i = 0; i < 3; i++) begin
    global_buffer #(
    .ADDR_BITS(7),
    .DATA_BITS(DATA_WIDTH)
    )
    lans_2_gbuff (
        .clk_i   (clk_i),
        .rst_i   (rst_i),
        .wr_en   ( ((curr_state == FILL_DATA_S) && max_pooling_sram_write_sel == i) || (curr_state == RST_S) ),
        .index   (lans_2_idx),
        .data_in ( (curr_state == RST_S) ? 0 : lans_2_in[fill_data_cnt] ),
        .data_out(lans_2_out[i])
    );
end
endgenerate

generate
for (genvar i = 0; i < 3; i++) begin
    global_buffer #(
    .ADDR_BITS(7),
    .DATA_BITS(DATA_WIDTH)
    )
    lans_3_gbuff (
        .clk_i   (clk_i),
        .rst_i   (rst_i),
        .wr_en   ( ((curr_state == FILL_DATA_S) && max_pooling_sram_write_sel == i) || (curr_state == RST_S) ),
        .index   (lans_3_idx),
        .data_in ( (curr_state == RST_S) ? 0 : lans_3_in[fill_data_cnt] ),
        .data_out(lans_3_out[i])
    );
end
endgenerate

generate
for (genvar i = 0; i < 3; i++) begin
    global_buffer #(
    .ADDR_BITS(7),
    .DATA_BITS(DATA_WIDTH)
    )
    lans_4_gbuff (
        .clk_i   (clk_i),
        .rst_i   (rst_i),
        .wr_en   ( ((curr_state == FILL_DATA_S) && max_pooling_sram_write_sel == i) || (curr_state == RST_S) ),
        .index   (lans_4_idx),
        .data_in ( (curr_state == RST_S) ? 0 : lans_4_in[fill_data_cnt] ),
        .data_out(lans_4_out[i])
    );
end
endgenerate

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        read_out_cnt <= 0;
    end
    else if(curr_state == READ_DATA_S) begin
        read_out_cnt <= read_out_cnt + 1;
    end
    else begin
        read_out_cnt <= 0;
    end
end

/*
 * store lans output
 */
always_ff @( posedge clk_i ) begin
    if(curr_state == READ_DATA_S && read_out_cnt == 1) begin
        {lans_1_data[0], lans_1_data[3], lans_1_data[6]} <= {lans_1_out[0], lans_1_out[1], lans_1_out[2]};
        {lans_2_data[0], lans_2_data[3], lans_2_data[6]} <= {lans_2_out[0], lans_2_out[1], lans_2_out[2]};
        {lans_3_data[0], lans_3_data[3], lans_3_data[6]} <= {lans_3_out[0], lans_3_out[1], lans_3_out[2]};
        {lans_4_data[0], lans_4_data[3], lans_4_data[6]} <= {lans_4_out[0], lans_4_out[1], lans_4_out[2]};
    end
    else if(curr_state == READ_DATA_S && read_out_cnt == 2) begin
        {lans_1_data[1], lans_1_data[4], lans_1_data[7]} <= {lans_1_out[0], lans_1_out[1], lans_1_out[2]};
        {lans_2_data[1], lans_2_data[4], lans_2_data[7]} <= {lans_2_out[0], lans_2_out[1], lans_2_out[2]};
        {lans_3_data[1], lans_3_data[4], lans_3_data[7]} <= {lans_3_out[0], lans_3_out[1], lans_3_out[2]};
        {lans_4_data[1], lans_4_data[4], lans_4_data[7]} <= {lans_4_out[0], lans_4_out[1], lans_4_out[2]};
    end
    else if(curr_state == READ_DATA_S && read_out_cnt == 3) begin
        {lans_1_data[2], lans_1_data[5], lans_1_data[15]} <= {lans_1_out[0], lans_1_out[1], lans_1_out[2]};
        {lans_2_data[2], lans_2_data[5], lans_2_data[15]} <= {lans_2_out[0], lans_2_out[1], lans_2_out[2]};
        {lans_3_data[2], lans_3_data[5], lans_3_data[15]} <= {lans_3_out[0], lans_3_out[1], lans_3_out[2]};
        {lans_4_data[2], lans_4_data[5], lans_4_data[15]} <= {lans_4_out[0], lans_4_out[1], lans_4_out[2]};
    end

    // lans 1
    for(int i = 0; i < 7; i++) begin
        if(lans_1_cmp_result_valid_r[i]) begin
            lans_1_data[i+8]  <= (lans_1_cmp_result[i]==1) ? lans_1_data[i*2] 
                                                           : lans_1_data[i*2+1];
        end
    end
    // lans 2
    for(int i = 0; i < 7; i++) begin
        if(lans_2_cmp_result_valid_r[i]) begin
            lans_2_data[i+8]  <= (lans_2_cmp_result[i]==1) ? lans_2_data[i*2] 
                                                           : lans_2_data[i*2+1];
        end
    end
    // lans 3
    for(int i = 0; i < 7; i++) begin
        if(lans_3_cmp_result_valid_r[i]) begin
            lans_3_data[i+8]  <= (lans_3_cmp_result[i]==1) ? lans_3_data[i*2] 
                                                           : lans_3_data[i*2+1];
        end
    end
    // lans 4
    for(int i = 0; i < 7; i++) begin
        if(lans_4_cmp_result_valid_r[i]) begin
            lans_4_data[i+8]  <= (lans_4_cmp_result[i]==1) ? lans_4_data[i*2] 
                                                           : lans_4_data[i*2+1];
        end
    end

end

/*
 * each lans use 8 comparators
 * total 8x4 = 32 comparators
 */
always_ff @( posedge clk_i ) begin
    for (int i = 0; i < 4; i++) begin
        if(curr_state == START_MAX_POOLING_S) begin
            lans_1_cmp_in_valid[i] <= 1;
            lans_2_cmp_in_valid[i] <= 1;
            lans_3_cmp_in_valid[i] <= 1;
            lans_4_cmp_in_valid[i] <= 1;
        end
        else begin
            lans_1_cmp_in_valid[i] <= 0;
            lans_2_cmp_in_valid[i] <= 0;
            lans_3_cmp_in_valid[i] <= 0;
            lans_4_cmp_in_valid[i] <= 0;
        end 
    end
    // lans 1
    lans_1_cmp_in_valid[4]  <= lans_1_cmp_result_valid_r[0];
    lans_1_cmp_in_valid[5]  <= lans_1_cmp_result_valid_r[2];
    lans_1_cmp_in_valid[6]  <= lans_1_cmp_result_valid_r[4];
    lans_1_cmp_in_valid[7]  <= lans_1_cmp_result_valid_r[6];
    // lans 2
    lans_2_cmp_in_valid[4]  <= lans_2_cmp_result_valid_r[0];
    lans_2_cmp_in_valid[5]  <= lans_2_cmp_result_valid_r[2];
    lans_2_cmp_in_valid[6]  <= lans_2_cmp_result_valid_r[4];
    lans_2_cmp_in_valid[7]  <= lans_2_cmp_result_valid_r[6];
    // lans 3
    lans_3_cmp_in_valid[4]  <= lans_3_cmp_result_valid_r[0];
    lans_3_cmp_in_valid[5]  <= lans_3_cmp_result_valid_r[2];
    lans_3_cmp_in_valid[6]  <= lans_3_cmp_result_valid_r[4];
    lans_3_cmp_in_valid[7]  <= lans_3_cmp_result_valid_r[6];
    // lans 4
    lans_4_cmp_in_valid[4]  <= lans_4_cmp_result_valid_r[0];
    lans_4_cmp_in_valid[5]  <= lans_4_cmp_result_valid_r[2];
    lans_4_cmp_in_valid[6]  <= lans_4_cmp_result_valid_r[4];
    lans_4_cmp_in_valid[7]  <= lans_4_cmp_result_valid_r[6];
end

always_ff @( posedge clk_i ) begin
    lans_1_cmp_result_valid_r <= lans_1_cmp_result_valid;
    lans_2_cmp_result_valid_r <= lans_2_cmp_result_valid;
    lans_3_cmp_result_valid_r <= lans_3_cmp_result_valid;
    lans_4_cmp_result_valid_r <= lans_4_cmp_result_valid;
end

always_ff @( posedge clk_i ) begin
    if(rst_i) begin
        pooling_result_cnt <= 0;
    end
    else if(lans_1_cmp_result_valid_r[7]) begin
        pooling_result_cnt <= pooling_result_cnt + 1;
    end
    
end


// lans 1
// mid
generate
for (genvar i = 0; i < 7; i++) begin
    floating_point_cmp cmp_1(
        .aclk(clk_i),
        .s_axis_a_tdata(lans_1_data[i*2]),
        .s_axis_a_tvalid(lans_1_cmp_in_valid[i]),
        .s_axis_b_tdata(lans_1_data[i*2+1]),
        .s_axis_b_tvalid(lans_1_cmp_in_valid[i]),
        .m_axis_result_tdata(lans_1_cmp_result[i]),
        .m_axis_result_tvalid(lans_1_cmp_result_valid[i])
    );
end
endgenerate
// last
floating_point_cmp cmp_1_last(
        .aclk(clk_i),
        .s_axis_a_tdata(lans_1_data[14]),
        .s_axis_a_tvalid(lans_1_cmp_in_valid[7]),
        .s_axis_b_tdata(lans_1_data[15]),
        .s_axis_b_tvalid(lans_1_cmp_in_valid[7]),
        .m_axis_result_tdata(lans_1_cmp_result[7]),
        .m_axis_result_tvalid(lans_1_cmp_result_valid[7])
    );

always_ff @( posedge clk_i ) begin
    if(lans_1_cmp_result_valid_r[7]) begin
        lans_1_max_val[pooling_result_cnt] <= (lans_1_cmp_result[7]==1) ? lans_1_data[14] 
                                                     : lans_1_data[15];
    end
end

// lans 2
// mid
generate
for (genvar i = 0; i < 7; i++) begin
    floating_point_cmp cmp_2(
        .aclk(clk_i),
        .s_axis_a_tdata(lans_2_data[i*2]),
        .s_axis_a_tvalid(lans_2_cmp_in_valid[i]),
        .s_axis_b_tdata(lans_2_data[i*2+1]),
        .s_axis_b_tvalid(lans_2_cmp_in_valid[i]),
        .m_axis_result_tdata(lans_2_cmp_result[i]),
        .m_axis_result_tvalid(lans_2_cmp_result_valid[i])
    );
end
endgenerate
// last
floating_point_cmp cmp_2_last(
        .aclk(clk_i),
        .s_axis_a_tdata(lans_2_data[14]),
        .s_axis_a_tvalid(lans_2_cmp_in_valid[7]),
        .s_axis_b_tdata(lans_2_data[15]),
        .s_axis_b_tvalid(lans_2_cmp_in_valid[7]),
        .m_axis_result_tdata(lans_2_cmp_result[7]),
        .m_axis_result_tvalid(lans_2_cmp_result_valid[7])
    );

always_ff @( posedge clk_i ) begin
    if(lans_2_cmp_result_valid_r[7]) begin
        lans_2_max_val[pooling_result_cnt] <= (lans_2_cmp_result[7]==1) ? lans_2_data[14] 
                                                     : lans_2_data[15];
    end
end

// lans 3
// mid
generate
for (genvar i = 0; i < 7; i++) begin
    floating_point_cmp cmp_3(
        .aclk(clk_i),
        .s_axis_a_tdata(lans_3_data[i*2]),
        .s_axis_a_tvalid(lans_3_cmp_in_valid[i]),
        .s_axis_b_tdata(lans_3_data[i*2+1]),
        .s_axis_b_tvalid(lans_3_cmp_in_valid[i]),
        .m_axis_result_tdata(lans_3_cmp_result[i]),
        .m_axis_result_tvalid(lans_3_cmp_result_valid[i])
    );
end
endgenerate
// last
floating_point_cmp cmp_3_last(
        .aclk(clk_i),
        .s_axis_a_tdata(lans_3_data[14]),
        .s_axis_a_tvalid(lans_3_cmp_in_valid[7]),
        .s_axis_b_tdata(lans_3_data[15]),
        .s_axis_b_tvalid(lans_3_cmp_in_valid[7]),
        .m_axis_result_tdata(lans_3_cmp_result[7]),
        .m_axis_result_tvalid(lans_3_cmp_result_valid[7])
    );

always_ff @( posedge clk_i ) begin
    if(lans_3_cmp_result_valid_r[7]) begin
        lans_3_max_val[pooling_result_cnt] <= (lans_3_cmp_result[7]==1) ? lans_3_data[14] 
                                                     : lans_3_data[15];
    end
end

// lans 4
// mid
generate
for (genvar i = 0; i < 7; i++) begin
    floating_point_cmp cmp_4(
        .aclk(clk_i),
        .s_axis_a_tdata(lans_4_data[i*2]),
        .s_axis_a_tvalid(lans_4_cmp_in_valid[i]),
        .s_axis_b_tdata(lans_4_data[i*2+1]),
        .s_axis_b_tvalid(lans_4_cmp_in_valid[i]),
        .m_axis_result_tdata(lans_4_cmp_result[i]),
        .m_axis_result_tvalid(lans_4_cmp_result_valid[i])
    );
end
endgenerate
// last
floating_point_cmp cmp_4_last(
        .aclk(clk_i),
        .s_axis_a_tdata(lans_4_data[14]),
        .s_axis_a_tvalid(lans_4_cmp_in_valid[7]),
        .s_axis_b_tdata(lans_4_data[15]),
        .s_axis_b_tvalid(lans_4_cmp_in_valid[7]),
        .m_axis_result_tdata(lans_4_cmp_result[7]),
        .m_axis_result_tvalid(lans_4_cmp_result_valid[7])
    );

always_ff @( posedge clk_i ) begin
    if(lans_4_cmp_result_valid_r[7]) begin
        lans_4_max_val[pooling_result_cnt] <= (lans_4_cmp_result[7]==1) ? lans_4_data[14] 
                                                     : lans_4_data[15];
    end
end

endmodule