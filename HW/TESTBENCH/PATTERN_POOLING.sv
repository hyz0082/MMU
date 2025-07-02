`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 02/04/2025 04:13:52 PM
// Design Name: 
// Module Name: tb
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

`define CYCLE_TIME 20.0

module tb_pooling
#(parameter ACLEN  = 8,
  parameter DATA_WIDTH = 16
//   parameter CLSIZE = `CLP
) ();


/////////// System signals   ///////////////////////////////////////////////
logic                      clk_i = 0;
logic                      rst_i;

//////////// clk_i ////////////
real CYCLE;
initial CYCLE = `CYCLE_TIME;
always #(CYCLE/2.0) clk_i = ~clk_i;

logic                        tpu_cmd_valid;     // tpu valid
logic   [DATA_WIDTH-1 : 0]   tpu_param_1_in;    // data 1
logic   [DATA_WIDTH-1 : 0]   tpu_param_2_in;     // data 2

logic                      ret_valid;
logic   [DATA_WIDTH-1 : 0] ret_data_out;
logic   [DATA_WIDTH-1 : 0] ret_max_out;
logic                      tpu_busy;     // 0->idle, 1->busy


reg [127:0] GOLDEN [65535:0];

logic [31:0] A_matrix [0:288] [0:288];
logic [31:0] W_matrix [0:4] [0:2] [0:2];
logic [31:0] A_matrix_2 [0:288] [0:288];
logic [31:0] W_matrix_2 [0:4] [0:2] [0:2];
logic [31:0] C_matrix_golden [0:4] [0:288] [0:288];
logic [31:0] C_matrix [0:288] [0:288];

integer K_golden;
integer M_golden;
integer N_golden;

integer cycles;
integer total_cycles;
integer in_fd, out_fd;
integer patcount;
integer err;
integer PATNUM;
logic   [DATA_WIDTH*4-1 : 0]   tpu_data_1_in;
logic   [DATA_WIDTH*4-1 : 0]   tpu_data_2_in;
logic   [DATA_WIDTH*4-1 : 0]   tpu_data_3_in;
logic   [DATA_WIDTH*4-1 : 0]   tpu_data_4_in;


logic                           reset_lans;
logic                           set_lans_idx;
logic                           sram_next;
logic                           pooling_start;

logic                          bn_valid;
logic   [DATA_WIDTH*4-1 : 0]   bn_out_1;
logic   [DATA_WIDTH*4-1 : 0]   bn_out_2;
logic   [DATA_WIDTH*4-1 : 0]   bn_out_3;
logic   [DATA_WIDTH*4-1 : 0]   bn_out_4;

    // max pooling result
logic                          max_pooling_valid;
logic   [DATA_WIDTH*4-1 : 0]   max_pooling_out_1;
logic   [DATA_WIDTH*4-1 : 0]   max_pooling_out_2;
logic   [DATA_WIDTH*4-1 : 0]   max_pooling_out_3;
logic   [DATA_WIDTH*4-1 : 0]   max_pooling_out_4;
    
logic                          busy;     // 0->idle, 1->busy

POOLING p1
(
    clk_i, 
    rst_i,
    reset_lans,
    set_lans_idx,
    sram_next,
    pooling_start,

    bn_valid,
    bn_out_1,
    bn_out_2,
    bn_out_3,
    bn_out_4,

    max_pooling_valid,
    max_pooling_out_1,
    max_pooling_out_2,
    max_pooling_out_3,
    max_pooling_out_4,
    
    busy     // 0->idle, 1->busy
);


integer dummy_var_for_iverilog;

initial begin

    rst_i = 1'b1;
    cycles = 0;
    total_cycles = 0;

    reset_task;
    

    @(negedge clk_i);
    set_lans_idx = 1;
    @(negedge clk_i);
    set_lans_idx = 0;
    @(negedge clk_i);

    // `ifdef VIVADO_ENV
    //     in_fd = $fopen("input.txt", "r");
    //     out_fd =  $fopen("output.txt", "r");
    // `else
    //     in_fd = $fopen("./TESTBENCH/input.txt", "r");
    //     out_fd =  $fopen("output.txt", "r");
    // `endif

    //* PATNUM
    // dummy_var_for_iverilog = $fscanf(in_fd, "%d", PATNUM);
    PATNUM = 0;
    // set_bn_cmd;
    set_pooling_cmd;
    trigger_pooling_cmd;
    wait_finished;

    // $display("got = %x", ret_max_out);
    $finish;
    @(negedge clk_i);

    @(negedge clk_i);
    tpu_cmd_valid = 1;
    tpu_param_1_in = 0;
    tpu_param_2_in = 1;
    // tpu_cmd = SW_READ_DATA;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    wait (ret_valid);
    $display("expect = %x", ret_data_out);
    @(negedge clk_i);
    $finish;
    for(patcount = 0; patcount < PATNUM; patcount = patcount + 1) begin

        //* read input
        read_KMN;
        read_A_Matrix;
        read_W_Matrix;
        read_golden;

        //* start to feed data
        repeat(3) @(negedge clk_i);

        reset_cmd_task;

        set_KMN_cmd;

        send_data;
        // 64*9 9*5 
        set_conv_cmd;

        set_idx_cmd;

        reset_preload_cmd;

        trigger_conv_cmd;

        wait_finished;

        // second round
        reset_cmd_task;

        set_preload_cmd;
        
        set_conv_cmd;

        send_data_2;
        
        trigger_conv_cmd;
        
        wait_finished;

        golden_check;
        // $finish;

        $display("\033[0;34mPASS PATTERN NO.%4d,\033[m \033[0;32m Cycles: %3d\033[m", patcount ,cycles);
        total_cycles = total_cycles + cycles;
        cycles = 0;
        repeat(5) @(negedge clk_i);
    end

    YOU_PASS_task;
    $finish;

end

task reset_task ; begin

    set_lans_idx = 0;
    pooling_start = 0;
    reset_lans = 0;
    sram_next = 0;

    #(3*`CYCLE_TIME); rst_i = 1;
    #(3*`CYCLE_TIME);

    #(`CYCLE_TIME); rst_i = 0;

    #(3*`CYCLE_TIME);

    @(negedge clk_i);
    reset_lans = 1;
    @(negedge clk_i);
    reset_lans = 0;
    #(3*`CYCLE_TIME);
    @(negedge clk_i);
    wait(busy == 0);
    @(negedge clk_i);

end endtask

task reset_cmd_task ; begin

    // #(3*`CYCLE_TIME); rst_i = 1;
    // #(3*`CYCLE_TIME);
    @(negedge clk_i);
    tpu_cmd_valid = 1;
    // tpu_cmd = RESET;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    // tpu_cmd = 0;

end endtask


// task reset_mmu ; begin

//     #(3*`CYCLE_TIME);

//     wait (!mmu_busy);
//     @(negedge clk_i);
//     mmu_cmd_valid = 1;
//     mmu_cmd = 0;
//     @(negedge clk_i);
//     mmu_cmd_valid = 0;

//     #(3*`CYCLE_TIME);

// end endtask

task read_KMN; begin
    dummy_var_for_iverilog = $fscanf(in_fd, "%h", K_golden);
    dummy_var_for_iverilog = $fscanf(in_fd, "%h", M_golden);
    dummy_var_for_iverilog = $fscanf(in_fd, "%h", N_golden);
end endtask

task read_A_Matrix; begin
    logic [31:0] rbuf;

    integer i, j;
    for(i = 0; i < 10; i++) begin
        for(j = 0; j < 10; j++) begin
            dummy_var_for_iverilog = $fscanf(in_fd, "%h", rbuf);
            // $display("A[%3d][%3d] = %x", i, j, rbuf);
            A_matrix[i][j] = rbuf;   
        end 
    end

    for(i = 0; i < 10; i++) begin
        for(j = 0; j < 10; j++) begin
            dummy_var_for_iverilog = $fscanf(in_fd, "%h", rbuf);
            A_matrix_2[i][j] = rbuf;   
        end 
    end
end endtask

task read_W_Matrix; begin
    logic [31:0] rbuf;

    integer i, j;
    for (int m = 0; m < 5; m++) begin
        for(i = 0; i < 3; i++) begin
            for(j = 0; j < 3; j++) begin
                dummy_var_for_iverilog = $fscanf(in_fd, "%h", rbuf);
                // $display("golden[%3d][%3d][%3d] = %x", m, i, j, rbuf);
                W_matrix[m][i][j] = rbuf;   
            end 
        end
    end

    for (int m = 0; m < 5; m++) begin
        for(i = 0; i < 3; i++) begin
            for(j = 0; j < 3; j++) begin
                dummy_var_for_iverilog = $fscanf(in_fd, "%h", rbuf);
                W_matrix_2[m][i][j] = rbuf;   
            end 
        end
    end
end endtask

task read_golden; begin
    logic [31:0] rbuf;

    integer i, j;
    // skip mid output
    for(int m = 0; m < 5; m++) begin
        for(i = 0; i < 8; i++) begin
            for(j = 0; j < 8; j++) begin
                dummy_var_for_iverilog = $fscanf(in_fd, "%h", rbuf);
                C_matrix_golden[m][i][j] = rbuf;   
            end 
        end
    end
    // skip mid output
    for(int m = 0; m < 5; m++) begin
        for(i = 0; i < 8; i++) begin
            for(j = 0; j < 8; j++) begin
                dummy_var_for_iverilog = $fscanf(in_fd, "%h", rbuf);
                C_matrix_golden[m][i][j] = rbuf;   
            end 
        end
    end

    for(int m = 0; m < 5; m++) begin
        for(i = 0; i < 8; i++) begin
            for(j = 0; j < 8; j++) begin
                dummy_var_for_iverilog = $fscanf(in_fd, "%h", rbuf);
                C_matrix_golden[m][i][j] = rbuf;   
            end 
        end
    end
end endtask

task send_data; begin

    integer i, j, k;
    integer c;
    k = 0;
    wait (!tpu_busy);
    // send data
    for(i = 0; i < 10; i++) begin
        for(j = 0; j < 10; j++) begin
            @(negedge clk_i);
            tpu_param_1_in = A_matrix[i][j];
            tpu_param_2_in = i * 10 + j;
            tpu_cmd_valid = 1;
            // tpu_cmd = SW_WRITE_DATA;
        end
    end
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    // send weight
    for(k = 0; k < 5; k++) begin
        for(i = 0; i < 3; i++) begin
            for(j = 0; j < 3; j++) begin
                @(negedge clk_i);
                tpu_param_1_in = W_matrix[k][i][j];
                tpu_param_2_in = k * 9 + (i * 3 + j);
                tpu_cmd_valid = 1;
                // tpu_cmd = SW_WRITE_WEIGHT;
            end
        end
    end
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    
    c = 0;
    for (i = 0; i < 8; i++) begin
        for (j = 0; j < 8; j++) begin
            for(int ki = 0; ki < 3; ki++) begin
                for(int kj = 0; kj < 3; kj++) begin
                    @(negedge clk_i);
                    tpu_param_1_in = (i + ki) * 10 + (j + kj);
                    tpu_param_2_in = c;
                    tpu_cmd_valid = 1;
                    // tpu_cmd = SW_WRITE_I;
                    c++;
                end
            end
        end
    end

    @(negedge clk_i);
    tpu_cmd_valid = 0;

end endtask

task send_data_2; begin

    integer i, j, k;
    integer c;
    k = 0;
    wait (!tpu_busy);
    // send data
    for(i = 0; i < 10; i++) begin
        for(j = 0; j < 10; j++) begin
            @(negedge clk_i);
            tpu_param_1_in = A_matrix_2[i][j];
            tpu_param_2_in = i * 10 + j;
            tpu_cmd_valid = 1;
            // tpu_cmd = SW_WRITE_DATA;
        end
    end
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    // send weight
    for(k = 0; k < 5; k++) begin
        for(i = 0; i < 3; i++) begin
            for(j = 0; j < 3; j++) begin
                @(negedge clk_i);
                tpu_param_1_in = W_matrix_2[k][i][j];
                tpu_param_2_in = k * 9 + (i * 3 + j);
                tpu_cmd_valid = 1;
                // tpu_cmd = SW_WRITE_WEIGHT;
            end
        end
    end
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    
    c = 0;
    // for (i = 0; i < 8; i++) begin
    //     for (j = 0; j < 8; j++) begin
    //         for(int ki = 0; ki < 3; ki++) begin
    //             for(int kj = 0; kj < 3; kj++) begin
    //                 @(negedge clk_i);
    //                 tpu_param_1_in = (i + ki) * 10 + (j + kj);
    //                 tpu_param_2_in = c;
    //                 tpu_cmd_valid = 1;
    //                 tpu_cmd = SW_WRITE_I;
    //                 c++;
    //             end
    //         end
    //     end
    // end

    // @(negedge clk_i);
    // tpu_cmd_valid = 0;

end endtask

task set_KMN_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_param_1_in = 0; // K
    tpu_param_2_in = 9;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_KMN;
    @(negedge clk_i);
    tpu_param_1_in = 1; // M
    tpu_param_2_in = 64;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_KMN;
    @(negedge clk_i);
    tpu_param_1_in = 2; // N
    tpu_param_2_in = 5;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_KMN;
    @(negedge clk_i);
    tpu_cmd_valid = 0;

end endtask

task set_conv_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_param_1_in = 9;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_CONV_MODE;
    @(negedge clk_i);
    tpu_cmd_valid = 0;

end endtask

task set_bn_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_param_1_in = 1;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_FIX_MAC_MODE;
    @(negedge clk_i);
    tpu_cmd_valid = 0;

end endtask

task set_pooling_cmd; begin

    @(negedge clk_i);
    bn_valid = 1;
    bn_out_1 = {16'h3c00, 16'h3c01, 16'h3c02, 16'h3c03};
    bn_out_2 = {16'h4900, 16'h4901, 16'h4902, 16'h4903};
    bn_out_3 = {16'h5640, 16'h5641, 16'h5642, 16'h5643};
    bn_out_4 = {16'h63d0, 16'h63d1, 16'h63d2, 16'h63d3};
    @(negedge clk_i);
    bn_valid = 0;
    #(7*`CYCLE_TIME);
    @(negedge clk_i);
    

end endtask

task set_idx_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_param_1_in = 0;
    tpu_param_2_in = 0;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_ROW_IDX;
    @(negedge clk_i);
    tpu_param_1_in = 1;
    tpu_param_2_in = 9;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_ROW_IDX;
    @(negedge clk_i);
    tpu_param_1_in = 2;
    tpu_param_2_in = 18;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_ROW_IDX;
    @(negedge clk_i);
    tpu_param_1_in = 3;
    tpu_param_2_in = 27;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_ROW_IDX;
    @(negedge clk_i);
    tpu_cmd_valid = 0;

end endtask

task reset_preload_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_param_1_in = 0;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_PRELOAD;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    
end endtask

task set_mul_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_param_1_in = 0;
    tpu_param_2_in = 32'h3f800000;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_MUL_VAL;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    @(negedge clk_i);
    tpu_param_1_in = 1;
    tpu_param_2_in = 32'h3f800000;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_MUL_VAL;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    @(negedge clk_i);
    tpu_param_1_in = 2;
    tpu_param_2_in = 32'h3f800000;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_MUL_VAL;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    @(negedge clk_i);
    tpu_param_1_in = 3;
    tpu_param_2_in = 32'h3f800000;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_MUL_VAL;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    
end endtask

task set_add_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_param_1_in = 0;
    tpu_param_2_in = 32'h00000000;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_ADD_VAL;
    @(negedge clk_i);
    tpu_cmd_valid = 0;

    @(negedge clk_i);
    tpu_param_1_in = 1;
    tpu_param_2_in = 32'h00000000;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_ADD_VAL;
    @(negedge clk_i);
    tpu_cmd_valid = 0;

    @(negedge clk_i);
    tpu_param_1_in = 2;
    tpu_param_2_in = 32'h00000000;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_ADD_VAL;
    @(negedge clk_i);
    tpu_cmd_valid = 0;

    @(negedge clk_i);
    tpu_param_1_in = 3;
    tpu_param_2_in = 32'h00000000;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_ADD_VAL;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    
end endtask

task sw_write_partial_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_data_1_in = {32'h4048a4f0, 32'h40000000, 32'h40777e5a, 32'h0};
    tpu_data_2_in = {32'h3f800000, 32'h0, 32'h0, 32'h0};
    tpu_data_3_in = {32'h405b4841, 32'h0, 32'h0, 32'h0};
    tpu_data_4_in = {32'h405b4841, 32'h0, 32'h0, 32'h0};
    tpu_param_2_in = 0;

    tpu_cmd_valid = 1;
    // tpu_cmd = SW_WRITE_PARTIAL;
    @(negedge clk_i);
    tpu_cmd_valid = 0;

    // @(negedge clk_i);
    // tpu_data_1_in = {32'h41a00000, 32'h41a00000, 32'h0, 32'h0};
    // tpu_param_2_in = 1;
    // tpu_cmd_valid = 1;
    // tpu_cmd = SW_WRITE_PARTIAL;
    // @(negedge clk_i);
    // tpu_cmd_valid = 0;

    // @(negedge clk_i);
    // tpu_data_1_in = {32'h40a00000, 32'h40c00000, 32'h0, 32'h0};
    // tpu_param_2_in = 2;
    // tpu_cmd_valid = 1;
    // tpu_cmd = SW_WRITE_PARTIAL;
    // @(negedge clk_i);
    // tpu_cmd_valid = 0;

    // @(negedge clk_i);
    // tpu_data_1_in = {32'h40a00000, 32'h40c00000, 32'h0, 32'h0};
    // tpu_param_2_in = 3;
    // tpu_cmd_valid = 1;
    // tpu_cmd = SW_WRITE_PARTIAL;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    @(negedge clk_i);
    @(negedge clk_i);
    @(negedge clk_i);
    
end endtask

task set_preload_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_param_1_in = 1;
    tpu_cmd_valid = 1;
    // tpu_cmd = SET_PRELOAD;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    
end endtask

task trigger_conv_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_cmd_valid = 1;
    // tpu_cmd = TRIGGER_CONV;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    
end endtask

task trigger_pooling_cmd; begin

    @(negedge clk_i);
    pooling_start = 1;
    @(negedge clk_i);
    pooling_start = 0;
    
end endtask

task wait_finished; begin

    integer i;

    cycles = 0;
    @(max_pooling_valid)
    @(negedge clk_i);
    $display("got = %x", max_pooling_out_1);
    $display("got = %x", max_pooling_out_2);
    $display("got = %x", max_pooling_out_3);
    $display("got = %x", max_pooling_out_4);

    #(10*`CYCLE_TIME);
    // wait (!busy);
    // @(negedge clk_i);


    // while(mmu_busy === 1'b1) begin
    //     // cycles = cycles + 1;
    //     // if(cycles >= 1500000) begin
    //         // exceed_1500000_cycles;
    //     // end
    //     @(negedge clk_i);
    // end

end endtask


task golden_check; begin
    integer nrow_gc;
    integer i, j;
    integer k, cnt, out, off_s;
    err = 0;
    cnt = 0;
    off_s = 0;
    // C_matrix_golden
    // logic [31:0] rbuf;
    // logic   [DATA_WIDTH-1 : 0]  out;
    for(int m = 0; m < 5; m++) begin
        off_s = (m/4)*64;
        for(i = 0; i < 8; i++) begin
            for(j = 0; j < 8; j++) begin
                @(negedge clk_i);
                tpu_cmd_valid = 1;
                tpu_param_1_in = ((m%4)*4) + cnt;
                tpu_param_2_in = off_s;
                // tpu_cmd = SW_READ_DATA;
                @(negedge clk_i);
                tpu_cmd_valid = 0;
                cnt++;
                if(cnt == 4) begin
                    cnt = 0;
                    off_s++;
                end
                wait (ret_valid);
                @(negedge clk_i);
                if(ret_data_out > C_matrix_golden[m][i][j])
                    out = ret_data_out - C_matrix_golden[m][i][j];
                else
                    out = C_matrix_golden[m][i][j] - ret_data_out;
                if(out >= 10) begin
                    $display("golden[%3d][%3d][%3d] = %x, expect = %x", m, i, j, C_matrix_golden[m][i][j], ret_data_out);
                    err++;
                end
                    
                // dummy_var_for_iverilog = $fscanf(in_fd, "%h", rbuf);
                // C_matrix_golden[m][i][j] = rbuf;   
            end 
        end
    end
    
    

    // for(i = 0;i< M_golden; i++) begin
    //     for(j = 0; j < N_golden; j++) begin
    //         if(C_matrix[i][j] !== C_matrix_golden[i][j]) begin
    //             $display("gbuff[%3d][%3d] = %x, expect = %x", i, j, C_matrix[i][j], C_matrix_golden[i][j]);
    //             err = err + 1;
    //         end 
    //     end
    // end
    if(err != 0) begin
        wrong_ans;
    end


end endtask


task wrong_ans; begin
    $display("----------------------------------------------------------------------------------------");
    $display("                             Unfortunately, your answer is wrong                        ");
    $display("----------------------------------------------------------------------------------------");
    $finish;
end endtask


task YOU_PASS_task; begin
	$display ("----------------------------------------------------------------------------------------------------------------------");
	$display ("                                                  Congratulations!                						             ");
	$display ("                                           You have passed all patterns!          						             ");
	$display ("                                           Your execution cycles = %5d cycles   						                 ", total_cycles);
	$display ("                                           Your clock period = %.1f ns        					                     ", `CYCLE_TIME);
	$display ("                                           Your total latency = %.1f ns         						                 ", total_cycles*`CYCLE_TIME);
	$display ("----------------------------------------------------------------------------------------------------------------------");
	$finish;
end endtask



endmodule
