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

module tb
#(parameter ACLEN  = 8,
  parameter DATA_WIDTH = 32
//   parameter CLSIZE = `CLP
) ();

localparam  RESET             = 0;  // whole
localparam  TRIGGER_CONV      = 1;  // whole
localparam  TRIGGER_CONV_LAST = 2;  // whole
localparam  SET_MUL_VAL       = 3;
localparam  SET_ADD_VAL       = 4;
localparam  SET_PE_VAL        = 5;
localparam  SET_CONV_MODE     = 6; // whole
localparam  SET_FIX_MAC_MODE  = 7; // whole
localparam  FORWARD_S         = 8;
localparam  SW_WRITE_DATA     = 9;
localparam  SW_WRITE_WEIGHT   = 10;
localparam  SW_WRITE_I        = 11;
localparam  SET_ROW_IDX       = 12;
localparam  SET_KMN           = 13;
localparam  SW_READ_DATA      = 14; // whole
localparam  SET_PRELOAD       = 15; // whole
localparam  SW_WRITE_PARTIAL  = 16; // whole
localparam  TRIGGER_BN        = 17; // whole
localparam  SET_MAX_POOLING   = 18;

/////////// System signals   ///////////////////////////////////////////////
logic                      clk_i = 0;
logic                      rst_i;

//////////// clk_i ////////////
real CYCLE;
initial CYCLE = `CYCLE_TIME;
always #(CYCLE/2.0) clk_i = ~clk_i;

logic                        tpu_cmd_valid;     // tpu valid
logic   [ACLEN-1 : 0]        tpu_cmd;
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
TPU t1
(
    clk_i, rst_i,
    tpu_cmd_valid,     // tpu valid
    tpu_cmd,           // tpu
    tpu_param_1_in,    // data 1
    tpu_param_2_in,     // data 2
    tpu_data_1_in,
    tpu_data_2_in,
    tpu_data_3_in,
    tpu_data_4_in,
    ret_valid,
    ret_data_out,
    ret_max_out,
    tpu_busy     
);


integer dummy_var_for_iverilog;

initial begin

    rst_i = 1'b1;
    cycles = 0;
    total_cycles = 0;

    tpu_cmd_valid = 0;

    reset_task;

    @(negedge clk_i);

    reset_cmd_task;

    // @(negedge clk_i);
    // mmu_cmd_valid = 0;

    `ifdef VIVADO_ENV
        in_fd = $fopen("input.txt", "r");
        out_fd =  $fopen("output.txt", "r");
    `else
        in_fd = $fopen("./TESTBENCH/input.txt", "r");
        out_fd =  $fopen("output.txt", "r");
    `endif

    //* PATNUM
    dummy_var_for_iverilog = $fscanf(in_fd, "%d", PATNUM);
    PATNUM = 0;
    set_bn_cmd;
    set_max_pooling_cmd;
    set_mul_cmd; //10
    set_add_cmd; //10
    sw_write_partial_cmd;
    trigger_bn_cmd;
    wait_finished;

    // @(negedge clk_i);
    // tpu_cmd_valid = 1;
    // tpu_param_1_in = 0;
    // tpu_param_2_in = 0;
    // tpu_cmd = SW_READ_DATA;
    // @(negedge clk_i);
    // tpu_cmd_valid = 0;
    // wait (ret_valid);
    $display("got = %x", ret_max_out);
    $finish;
    @(negedge clk_i);

    @(negedge clk_i);
    tpu_cmd_valid = 1;
    tpu_param_1_in = 0;
    tpu_param_2_in = 1;
    tpu_cmd = SW_READ_DATA;
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

    #(3*`CYCLE_TIME); rst_i = 1;
    #(3*`CYCLE_TIME);

    if(tpu_busy !== 1'b0) begin
        $display("----------------------------------------------------------------");
        $display("                        Reset failed!                           ");
        $display("         Output signal should be 0 after initial RESET at %8t   ", $time);
        $display("----------------------------------------------------------------");
        #(100);
        $finish;
    end

    #(`CYCLE_TIME); rst_i = 0;

end endtask

task reset_cmd_task ; begin

    // #(3*`CYCLE_TIME); rst_i = 1;
    // #(3*`CYCLE_TIME);
    @(negedge clk_i);
    tpu_cmd_valid = 1;
    tpu_cmd = RESET;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    tpu_cmd = 0;

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
            tpu_cmd = SW_WRITE_DATA;
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
                tpu_cmd = SW_WRITE_WEIGHT;
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
                    tpu_cmd = SW_WRITE_I;
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
            tpu_cmd = SW_WRITE_DATA;
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
                tpu_cmd = SW_WRITE_WEIGHT;
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
    tpu_cmd = SET_KMN;
    @(negedge clk_i);
    tpu_param_1_in = 1; // M
    tpu_param_2_in = 64;
    tpu_cmd_valid = 1;
    tpu_cmd = SET_KMN;
    @(negedge clk_i);
    tpu_param_1_in = 2; // N
    tpu_param_2_in = 5;
    tpu_cmd_valid = 1;
    tpu_cmd = SET_KMN;
    @(negedge clk_i);
    tpu_cmd_valid = 0;

end endtask

task set_conv_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_param_1_in = 9;
    tpu_cmd_valid = 1;
    tpu_cmd = SET_CONV_MODE;
    @(negedge clk_i);
    tpu_cmd_valid = 0;

end endtask

task set_bn_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_param_1_in = 1;
    tpu_cmd_valid = 1;
    tpu_cmd = SET_FIX_MAC_MODE;
    @(negedge clk_i);
    tpu_cmd_valid = 0;

end endtask

task set_max_pooling_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_cmd_valid = 1;
    tpu_cmd = SET_MAX_POOLING;
    @(negedge clk_i);
    tpu_cmd_valid = 0;

end endtask

task set_idx_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_param_1_in = 0;
    tpu_param_2_in = 0;
    tpu_cmd_valid = 1;
    tpu_cmd = SET_ROW_IDX;
    @(negedge clk_i);
    tpu_param_1_in = 1;
    tpu_param_2_in = 9;
    tpu_cmd_valid = 1;
    tpu_cmd = SET_ROW_IDX;
    @(negedge clk_i);
    tpu_param_1_in = 2;
    tpu_param_2_in = 18;
    tpu_cmd_valid = 1;
    tpu_cmd = SET_ROW_IDX;
    @(negedge clk_i);
    tpu_param_1_in = 3;
    tpu_param_2_in = 27;
    tpu_cmd_valid = 1;
    tpu_cmd = SET_ROW_IDX;
    @(negedge clk_i);
    tpu_cmd_valid = 0;

end endtask

task reset_preload_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_param_1_in = 0;
    tpu_cmd_valid = 1;
    tpu_cmd = SET_PRELOAD;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    
end endtask

task set_mul_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_param_1_in = 0;
    tpu_param_2_in = 32'h3f800000;
    tpu_cmd_valid = 1;
    tpu_cmd = SET_MUL_VAL;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    @(negedge clk_i);
    tpu_param_1_in = 1;
    tpu_param_2_in = 32'h3f800000;
    tpu_cmd_valid = 1;
    tpu_cmd = SET_MUL_VAL;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    @(negedge clk_i);
    tpu_param_1_in = 2;
    tpu_param_2_in = 32'h3f800000;
    tpu_cmd_valid = 1;
    tpu_cmd = SET_MUL_VAL;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    @(negedge clk_i);
    tpu_param_1_in = 3;
    tpu_param_2_in = 32'h3f800000;
    tpu_cmd_valid = 1;
    tpu_cmd = SET_MUL_VAL;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    
end endtask

task set_add_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_param_1_in = 0;
    tpu_param_2_in = 32'h00000000;
    tpu_cmd_valid = 1;
    tpu_cmd = SET_ADD_VAL;
    @(negedge clk_i);
    tpu_cmd_valid = 0;

    @(negedge clk_i);
    tpu_param_1_in = 1;
    tpu_param_2_in = 32'h00000000;
    tpu_cmd_valid = 1;
    tpu_cmd = SET_ADD_VAL;
    @(negedge clk_i);
    tpu_cmd_valid = 0;

    @(negedge clk_i);
    tpu_param_1_in = 2;
    tpu_param_2_in = 32'h00000000;
    tpu_cmd_valid = 1;
    tpu_cmd = SET_ADD_VAL;
    @(negedge clk_i);
    tpu_cmd_valid = 0;

    @(negedge clk_i);
    tpu_param_1_in = 3;
    tpu_param_2_in = 32'h00000000;
    tpu_cmd_valid = 1;
    tpu_cmd = SET_ADD_VAL;
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
    tpu_cmd = SW_WRITE_PARTIAL;
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
    tpu_cmd = SET_PRELOAD;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    
end endtask

task trigger_conv_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_cmd_valid = 1;
    tpu_cmd = TRIGGER_CONV;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    
end endtask

task trigger_bn_cmd; begin

    integer i, j, k;
    @(negedge clk_i);
    tpu_cmd_valid = 1;
    tpu_cmd = TRIGGER_BN;
    @(negedge clk_i);
    tpu_cmd_valid = 0;
    
end endtask

task wait_finished; begin

    integer i;

    cycles = 0;
    wait (!tpu_busy);
    @(negedge clk_i);
    // for(i = 0; i < 7; i++) begin
    //     wait (!mmu_busy);
    //     @(negedge clk_i);
    //     data_1_in   = 0;
    //     data_2_in   = 0;
    //     data_3_in   = 0;
    //     data_4_in   = 0;
    //     weight_1_in = 0;
    //     weight_2_in = 0;
    //     weight_3_in = 0;
    //     weight_4_in = 0;
    //     mmu_cmd_valid = 1;
    //     mmu_cmd = 1;
    //     @(negedge clk_i);
    //     mmu_cmd_valid = 0;
    // end


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
                tpu_cmd = SW_READ_DATA;
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
    $display("                                       -----------                                      ");
    $display("                                     -=====,======--                                    ");
    $display("                                   ---------,-,,,,,,-                                   ");
    $display("                                 :-=====:=====:=======:                                 ");
    $display("                               :=----------------------::                               ");
    $display("                              :--------------------------::                             ");
    $display("                             :-----------------------------:                            ");
    $display("                            :-------------------------------:                           ");
    $display("                           :---------------------------------:                          ");
    $display("                          :--------------=:::==::::-----------:                         ");
    $display("                          :----------:====--,,,,--====:=------:                         ");
    $display("                         :-------:===,=++++...++++..--===:-----:                        ");
    $display("                         :-===--.......+    +.+    +......====-:                        ");
    $display("                         /==-.........+      +      +.........../                       ");
    $display("                         /-..........+     - / -     +........../                       ");
    $display("                         /...........+    -# . #-    +........../                       ");
    $display("                         /...........+       +       +........../                       ");
    $display("                         /...........+      +.+      +........../                       ");
    $display("                         /............+    +...+    +.........../                       ");
    $display("                         /.............++++.....++++.........../                        ");
    $display("                          /..................................-./                        ");
    $display("                          /.-...............................-./                         ");
    $display("                           /.-..........M##M$HM###M........-../                         ");
    $display("                            /.-..........# FAIL #$........-../                          ");
    $display("                             /.-....-......$$$$+-....-.....-/                           ");
    $display("                              /......--.........,,,,...--.//                            ");
    $display("                               //..----.............---.//                              ");
    $display("                                 ///-..-------------.-//                                ");
    $display("                                    //////////////////                                  ");
    $display("----------------------------------------------------------------------------------------");
    $display("                             Unfortunately, your answer is wrong                        ");
    $display("----------------------------------------------------------------------------------------");
    $finish;
end endtask


task YOU_PASS_task; begin
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@8OOOOOOO8@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@O               .o8@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@8:.                   .o@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@o                         :O@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                           .o8@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@888888@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@88888888OOO88@@@@@@@@@@                             :@@@@@@@");
    $display("@@@@@@@@@@@@8o:.          .o8@@@@@@@@@@@@@@@@@@@88Oo:.                      .:ooo                              o@@@@@@");
    $display("@@@@@@@@@@8                  .8@@@@@@@@@@@@8O:.           ..::::::ooo:.                                        .8@@@@@");
    $display("@@@@@@@O.                      8@@@@@8O:.        .:O88@@@@@@@@@@@@@@@@@@@@@@@88Oo.                             :8@@@@@");
    $display("@@@@@@o                        :8@@8.      .:o8@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@OO:                         o@@@@@@");
    $display("@@@@@8                          :o.     .O8@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@88@@8o.                      8@@@@@@");
    $display("@@@@:                               o8@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@8:          :OO.                  o@@@@@8@");
    $display("@@@o.                             :O@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.              OO:              :8@@@@@@@@");
    $display("@@8.                           O8@@@@@@@@@@O:.    .oO@@@@@@@@@@@@@@@@@@@@@@@.                o88          O@@@@@@@@@@@");
    $display("@@O.                         :O@@@@@@@@@@:           o@@@@@@@@@@@@@@@@@@@@@@.                 .88o.     oO@@@@@@@@@@@@");
    $display("@@O.                       :8@@@@@@@@@@8:            .O@@@@@@@@@@@@@@@@@@@@@o                  .@@8O:   o8@@@@@8@@@@@@");
    $display("@@@:                      8@@@@@@@@@@O.               :8@@@@@@@@@@@@@@@@@@@@8o                  O@@@@.    8@@@@@@@@@8@");
    $display("@@@@o                    :@@@@@@@@@@o                 :8@@@@@@@@@@@@@@@@@@@@@@o                 O@@@@O:   .O@@@@@@@@@@");
    $display("@@@@@@.                .O@@@@@@@@@@8                  O@@@@@@@@@@@@@@@@@@@@@@@@@O             .O@@@@@@@@o   :@@@@@@@@@");
    $display("@@@@@@@O:.           .O@@@@@@@@@@@@o                 .8@@@@@@@@@@@@888O8@@@@@@@@@o.         .o8@@@@@@@@@@o   o8@@@@@@@");
    $display("@@@@@@@@@@8.         o@@@@@@@@@@@@@:                 o@@@@@@@O:.         :O@@@@@@@@Oo.   .:8@@@@@@@@@@@@@8     @@@@@@@");
    $display("@@@@@@@@@@@@@@@@:    8@@@@@@@@@@@@@8               :8@@@@8:              .O@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@     @@@@@@");
    $display("@@@@@@@@@@@@@@@@    :@@@@@@@@@@@@@@@O.             8@@@@@8:              o@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@8@@@    @@@@@@");
    $display("@@@@@@@@@@@@@@@O   :@@@@@@@@@@@@@@@@@@@8O:....:O8@@@@@@@@@@@o          O@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@8@@@@    @@@@@");
    $display("@@@@@@@@@@@@@@8:  :O@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Oo.    .o8@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    @@@@");
    $display("@@@@@@@@@@@@@8:   o@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@8  :@@@@@@@@@@@@@@@@@@@@@@@8Ooo\033[0;40;31m:::::\033[0;40;37moOO8@@8OOo   o@@@");
    $display("@@@@@@@@@@@@@O   O@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@8. .8@@8o:O@@@@@@@@@@@@@8O\033[0;40;31m:::::::::::::::\033[0;40;37mO@@@O   :@@@");
    $display("@@@@@@@@@@@@@O   O@@@@@@@@@@@@@@@@@@88888@@@@@@@@@@@@@@@@@@O:oO8@8.  .:    o@@@@@@@@@@@@O\033[0;40;31m::::::::::::::::::\033[0;40;37mo8@O   :8@@");
    $display("@@@@@@@@@@@@@O   O@@@@@@@@@@@@@\033[1;40;31mO\033[0;40;31m:::::::::::::\033[0;40;37mo8@@@@@@@@@@@@8.              :@@@@@@@@@@8o\033[0;40;31m::::::::::::::::::::\033[0;40;37mo8@:   .@@");
    $display("@@@@@@@@@@@@O.  .8@@@@@@@@@@8Oo\033[0;40;31m.:::::::::::::::\033[0;40;37moO@@@@@@@@@@8:              .@@@@@@@@@@O\033[0;40;31m::::::::::::::::::::::\033[0;40;37mo8O    @@");
    $display("@@@@@@@@@@@@o   O@@@@@@@@@@8o\033[0;40;31m::::::::::::::::::::\033[0;40;37mo8@@@@@@@@@O              .@@@@@@@@@@O\033[0;40;31m::::::::::::::::::::::\033[0;40;37mo8O    @@");
    $display("@@@@@@@@@@@@O.  :8@@@@@@@@o\033[0;40;31m::::::::::::::::::::::::\033[0;40;37m8@@@@@@@@@              :@@@@@@@@@@8o\033[0;40;31m:::::::::::::::::::::\033[0;40;37mO@o    @@");
    $display("@@@@@@@@@@@@8:  :8@@@@@@@8\033[0;40;31m:::::::::::::::::::::::::\033[0;40;37m8@@@@@@@@@              O@@@@@@@@@@@O\033[0;40;31m::::::::::::::::::::\033[0;40;37mo8@:   :@@");
    $display("@@@@@@@@@@@@@O   O@@@@@@8O\033[0;40;31m:::::::::::::::::::::::::\033[0;40;37mo8@@@@@@@@O           .8@@@@@@@@@@@@@8o\033[0;40;31m::::::::::::::::\033[0;40;37mo8@@@   .O@@");
    $display("@@@@@@@@@@@@@O   O8@@@@@8O\033[0;40;31m:::::::::::::::::::::::::\033[0;40;37mo8@@@@@@@@@8:       .O@@@@@@@@@@@@@@@@@O\033[0;40;31m::::::::::::::\033[0;40;37mo@@@@8   .8@@");
    $display("@@@@@@@@@@@@@O   O@@@@@@@O\033[0;40;31m::::::::::::::::::::::::.\033[0;40;37mO8@@@@8OOooo:.     :@@@@@@@@@@@@@@@@@@@@8OOo\033[0;40;31m::::::\033[0;40;37mooO8@@@@@o   :@@@");
    $display("@@@@@@@@@@@@@8.  o8@@@@@@@\033[0;40;31m:::::::::::::::::::::::::\033[0;40;37m8@8O.                  .:O8@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@8o   o@@@@");
    $display("@@@@@@@@@@@@@8:  .O@@@@@@@O\033[0;40;31m:::::::::::::::::::::::\033[0;40;37mo@O.    .:oOOOo::.           .:OO8@@@@@@@@@@@@@@@@@@@@@@@@O.  :8@@@@");
    $display("@@@@@@@@@@@@@@8.  :8@@@@@@@8o\033[0;40;31m:::::::::::::::::::\033[0;40;37mO8@O    8@@@@@@@@@@@@@@@@@8O..         :oO8@@@@@@@@@@@@@@@8o.  .8@@@@@");
    $display("@@@@@@@@@@@@@@@O   :8@@@@@@@@8O\033[0;40;31m:::::::::::::::\033[0;40;37mO8@@@:   .@@@@@@@@@@@@@@@@@@@@@@88Oo:.       .:O8@@@@@@@@@@@.    O@@@@@@");
    $display("@@@@@@@@@@@@@@@8    O@@@@@@@@@@8Oo\033[0;40;31m::::::::\033[0;40;37mooO8@@@@@O.   O@@@@@@@@@@@@@@@@@@@@@@@@@@@@8:.      .o@@@@@@@@@o    O@@@@@@@");
    $display("@@@@@@@@@@@@@@@@o    8@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@8:    :O8@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@o.    :O@@@8o.  .o@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@:    :8@@@@@@@@@@@@@@@@@@@@@@@@@@@@@8:      ...:oO8@@@@@@@@@@@@@@@@@@@@@@@@@O:   .O8.    .O@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@O:    :@@@@@@@@@@@@@@@@@@@@@@@@@@@O.   \033[0;40;33m...\033[0;40;37m          O@@@@@@@@@@@@@@@@@@@@@@@O       .O8@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@:    :O@@@@@@@@@@@@@@@@@@@@@@@@@O   \033[0;40;33m:O888Ooo:..\033[0;40;37m    :8@@@@@@@@@@@@@@@@@@@@O:     :O@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@8o     .O8@@@@@@@@@@@@@@@@@@@@@O:  \033[0;40;33m.o8888888888O.\033[0;40;37m  .O@@@@@@@@@@@OO888@8O:.    :O@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@O        o8@@@@@@@@@@@@@@@@@@@o   \033[0;40;33m:88888888888o\033[0;40;37m   o8@@@@@@@:              o8@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@:          .:88@@@@@@@@@@@@@8:   \033[0;40;33mo8888O88888O.\033[0;40;37m  .8@@@@@@@O    \033[1;40;33m..\033[0;40;37m     .::O@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@O.                  .:o          \033[0;40;33m8888\033[0;40;37m@@@@\033[0;40;33m888o.\033[0;40;37m  o8@@@@@8o   \033[0;40;33mo88o.\033[0;40;37m   @@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@o        .OOo:.                 \033[0;40;33mO88\033[0;40;37m@@@@@\033[0;40;33m888o.\033[0;40;37m  :8@@@@@o   \033[0;40;33m:O88.\033[0;40;37m   .@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@8o         :@@@@@O:             \033[0;40;33m.O8\033[0;40;37m@@@@\033[0;40;33m8888O:\033[0;40;37m   .O88O:   \033[0;40;33m.O88O\033[0;40;37m    O@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@:                             \033[0;40;33m.o8\033[0;40;37m@@@@\033[0;40;33m\033[0;40;33m888888O:\033[0;40;37m         \033[0;40;33m.888O:\033[0;40;37m   o8@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@8o                            \033[0;40;33m.O\033[0;40;37m@@@@\033[0;40;33m\888888888Oo:...ooO8888:   \033[0;40;37m:8@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@8o                         \033[0;40;33mo8\033[0;40;37m@@@@\033[0;40;33m888888888888888888888O.\033[0;40;37m  :8@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@o.                      \033[0;40;33m.8\033[0;40;37m@@@@\033[0;40;33m888888888888888888888O:\033[0;40;37m   o@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@O:.                 \033[0;40;33m.o8\033[0;40;37m@@@@@\033[0;40;33m88888888888888888888Oo\033[0;40;37m   :8@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@8OOo::::::.   \033[0;40;33mo888\033[0;40;37m@@@@@\033[0;40;33m88888888888888888888o.\033[0;40;37m   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@8:   \033[0;40;33mo888\033[0;40;37m@@@@@\033[0;40;33m88888888888888888888.\033[0;40;37m   .@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@8:   \033[0;40;33mo888\033[0;40;37m@@@@@\033[0;40;33m88888888888888888888O\033[0;40;37m   .O@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@O.   \033[0;40;33mO8888\033[0;40;37m@@@\033[0;40;33m88888888888888888888O.\033[0;40;37m   O@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@o    \033[0;40;33m8888888888888888888888888888o\033[0;40;37m   o8@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.    \033[0;40;33m. ..:oOO8888888888888888888o.\033[0;40;37m  .8@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@O.           \033[0;40;33m..:oO8888888888888O.\033[0;40;37m  .O@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@8OO.             \033[0;40;33m.oOO88O.\033[0;40;37m   O@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@88:..          \033[0;40;33m...\033[0;40;37m    8@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@88Ooo:.          @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@8OoOO@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    $display("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
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
