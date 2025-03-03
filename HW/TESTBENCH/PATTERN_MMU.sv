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
#(parameter ACLEN  = 4,
  parameter DATA_WIDTH = 32
//   parameter CLSIZE = `CLP
)
(

    );

/////////// System signals   ///////////////////////////////////////////////
logic                      clk_i = 0;
logic                      rst_i;

//////////// clk_i ////////////
real CYCLE;
initial CYCLE = `CYCLE_TIME;
always #(CYCLE/2.0) clk_i = ~clk_i;

logic                        mmu_cmd_valid; // cmd valid
logic   [ACLEN : 0]          mmu_cmd;       // cmd
logic   [DATA_WIDTH-1 : 0]   param_1_in;    // cmd ctrl
logic   [DATA_WIDTH-1 : 0]   param_2_in;    // data
/////////// MMU input   ///////////////////////////////////////////////
logic   [DATA_WIDTH-1 : 0]   data_1_in;
logic   [DATA_WIDTH-1 : 0]   data_2_in;
logic   [DATA_WIDTH-1 : 0]   data_3_in;
logic   [DATA_WIDTH-1 : 0]   data_4_in;
logic   [DATA_WIDTH-1 : 0]   weight_1_in;
logic   [DATA_WIDTH-1 : 0]   weight_2_in;
logic   [DATA_WIDTH-1 : 0]   weight_3_in;
logic   [DATA_WIDTH-1 : 0]   weight_4_in;
/////////// MMU outpupt ///////////////////////////////////////////////
logic   [DATA_WIDTH*4-1 : 0] rdata_1_out;
logic   [DATA_WIDTH*4-1 : 0] rdata_2_out;
logic   [DATA_WIDTH*4-1 : 0] rdata_3_out;
logic   [DATA_WIDTH*4-1 : 0] rdata_4_out; 

logic                        mmu_busy;     // 0 for idle, 1 for busy


reg [127:0] GOLDEN [65535:0];

logic [31:0] A_matrix [0:288] [0:288];
logic [31:0] B_matrix [0:288] [0:288];
logic [31:0] C_matrix_golden [0:288] [0:288];
logic [31:0] C_matrix [0:288] [0:288];
// logic [31:0] A_matrix [0:288] [0:288];

integer K_golden;
integer M_golden;
integer N_golden;

integer cycles;
integer total_cycles;
integer in_fd, out_fd;
integer patcount;
integer err;
integer PATNUM;

MMU m1(
    .clk_i(clk_i), 
    .rst_i(rst_i),
    .mmu_cmd_valid(mmu_cmd_valid),
    .mmu_cmd(mmu_cmd),       
    .param_1_in(param_1_in),   
    .param_2_in(param_2_in),
    .data_1_in(data_1_in),
    .data_2_in(data_2_in),
    .data_3_in(data_3_in),
    .data_4_in(data_4_in),
    .weight_1_in(weight_1_in),
    .weight_2_in(weight_2_in),
    .weight_3_in(weight_3_in),
    .weight_4_in(weight_4_in),
    .rdata_1_out(rdata_1_out),
    .rdata_2_out(rdata_2_out),
    .rdata_3_out(rdata_3_out),
    .rdata_4_out(rdata_4_out), 
    .mmu_busy(mmu_busy)
);

shortreal a = 10.66;
integer dummy_var_for_iverilog;

initial begin

    rst_i = 1'b1;
    // in_valid = 1'b0;
    cycles = 0;
    total_cycles = 0;

    mmu_cmd_valid = 0;

    // force clk_i = 0;


    reset_task;

    @(negedge clk_i);
    // #(3*`CYCLE_TIME); rst_i = 1;
    mmu_cmd = 0;
    mmu_cmd_valid = 1;
    @(negedge clk_i);
    mmu_cmd_valid = 0;

    `ifdef VIVADO_ENV
        in_fd = $fopen("input.txt", "r");
        out_fd =  $fopen("output.txt", "r");
    `else
        in_fd = $fopen("./TESTBENCH/input.txt", "r");
        out_fd =  $fopen("output.txt", "r");
    `endif

    //* PATNUM
    dummy_var_for_iverilog = $fscanf(in_fd, "%d", PATNUM);

    for(patcount = 0; patcount < PATNUM; patcount = patcount + 1) begin

        //* read input
        read_KMN;
        read_A_Matrix;
        read_B_Matrix;
        read_golden;

        //* start to feed data
        repeat(3) @(negedge clk_i);

        reset_mmu;

        send_data;

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

    // if(busy !== 1'b0) begin
    //     $display("----------------------------------------------------------------");
    //     $display("                        Reset failed!                           ");
    //     $display("         Output signal should be 0 after initial RESET at %8t   ", $time);
    //     $display("----------------------------------------------------------------");
    //     #(100);
    //     $finish;
    // end

    #(`CYCLE_TIME); rst_i = 0;
    // release clk_i;
end endtask


task reset_mmu ; begin

    #(3*`CYCLE_TIME);

    wait (!mmu_busy);
    @(negedge clk_i);
    mmu_cmd_valid = 1;
    mmu_cmd = 0;
    @(negedge clk_i);
    mmu_cmd_valid = 0;

    #(3*`CYCLE_TIME);

end endtask

task read_KMN; begin
    dummy_var_for_iverilog = $fscanf(in_fd, "%h", K_golden);
    dummy_var_for_iverilog = $fscanf(in_fd, "%h", M_golden);
    dummy_var_for_iverilog = $fscanf(in_fd, "%h", N_golden);
end endtask

task read_A_Matrix; begin
    logic [31:0] rbuf;

    integer i, j;
    for(i = 0; i < M_golden; i++) begin
        for(j = 0; j < K_golden; j++) begin
            dummy_var_for_iverilog = $fscanf(in_fd, "%h", rbuf);
            A_matrix[i][j] = rbuf;   
        end 
    end
end endtask

task read_B_Matrix; begin
    logic [31:0] rbuf;

    integer i, j;
    for(i = 0; i < K_golden; i++) begin
        for(j = 0; j < N_golden; j++) begin
            dummy_var_for_iverilog = $fscanf(in_fd, "%h", rbuf);
            B_matrix[i][j] = rbuf;   
        end 
    end
end endtask

task read_golden; begin
    logic [31:0] rbuf;

    integer i, j;
    for(i = 0; i < M_golden; i++) begin
        for(j = 0; j < N_golden; j++) begin
            dummy_var_for_iverilog = $fscanf(in_fd, "%h", rbuf);
            C_matrix_golden[i][j] = rbuf;   
        end 
    end
end endtask

task send_data; begin

    integer i, j;
    for(i = 0; i < 4; i++) begin
        wait (!mmu_busy);
        @(negedge clk_i);
        data_1_in   = A_matrix[0][i];
        data_2_in   = A_matrix[1][i];
        data_3_in   = A_matrix[2][i];
        data_4_in   = A_matrix[3][i];
        weight_1_in = B_matrix[i][0];
        weight_2_in = B_matrix[i][1];
        weight_3_in = B_matrix[i][2];
        weight_4_in = B_matrix[i][3];
        mmu_cmd_valid = 1;
        mmu_cmd = 1;
        @(negedge clk_i);
        mmu_cmd_valid = 0;
    end
end endtask

task wait_finished; begin

    integer i;

    cycles = 0;
    for(i = 0; i < 7; i++) begin
        wait (!mmu_busy);
        @(negedge clk_i);
        data_1_in   = 0;
        data_2_in   = 0;
        data_3_in   = 0;
        data_4_in   = 0;
        weight_1_in = 0;
        weight_2_in = 0;
        weight_3_in = 0;
        weight_4_in = 0;
        mmu_cmd_valid = 1;
        mmu_cmd = 1;
        @(negedge clk_i);
        mmu_cmd_valid = 0;
    end


    while(mmu_busy === 1'b1) begin
        // cycles = cycles + 1;
        // if(cycles >= 1500000) begin
            // exceed_1500000_cycles;
        // end
        @(negedge clk_i);
    end

end endtask


task golden_check; begin
    integer nrow_gc;
    integer i, j;
    err = 0;
    {C_matrix[0][0], C_matrix[1][0], C_matrix[2][0], C_matrix[3][0]} = rdata_1_out;
    {C_matrix[0][1], C_matrix[1][1], C_matrix[2][1], C_matrix[3][1]} = rdata_2_out;
    {C_matrix[0][2], C_matrix[1][2], C_matrix[2][2], C_matrix[3][2]} = rdata_3_out;
    {C_matrix[0][3], C_matrix[1][3], C_matrix[2][3], C_matrix[3][3]} = rdata_4_out;

    for(i = 0;i< M_golden; i++) begin
        for(j = 0; j < N_golden; j++) begin
            if(C_matrix[i][j] !== C_matrix_golden[i][j]) begin
                $display("gbuff[%3d][%3d] = %x, expect = %x", i, j, C_matrix[i][j], C_matrix_golden[i][j]);
                err = err + 1;
            end 
        end
    end
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
