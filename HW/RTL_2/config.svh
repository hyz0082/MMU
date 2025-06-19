`define FSM_WIDTH 8 
// `CLP
`define VIVADO_ENV


`ifndef DEF_PARAM
`define DEF_PARAM

localparam TPU_CMD_ADDR  = 32'hC4000000;
localparam PARAM_1_ADDR  = 32'hC4000004;
localparam PARAM_2_ADDR  = 32'hC4000008;
localparam BUSY_ADDR     = 32'hC4000020;
localparam RET_ADDR      = 32'hC4000010;
localparam RET_MAX_POOLING_ADDR      = 32'hC4002010;
localparam RET_SOFTMAX_ADDR          = 32'hC4002020;

localparam DRAM_R_ADDR   = 32'hC4002024;
localparam DRAM_R_LENGTH = 32'hC4002028;
localparam DRAM_RW          = 32'hC400202C;
localparam DRAM_TR   = 32'hC4002030;
localparam SRAM_OFFSET   = 32'hC4002034;
localparam WRITE_DATA_TYPE_ADDR = 32'hC4002038;
localparam [31:0] DRAM_WRITE_ADDR [0:3] = {32'hC400203C,
                                           32'hC4002040,
                                           32'hC4002044,
                                           32'hC4002048};
localparam NUM_LANS_ADDR     = 32'hC400204C;
localparam DRAM_WRITE_LEN    = 32'hC4002050;
localparam TR_DRAM_W         = 32'hC4002054;
localparam OUTPUT_RECV_CNT_ADDR = 32'hC4002058;
localparam SW_DATA_ADDR =  32'hC400205C;
localparam SW_WRITE_DRAM_MODE_ADDR = 32'hC4002060;
localparam RET_AVG_POOLING_ADDR    = 32'hC4002064;

localparam GEMM_CORE_SEL_ADDR = 32'hC4002068;
localparam BUSY_ADDR_2        = 32'hC400206C;

localparam READ_OFFSET = 32'hC4002070;
localparam READ_ROUNDS = 32'hC4002074;

localparam CURRINPUTTYPE = 32'hC4002078;
localparam INPUTLENGTH = 32'hC400207C;
localparam PADDINGSIZE = 32'hC4002080;

localparam RESET_SRAM = 32'hC4002084;

localparam INPUTHEIGHT = 32'hC4002088;

localparam BOUNDARYPADDINGSIZE = 32'hC400208C;

localparam BUSY_ADDR_3        = 32'hC4002090;
localparam BUSY_ADDR_4        = 32'hC4002094;



localparam [31:0] TPU_DATA_ADDR [0:15] = {32'hC4001000, 32'hC4001100, 32'hC4001200, 32'hC4001300,
                                          32'hC4001400, 32'hC4001500, 32'hC4001600, 32'hC4001700,
                                          32'hC4001800, 32'hC4001900, 32'hC4001A00, 32'hC4001B00,
                                          32'hC4001C00, 32'hC4001D00, 32'hC4001E00, 32'hC4001F00};
                                          
                                          
`else

 

`endif 