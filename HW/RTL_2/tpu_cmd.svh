//#########################
//    TPU cmd table
//#########################
localparam  RESET             = 0;  // whole
localparam  TRIGGER_CONV      = 1;  // whole
localparam  TRIGGER_CONV_LAST = 2;  // whole
localparam  SET_MUL_VAL       = 3;  // tpu_cmd   : 7
                                   // param_1_in: column number
                                   // param_2_in: multipled value
localparam  SET_ADD_VAL       = 4;  // partial
                                   // param_1_in: column number
                                   // param_2_in: added value
localparam  SET_PE_VAL        = 5;  // partial
                                   // param_1_in: PE number
                                   // param_2_in: index
localparam  SET_CONV_MODE     = 6; // whole
localparam  SET_FIX_MAC_MODE  = 7; // whole
localparam  FORWARD           = 8;
localparam  SW_WRITE_DATA     = 9;  // tpu_cmd   : 2
                                   // param_1_in: index
                                   // param_2_in: data
localparam  SW_WRITE_WEIGHT   = 10;  // tpu_cmd   : 3
                                    // param_1_in: index
                                    // param_2_in: data

localparam  SW_WRITE_I    = 11;  // tpu_cmd   : 4
                                   // param_1_in: index
                                   // param_2_in: data
localparam  SET_ROW_IDX       = 12; // partial
                                   // param_1_in: idx (0~4)
                                   // param_2_in: value
localparam  SET_KMN           = 13; // partial
                                   // param_1_in: idx (0:K, 1:M, 2:N)
                                   // param_2_in: value
localparam  SW_READ_DATA       = 14; // whole
localparam  SET_PRELOAD        = 15; // whole
localparam  SW_WRITE_PARTIAL   = 16; // whole
localparam  TRIGGER_BN   = 17; // whole
localparam  SET_MAX_POOLING = 18; // whole
localparam  SET_DIVISOR = 19; // whole
localparam  SET_SOFTMAX = 20; // whole
localparam  TRIGGER_SOFTMAX = 21; // whole
localparam  SET_MODE = 22; // param_1: mode
                           // param_2: len
localparam  TRIGGER_ADD = 23; 
localparam  SET_RELU = 24;
localparam  SET_AVERAGE_POOLING = 25;
localparam  SET_BN_MUL_SRAM_0 = 26;
localparam  SET_BN_MUL_SRAM_1 = 27;
localparam  SET_BN_MUL_SRAM_2 = 28;
localparam  SET_BN_MUL_SRAM_3 = 29;
localparam  SET_BN_ADD_SRAM_0 = 30;
localparam  SET_BN_ADD_SRAM_1 = 31;
localparam  SET_BN_ADD_SRAM_2 = 32;
localparam  SET_BN_ADD_SRAM_3 = 33;

localparam  SET_I_OFFSET_1    = 34;
localparam  SET_I_OFFSET_2    = 35;
localparam  SET_I_OFFSET_3    = 36;
localparam  SET_I_OFFSET_4    = 37;

localparam  SET_COL_IDX       = 38; 

localparam RESET_LANS     = 39; 
localparam SET_LANS_IDX   = 40;
localparam SRAM_NEXT      = 41;
localparam POOLING_START  = 42;
localparam RESET_POOLING_IDX = 43;