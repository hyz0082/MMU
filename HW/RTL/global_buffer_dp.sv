//============================================================================//
// AIC2021 Project1 - TPU Design                                              //
// file: global_buffer.v                                                      //
// description: global buffer read write behavior module                      //
// authors: kaikai (deekai9139@gmail.com)                                     //
//          suhan  (jjs93126@gmail.com)                                       //
//============================================================================//
module global_buffer_dp
#(parameter ADDR_BITS=8, 
  parameter DATA_BITS=8
)
(
      input   logic                        clk_i, 
      input   logic                        rst_i, 
      input   logic                        wr_en_1, // Write enable: 1->write 0->read
      input   logic                        wr_en_2, // Write enable: 1->write 0->read
      
      input   logic  [ADDR_BITS-1:0]       index_1, 
      input   logic  [ADDR_BITS-1:0]       index_2, 
      
      input   logic  [DATA_BITS-1:0]       data_in_1,
      input   logic  [DATA_BITS-1:0]       data_in_2, 
      
      output  logic  [DATA_BITS-1:0]       data_out_1,
      output  logic  [DATA_BITS-1:0]       data_out_2
);

  // input clk_i;
  // input rst_i;
  // input wr_en; // Write enable: 1->write 0->read
  // input      [ADDR_BITS-1:0] index;
  // input      [DATA_BITS-1:0] data_in;
  // output reg [DATA_BITS-1:0] data_out;

  integer i;

  localparam DEPTH = 2**ADDR_BITS;

//----------------------------------------------------------------------------//
// Global buffer (Don't change the name)                                      //
//----------------------------------------------------------------------------//
  // reg [`GBUFF_ADDR_SIZE-1:0] gbuff [`WORD_SIZE-1:0];
  reg [DATA_BITS-1:0] gbuff [DEPTH-1:0];

//----------------------------------------------------------------------------//
// Global buffer read write behavior                                          //
//----------------------------------------------------------------------------//
  // port 1
  always @ (posedge clk_i) begin
      if(wr_en_1) begin
        gbuff[index_1] <= data_in_1;
      end
      else begin
        data_out_1 <= gbuff[index_1];
      end
    end
  // port 2
  always @ (posedge clk_i) begin
      if(wr_en_2) begin
        gbuff[index_2] <= data_in_2;
      end
      else begin
        data_out_2 <= gbuff[index_2];
      end
    end

endmodule
