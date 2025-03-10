//============================================================================//
// AIC2021 Project1 - TPU Design                                              //
// file: global_buffer.v                                                      //
// description: global buffer read write behavior module                      //
// authors: kaikai (deekai9139@gmail.com)                                     //
//          suhan  (jjs93126@gmail.com)                                       //
//============================================================================//
module global_buffer 
#(parameter ADDR_BITS=8, 
  parameter DATA_BITS=8
)
(
      input   logic                        clk_i, 
      input   logic                        rst_i, 
      input   logic                        wr_en, // Write enable: 1->write 0->read
      input   logic  [ADDR_BITS-1:0]       index, 
      input   logic  [DATA_BITS-1:0]       data_in, 
      output  logic  [DATA_BITS-1:0]       data_out
);

  // input clk_i;
  // input rst_i;
  // input wr_en; // Write enable: 1->write 0->read
  // input      [ADDR_BITS-1:0] index;
  // input      [DATA_BITS-1:0] data_in;
  // output reg [DATA_BITS-1:0] data_out;

  integer i;

  parameter DEPTH = 2**ADDR_BITS;

//----------------------------------------------------------------------------//
// Global buffer (Don't change the name)                                      //
//----------------------------------------------------------------------------//
  // reg [`GBUFF_ADDR_SIZE-1:0] gbuff [`WORD_SIZE-1:0];
  reg [DATA_BITS-1:0] gbuff [DEPTH-1:0];

//----------------------------------------------------------------------------//
// Global buffer read write behavior                                          //
//----------------------------------------------------------------------------//
  always @ (posedge clk_i) begin
    // if(rst_i)begin
    //   for(i=0; i<(DEPTH); i=i+1)
    //     gbuff[i] <= 'd0;
    // end
    // else begin
      if(wr_en) begin
        gbuff[index] <= data_in;
      end
      else begin
        data_out <= gbuff[index];
      end
    end
  // end

endmodule
