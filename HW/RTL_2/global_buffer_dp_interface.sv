//============================================================================
//                                         
// file:                                                
// description:              
// authors:                                     
//                                              
//============================================================================
module global_buffer_dp_if
(
    input   logic                  clk_i, 
    dual_port_sram_in_interface.slave  input_sram_if_ctrl,
    dual_port_sram_out_interface.slave input_sram_if_data
);
// `include "interface.svh"
localparam DEPTH = 2**15;
reg [16-1:0] gbuff [DEPTH-1:0];

//----------------------------------------------------------------------------//
// Global buffer read write behavior                                          //
//----------------------------------------------------------------------------//
  // port 1
  always @ (posedge clk_i) begin
      if(input_sram_if_ctrl.port_a_we) begin
        gbuff[input_sram_if_ctrl.port_a_addr] <= input_sram_if_ctrl.port_a_wdata;
      end
      else begin
        input_sram_if_data.port_a_rdata <= gbuff[input_sram_if_ctrl.port_a_addr];
      end
    end
  // port 2
  always @ (posedge clk_i) begin
    input_sram_if_data.port_b_rdata <= gbuff[input_sram_if_ctrl.port_b_addr];
  end

endmodule
