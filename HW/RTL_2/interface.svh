// dual_port_sram_interface.sv

interface dual_port_sram_in_interface();
    // -------------------------------------------------------------------------
    // Port A Signals
    // -------------------------------------------------------------------------
    logic                  port_a_we;       // Port A Write Enable (active high, 0 for read, 1 for write)
    logic [15-1:0] port_a_addr;     
    logic [16-1:0] port_a_wdata;     
    // -------------------------------------------------------------------------
    // Port B Signals
    // -------------------------------------------------------------------------
    logic [15-1:0] port_b_addr;        
    // Master Modport: For a module that controls the SRAM (e.g., CPU, accelerator)
    modport master (
        output port_a_we,
        output port_a_addr,
        output port_a_wdata,

        output port_b_addr
    );

    // Slave Modport: For the SRAM module itself
    modport slave (
        input  port_a_we,
        input  port_a_addr,
        input  port_a_wdata,

        input  port_b_addr
    );

endinterface

interface dual_port_sram_out_interface();
    // -------------------------------------------------------------------------
    // Port A Signals
    // -------------------------------------------------------------------------
    logic [16-1:0] port_a_rdata;    
    // -------------------------------------------------------------------------
    // Port B Signals
    // -------------------------------------------------------------------------    
    logic [16-1:0] port_b_rdata;    

    // Master Modport: For a module that controls the SRAM (e.g., CPU, accelerator)
    modport master (
        input  port_a_rdata,
        input  port_b_rdata
    );

    // Slave Modport: For the SRAM module itself
    modport slave (
        output port_a_rdata,
        output port_b_rdata 
    );

endinterface