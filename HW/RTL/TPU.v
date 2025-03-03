
module TPU(
    clk,
    rst_n,

    in_valid,
    K,
    M,
    N,
    busy,

    A_wr_en,
    A_index,
    A_data_in,
    A_data_out,

    B_wr_en,
    B_index,
    B_data_in,
    B_data_out,

    C_wr_en,
    C_index,
    C_data_in,
    C_data_out
);


input clk;
input rst_n;
input            in_valid;
input [7:0]      K;
input [7:0]      M;
input [7:0]      N;
output  reg      busy;

output           A_wr_en;
output [15:0]    A_index;
output [31:0]    A_data_in;
input  [31:0]    A_data_out;

output           B_wr_en;
output [15:0]    B_index;
output [31:0]    B_data_in;
input  [31:0]    B_data_out;

output           C_wr_en;
output [15:0]    C_index;
output reg [127:0]   C_data_in;
input  [127:0]   C_data_out;


parameter  IDLE       = 0;
parameter  SEND_REQ   = 1;
parameter  SA_IN      = 2;
parameter  SA_FORWARD = 3;
parameter  OUTPUT     = 4;
parameter  NEW_ROUND  = 5;

reg [3:0] curr_state, next_state;
reg [7:0] K_reg, M_reg, N_reg;

// BUFF A B C index
reg [15:0] curr_A_index, curr_B_index, curr_C_index;
reg [8:0] input_offset;
reg [8:0] cumulation;
reg [8:0] output_offset_temp, output_offset_total;
reg [8:0] b_offset_total;
//* Implement your design here

reg signed [7:0] row_1_reg;
reg signed [7:0] row_2_reg [0:1];
reg signed [7:0] row_3_reg [0:2];

reg signed [7:0] col_1_reg;
reg signed[7:0] col_2_reg [0:1];
reg signed [7:0] col_3_reg [0:2];


reg signed [7:0] data_in [0:3];
reg signed [7:0] weight_in [0:3];

wire signed [7:0] data_out [0:16];
wire signed [7:0] weight_out [0:16];

wire signed [31:0] mac_value[1:16];

reg [8:0] sa_in_cnt, sa_forward_cnt;


//************************
//         FSM
//************************
always@(posedge clk, negedge rst_n) begin
    if(~rst_n) begin
        curr_state <= IDLE;
    end
    else begin
        curr_state <= next_state;
    end
end
//************************
//   NEXT_STATE LOGIC
//************************
always@(*) begin
    case (curr_state)
    IDLE: if(in_valid) next_state = SEND_REQ;
          else         next_state = IDLE;
    SEND_REQ: next_state = SA_IN;
    SA_IN   : if(sa_in_cnt + 1 == K_reg) next_state = SA_FORWARD;
              else                       next_state = SA_IN;
    SA_FORWARD: if(sa_forward_cnt == 6)  next_state = OUTPUT;
                else                     next_state = SA_FORWARD;
    OUTPUT    :if(output_offset_temp == 3 || output_offset_total + 1  == M_reg) next_state = NEW_ROUND;
               else next_state = OUTPUT;
    NEW_ROUND : if ((b_offset_total + 4 >= N_reg) && (cumulation + 4 >= M_reg)) next_state = IDLE;
                else next_state = SEND_REQ;     
    default: next_state = IDLE;
    endcase
end

//************************
//    BUFFER A SIGNAL
//************************
assign A_wr_en = 0;
assign A_index = (curr_state == SEND_REQ || curr_state == SA_IN) 
                    ? curr_A_index + input_offset 
                    : 0;
//************************
//    BUFFER B SIGNAL
//************************
assign B_wr_en = 0;
assign B_index = (curr_state == SEND_REQ || curr_state == SA_IN) 
                   ? curr_B_index + input_offset
                   : 0;

//************************
//    BUFFER C SIGNAL
//************************
assign C_wr_en = (curr_state == OUTPUT);
assign C_index = curr_C_index;

always @(*) begin
    if(output_offset_temp == 0     ) C_data_in = {mac_value[ 1], mac_value[ 2], mac_value[ 3], mac_value[ 4]};
    else if(output_offset_temp == 1) C_data_in = {mac_value[ 5], mac_value[ 6], mac_value[ 7], mac_value[ 8]};
    else if(output_offset_temp == 2) C_data_in = {mac_value[ 9], mac_value[10], mac_value[11], mac_value[12]};
    else if(output_offset_temp == 3) C_data_in = {mac_value[13], mac_value[14], mac_value[15], mac_value[16]};
    else C_data_in = 0;
end

                                             
//************************
//     SA_IN COUNTER
//************************
always@(posedge clk, negedge rst_n) begin
    if(~rst_n) begin
        sa_in_cnt <= 0; 
    end
    else if(curr_state == OUTPUT || curr_state == IDLE) begin
        sa_in_cnt <= 0;
    end
    else if(curr_state == SA_IN) begin
        sa_in_cnt <= sa_in_cnt + 1;
    end
end
//************************
//  SA_FORWARD COUNTER
//************************
always@(posedge clk, negedge rst_n) begin
    if(~rst_n) begin
        sa_forward_cnt <= 0; 
    end
    else if(curr_state == OUTPUT || curr_state == IDLE) begin
        sa_forward_cnt <= 0;
    end
    else if(curr_state == SA_FORWARD) begin
        sa_forward_cnt <= sa_forward_cnt + 1;
    end
end


always@(posedge clk, negedge rst_n) begin
    if(~rst_n) begin
        input_offset <= 0; 
    end
    else if(curr_state == SA_IN) begin
        input_offset <= input_offset + 1;
    end
    else begin
        input_offset <= 0;
    end
end

//************************
//  SYSTOLIC ARRAY INPUT
//************************
always@(posedge clk, negedge rst_n) begin
    if(~rst_n) begin
        {data_in[0], data_in[1], data_in[2], data_in[3]} <= 0;
        {weight_in[0], weight_in[1], weight_in[2], weight_in[3]} <= 0;
    end
    else if(curr_state == SA_IN) begin
        data_in[0] <= A_data_out[31:24];
        data_in[1] <= row_1_reg;
        data_in[2] <= row_2_reg[0];
        data_in[3] <= row_3_reg[0];

        weight_in[0] <= B_data_out[31:24];
        weight_in[1] <= col_1_reg;
        weight_in[2] <= col_2_reg[0];
        weight_in[3] <= col_3_reg[0];
    end
    else if(curr_state == SA_FORWARD)begin
        data_in[0] <= 0;
        data_in[1] <= row_1_reg;
        data_in[2] <= row_2_reg[0];
        data_in[3] <= row_3_reg[0];

        weight_in[0] <= 0;
        weight_in[1] <= col_1_reg;
        weight_in[2] <= col_2_reg[0];
        weight_in[3] <= col_3_reg[0];
    end
    else begin
        {data_in[0], data_in[1], data_in[2], data_in[3]} <= 0;
        {weight_in[0], weight_in[1], weight_in[2], weight_in[3]} <= 0;
    end
end

//************************
//  BUFFER A B INDEX
//************************
always@(posedge clk, negedge rst_n) begin
    if(~rst_n) begin
        curr_A_index <= 0;
        curr_B_index <= 0;
        cumulation   <= 0;
        b_offset_total <= 0;
    end
    else if(curr_state == IDLE) begin
        curr_A_index <= 0;
        curr_B_index <= 0;
        cumulation   <= 0;
        b_offset_total <= 0;
    end
    else if(curr_state == NEW_ROUND) begin
        if(cumulation + 4 >= M_reg) begin // reset Matrix A index
            curr_A_index <= 0;        // take Matrix B next 4 cols
            cumulation   <= 0;
            curr_B_index <= curr_B_index + K_reg;
            b_offset_total <= b_offset_total + 4;
        end
        else begin // take Matrix A next 4 rows
            curr_A_index <= curr_A_index + K_reg;
            cumulation   <= cumulation + 4;
        end
    end
end

//************************
//      BUFFER C INDEX
//************************
always@(posedge clk, negedge rst_n) begin
    if(~rst_n) begin
        curr_C_index <= 0;
        output_offset_temp <= 0; 
        output_offset_total <= 0;
    end
    else if(curr_state == IDLE) begin
        curr_C_index <= 0;
        output_offset_temp <= 0; 
        output_offset_total <= 0;
    end
    else if(curr_state == OUTPUT) begin
        curr_C_index <= curr_C_index + 1;
        output_offset_temp  <= output_offset_temp + 1; 
        output_offset_total <= output_offset_total + 1;
    end
    else if(curr_state == NEW_ROUND) begin
        output_offset_temp  <= 0;
        if(output_offset_total == M_reg) begin
            output_offset_total <= 0;
        end
    end
end

//************************
//   STORE K M N VALUE
//************************
always@(posedge clk) begin
    if(in_valid) begin
        K_reg <= K;
        M_reg <= M;
        N_reg <= N;
    end
end

//************************
//  SYSTOLIC INPUT DELAY
//************************
always@(posedge clk, negedge rst_n) begin
    if(~rst_n) begin
        row_1_reg <= 0;
        row_2_reg[0] <= 0; row_2_reg[1] <= 0;
        row_3_reg[0] <= 0; row_3_reg[1] <= 0; row_3_reg[2] <= 0;

        col_1_reg <= 0;
        col_2_reg[0] <= 0; col_2_reg[1] <= 0;
        col_3_reg[0] <= 0; col_3_reg[1] <= 0; col_3_reg[2] <= 0;
    end
    else if(curr_state == SA_IN) begin
        // ROW 1
        row_1_reg    <= A_data_out[23:16];
        // ROW 2
        row_2_reg[1] <= A_data_out[15:8];
        row_2_reg[0] <= row_2_reg[1];
        // ROW 3
        row_3_reg[2] <= A_data_out[7:0];
        row_3_reg[1] <= row_3_reg[2];
        row_3_reg[0] <= row_3_reg[1]; 
        // COL 1
        col_1_reg <= B_data_out[23:16];
        // COL 2
        col_2_reg[1] <= B_data_out[15:8];
        col_2_reg[0] <= col_2_reg[1];
        // COL 3
        col_3_reg[2] <= B_data_out[7:0];
        col_3_reg[1] <= col_3_reg[2]; 
        col_3_reg[0] <= col_3_reg[1]; 
    end
    else begin
        // ROW 1
        row_1_reg    <= 0;
        // ROW 2
        row_2_reg[1] <= 0;
        row_2_reg[0] <= row_2_reg[1];
        // ROW 3
        row_3_reg[2] <= 0;
        row_3_reg[1] <= row_3_reg[2];
        row_3_reg[0] <= row_3_reg[1]; 
        // COL 1
        col_1_reg <= 0;
        // COL 2
        col_2_reg[1] <= 0;
        col_2_reg[0] <= col_2_reg[1];
        // COL 3
        col_3_reg[2] <= 0;
        col_3_reg[1] <= col_3_reg[2]; 
        col_3_reg[0] <= col_3_reg[1]; 
    end
end

//************************
//       4 X 4 PE
//************************
PE p1( .clk(clk), .rst_n(rst_n), .curr_state(curr_state),
       .data_in(data_in[0]),
       .weight_in(weight_in[0]),
       .data_out(data_out[1]),
       .weight_out(weight_out[1]),
       .mac_value(mac_value[1])
);

PE p2( .clk(clk), .rst_n(rst_n), .curr_state(curr_state),
       .data_in(data_out[1]),
       .weight_in(weight_in[1]),
       .data_out(data_out[2]),
       .weight_out(weight_out[2]),
       .mac_value(mac_value[2])
);

PE p3( .clk(clk), .rst_n(rst_n), .curr_state(curr_state),
       .data_in(data_out[2]),
       .weight_in(weight_in[2]),
       .data_out(data_out[3]),
       .weight_out(weight_out[3]),
       .mac_value(mac_value[3])
);

PE p4( .clk(clk), .rst_n(rst_n), .curr_state(curr_state),
       .data_in(data_out[3]),
       .weight_in(weight_in[3]),
       .data_out(data_out[4]),
       .weight_out(weight_out[4]),
       .mac_value(mac_value[4])
);

PE p5( .clk(clk), .rst_n(rst_n), .curr_state(curr_state),
       .data_in(data_in[1]),
       .weight_in(weight_out[1]),
       .data_out(data_out[5]),
       .weight_out(weight_out[5]),
       .mac_value(mac_value[5])
);

PE p6( .clk(clk), .rst_n(rst_n), .curr_state(curr_state),
       .data_in(data_out[5]),
       .weight_in(weight_out[2]),
       .data_out(data_out[6]),
       .weight_out(weight_out[6]),
       .mac_value(mac_value[6])
);

PE p7( .clk(clk), .rst_n(rst_n), .curr_state(curr_state),
       .data_in(data_out[6]),
       .weight_in(weight_out[3]),
       .data_out(data_out[7]),
       .weight_out(weight_out[7]),
       .mac_value(mac_value[7])
);

PE p8( .clk(clk), .rst_n(rst_n), .curr_state(curr_state),
       .data_in(data_out[7]),
       .weight_in(weight_out[4]),
       .data_out(data_out[8]),
       .weight_out(weight_out[8]),
       .mac_value(mac_value[8])
);

PE p9( .clk(clk), .rst_n(rst_n), .curr_state(curr_state),
       .data_in(data_in[2]),
       .weight_in(weight_out[5]),
       .data_out(data_out[9]),
       .weight_out(weight_out[9]),
       .mac_value(mac_value[9])
);

PE p10( .clk(clk), .rst_n(rst_n), .curr_state(curr_state),
       .data_in(data_out[9]),
       .weight_in(weight_out[6]),
       .data_out(data_out[10]),
       .weight_out(weight_out[10]),
       .mac_value(mac_value[10])
);

PE p11( .clk(clk), .rst_n(rst_n), .curr_state(curr_state),
       .data_in(data_out[10]),
       .weight_in(weight_out[7]),
       .data_out(data_out[11]),
       .weight_out(weight_out[11]),
       .mac_value(mac_value[11])
);

PE p12( .clk(clk), .rst_n(rst_n), .curr_state(curr_state),
       .data_in(data_out[11]),
       .weight_in(weight_out[8]),
       .data_out(data_out[12]),
       .weight_out(weight_out[12]),
       .mac_value(mac_value[12])
);

PE p13( .clk(clk), .rst_n(rst_n), .curr_state(curr_state),
       .data_in(data_in[3]),
       .weight_in(weight_out[9]),
       .data_out(data_out[13]),
       .weight_out(weight_out[13]),
       .mac_value(mac_value[13])
);

PE p14( .clk(clk), .rst_n(rst_n), .curr_state(curr_state),
       .data_in(data_out[13]),
       .weight_in(weight_out[10]),
       .data_out(data_out[14]),
       .weight_out(weight_out[14]),
       .mac_value(mac_value[14])
);

PE p15( .clk(clk), .rst_n(rst_n), .curr_state(curr_state),
       .data_in(data_out[14]),
       .weight_in(weight_out[11]),
       .data_out(data_out[15]),
       .weight_out(weight_out[15]),
       .mac_value(mac_value[15])
);

PE p16( .clk(clk), .rst_n(rst_n), .curr_state(curr_state),
       .data_in(data_out[15]),
       .weight_in(weight_out[12]),
       .data_out(data_out[16]),
       .weight_out(weight_out[16]),
       .mac_value(mac_value[16])
);

//************************
//    BUSY SIGNAL
//************************
always@(*) begin
    if(~rst_n) begin
        busy = 0;
    end
    else if(curr_state == IDLE) begin
        busy = 0;
    end
    else begin
        busy = 1;
    end
end


endmodule

module PE(
    clk,
    rst_n,

    curr_state,

    weight_in,
    data_in,

    weight_out,
    data_out, 
    mac_value
);


input clk;
input rst_n;
input [3:0] curr_state;

input signed [7:0]  weight_in;
input signed [7:0] data_in;

output reg signed [7:0]  weight_out;
output reg signed [7:0] data_out;
output   signed  [31:0] mac_value;

parameter  IDLE       = 0;
parameter  SEND_REQ   = 1;
parameter  SA_IN      = 2;
parameter  SA_FORWARD = 3;
parameter  OUTPUT     = 4;
parameter  NEW_ROUND  = 5;

reg signed [31:0] mac_reg;


always@(posedge clk, negedge rst_n) begin
    if(~rst_n) begin
        weight_out <= 0;
        data_out   <= 0;
    end
    else begin
        weight_out <= weight_in;
        data_out   <= data_in;
    end
end

always@(posedge clk, negedge rst_n) begin
    if(~rst_n) begin
        mac_reg <= 0;
    end
    else if(curr_state == IDLE || curr_state == NEW_ROUND) begin
        mac_reg <= 0;
    end
    else begin
        mac_reg <= mac_reg + weight_in * data_in;
    end
end

assign mac_value = mac_reg;

endmodule