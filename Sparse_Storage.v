`include "definitions.vh"

//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    00:38:12 12/09/2016 
// Design Name: 
// Module Name:    Big_Channel 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
/////////////////////////////////////////////////////////////
/////////////////////
module Sparse_Storage( 
	input wire clk, rst,
	
	
	output wire [`mult_bits * `channel_num -1:0] wr_data,
	output wire [`row_id_bits * `channel_num -1:0] wr_addr,
	output wire [`channel_num-1:0] wr_en
	);
	
	wire [`row_id_bits*`channel_num-1:0] row_id;
	wire [`channel_num-1:0] row_id_rd_en;
	wire [`channel_num-1:0] row_id_empty;
	
	wire [`vec_val_bits*`channel_num-1:0] vec_val;
	wire [`channel_num-1:0] vec_val_rd_en;
	wire [`channel_num-1:0] vec_val_empty;
	

	wire [`row_len_bits*`channel_num-1:0] row_len,
	wire [`channel_num-1:0] row_len_empty, 
	wire [`channel_num-1:0] row_len_rd_en,	
	
	wire [`col_id_bits*`channel_num-1 : 0] col_id,
	wire [`channel_num-1:0] col_id_empty,
	wire  [`channel_num-1:0] col_id_rd_en,	
	
	wire [`matrix_val_bits * `channel_num -1 :0] matrix_val,
	wire [`channel_num - 1 : 0] matrix_val_empty,
	wire [`channel_num - 1 : 0] matrix_val_rd_en,

	
//	wire [`col_id_bits*`channel_num-1: 0] program_addr,
//	wire [`vec_val_bits*`channel_num-1 : 0] program_data,
//	wire [`channel_num-1:0] program_wea,
	 
	assign row_len=fetcher_out[3*`row_len*`channel_num-1: 2*`row_len*`channel_num];
	assign row_len_rd_en=fetcher_out[3*`channel_num-1: 2*`channel_num];
	assign row_len_empty=fetcher_out[3*`channel_num-1: 2*`channel_num];

	
	assign col_id=fetcher_out[3*`col_id*`channel_num-1: 2*`col_id*`channel_num];
	assign col_id_rd_en=fetcher_out[3*`channel_num-1: 2*`channel_num];
	assign col_id_empty=fetcher_out[3*`channel_num-1: 2*`channel_num];

	assign matrix_val=fetcher_out[3*`matrix_val*`channel_num-1: 2*`matrix_val*`channel_num];
	assign matrix_val_rd_en=fetcher_out[3*`channel_num-1: 2*`channel_num];
	assign matrix_val_empty=fetcher_out[3*`channel_num-1: 2*`channel_num];

	


	


	Big_Channel Big_Channel_ (
		.clk(clk), 
		.rst(rst), 
		.matrix_val(matrix_val), 
		.matrix_val_empty(matrix_val_empty), 
		.matrix_val_rd_en(matrix_val_rd_en), 
		.vec_val(vec_val), 
		.vec_val_empty(vec_val_empty), 
		.vec_val_rd_en(vec_val_rd_en), 
		.row_id(row_id), 
		.row_id_empty(row_id_empty), 
		.row_id_rd_en(row_id_rd_en), 
		.wr_data(wr_data), 
		.wr_addr(wr_addr), 
		.wr_en(wr_en)
	);
	
	CISRDecoder CISRDecoder_ (
		.difference(difference), 
		.currentRowID(currentRowID), 
		.clk(clk), 
		.rst(rst), 
		.row_len(row_len), 
		.row_len_empty(row_len_empty), 
		.row_len_rd_en(row_len_rd_en), 
		.row_id_empty(row_id_empty), 
		.row_id_rd_en(row_id_rd_en), 
		.row_id(row_id)
	);

	
endmodule
