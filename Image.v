`include "definitions.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    19:21:08 12/04/2016 
// Design Name: 
// Module Name:    Image 
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
//////////////////////////////////////////////////////////////////////////////////


module Image(
    input wire clk, 
    input wire rst,
    input wire empty, //only when en==1, the image memory read pixels
    input wire start_over,//when start_over==1, the image memory would be clear.
	 input wire [7:0] pixel, // the image memory
	 
 
    output reg [7:0] rgb, // the output rgb color
    output wire hsync, //hsync signal
    output wire vsync, //vsync signal
	 output reg rd_en // ready for next pixel
    );
	 
	 
	 reg wr_en;
	 reg [9:0] index=0;
	 reg [9:0] write_index=0;
	 reg [9:0] read_index=0;
	 wire [7:0] rgb_o;
	 reg [7:0] pixel_i=`RED;
	 integer i,j;
	 	
	 reg clk_count;
    reg clk_monitor;
	 
	 always@(posedge clk) begin
		clk_count <= ~clk_count;
      if (clk_count) begin
			clk_monitor <= ~clk_monitor;
      end
	 end
    
	 reg [9:0] counter_x = 0;
    reg [9:0] counter_y = 0;
	 always @ (posedge clk_monitor) begin
       if (counter_x >= `PIXEL_WIDTH + `HSYNC_FRONT_PORCH + `HSYNC_PULSE_WIDTH + `HSYNC_BACK_PORCH) begin
           counter_x <= 0;
           if (counter_y >= `PIXEL_HEIGHT + `VSYNC_FRONT_PORCH + `VSYNC_PULSE_WIDTH + `VSYNC_BACK_PORCH) begin
               counter_y <= 0;
           end else begin
               counter_y <= counter_y + 1;
           end
       end else begin
           counter_x <= counter_x + 1;
       end
    end
	 
    assign hsync = ~(counter_x >= (`PIXEL_WIDTH + `HSYNC_FRONT_PORCH) &&
                     counter_x < (`PIXEL_WIDTH + `HSYNC_FRONT_PORCH + `HSYNC_PULSE_WIDTH));
    assign vsync = ~(counter_y >= (`PIXEL_HEIGHT + `VSYNC_FRONT_PORCH) &&
                     counter_y < (`PIXEL_HEIGHT + `VSYNC_FRONT_PORCH + `VSYNC_PULSE_WIDTH));
	 
	 bram bram_(.addra(index),
					.dina(pixel_i),
					.wea(wr_en),
					.clka(clk),
					.douta(rgb_o)
					);
					
 
	 always@(posedge clk) begin
		if(start_over==1 || rst==1) begin
			wr_en<=0;
			rd_en<=1;
			write_index<=0;
		end else if(start_over==0 && empty==0) begin
			index<=write_index;
			if(write_index<=`BLOCKS_WIDE*`BLOCKS_HIGH-1) begin
				write_index<=write_index+1;
				pixel_i<=pixel;
				wr_en<=1;
			end else begin
				wr_en<=0;
				rd_en<=0;
			end 
		end else if(start_over==0 && empty==1) begin
			wr_en<=0;
			rd_en<=0;
			if((counter_x > (`PIXEL_WIDTH - `BLOCK_SIZE*`BLOCKS_WIDE)/2) && 
				(counter_x < (`PIXEL_WIDTH + `BLOCK_SIZE*`BLOCKS_WIDE)/2) && 
				(counter_y > (`PIXEL_HEIGHT - `BLOCK_SIZE*`BLOCKS_HIGH)/2) &&
				(counter_y < (`PIXEL_HEIGHT + `BLOCK_SIZE*`BLOCKS_HIGH)/2)) begin
				j=(counter_y-(`PIXEL_HEIGHT-`BLOCK_SIZE*`BLOCKS_HIGH)/2)/`BLOCK_SIZE;
				i=(counter_x-(`PIXEL_WIDTH-`BLOCK_SIZE*`BLOCKS_WIDE)/2)/`BLOCK_SIZE;
				index<=j*`BLOCKS_WIDE+i;
				rgb<=rgb_o;
				
			end else begin
				rgb<=`BLACK;
			end
			
		end
	end

endmodule
