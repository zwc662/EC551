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
	 input wire [7:0] pixel, // the image memory
	 
	 output wire [7:0] Led,
    output reg [7:0] rgb, // the output rgb color
    output wire hsync, //hsync signal
    output wire vsync //vsync signal
    );
	 
	 
	 reg wr_en;
	 reg [9:0] index=0;
	 wire [7:0] rgb_o;
	 integer i,j;
	 	
	 reg clk_count;
    reg clk_monitor;
	 assign Led=pixel;
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
//					.dina(pixel_i),
//					.wea(wr_en),
					.clka(clk),
					.douta(rgb_o)
					);
					
 
	 always@(posedge clk) begin
		if(rst==1) begin
			index<=0;
			rgb<=`BLACK;
		end else begin
			if((counter_x > (`PIXEL_WIDTH - `BLOCK_SIZE*`BLOCKS_WIDE)/2) && 
				(counter_x < (`PIXEL_WIDTH + `BLOCK_SIZE*`BLOCKS_WIDE)/2) && 
				(counter_y > (`PIXEL_HEIGHT - `BLOCK_SIZE*`BLOCKS_HIGH)/2) &&
				(counter_y < (`PIXEL_HEIGHT + `BLOCK_SIZE*`BLOCKS_HIGH)/2)) begin
				j=(counter_y-(`PIXEL_HEIGHT-`BLOCK_SIZE*`BLOCKS_HIGH)/2)/`BLOCK_SIZE;
				i=(counter_x-(`PIXEL_WIDTH-`BLOCK_SIZE*`BLOCKS_WIDE)/2)/`BLOCK_SIZE;
				rgb<=rgb_o;
				case(pixel) 
					8'b010: 		  index<=0;
					8'b100:       index<=1;
					8'b110:       index<=2;
					default: 	  index<=j*`BLOCKS_WIDE+i;
				endcase; 
			end else begin
				rgb<=`BLACK;
			end
			
		end
	end

endmodule
