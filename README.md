# EC551 project
1.  BVB.zip include the Bank Vector Buffer(BVB) module
This module include 32 Block RAMs. The vector we want to process is split up and stored in those 32 BRAMs. The index of the vector values are equal to the least 5 significant bits of the each BRAM's index. The inputs of the module is column ids from matrix fetcher. If there are n channels, it would be a n*32 crossbar. We use the column id as index to find the vector values we want. For each column id input, we first use the last 5 least significant bits of it to find the BRAM index we want to search within. Then we use the column id to find the vector value in this BRAM. The outputs, vector values, are sent into FIFOs and transmitted to the accumulators at last.

P.S The BVB module can process data from all channels in parallel. It can output data to all channels  

2.  BVB_Simple.zip is just a simple version of the Bank Vector Buffer.
This one can't output data into multiple channels in one clock cycle. 

3.  CISR.zip send row index to the   
All modules are parameterized.
Image.zip is the VGA controller.


You must generate the IP CORES by yourself.  
The name of the ip core you generate should be the same with the one called in the module!!!!

There is a parameter definitions.vh file in the zip files. Almost all the parameters I called are from this file.

For instance, the parameter row_id_bits is the row ID number bit length.  Parametermult_out_bits is the bit length of the product of vector value and matrix value. matrix_val_bits and vec_val_bits are the bit lengths of one single matrix and vector element value.
