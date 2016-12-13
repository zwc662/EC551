## EC551 project
FPGA-Based Sparse Matrix-Vector Multiplication Accelerator

## Getting Started
Our project is focused on  for Deep Convolutional Neural Network
Referring to several papers for reference, we implemented an sparse matrix-vector multiplication accelerator on our FPGA board. We use python to do the sparse matrix storage, CISR-encoding and neural networking training jobs. We also use verilog to construct the CISR-decoder, data channel, multiplication and other modules, which all together form the accelerator itself.

##Files included in this repository.
The files can be categorized into two parts.

  The folders comprise the python code
1. The networking_training folder include the python program of neural network training.
2. The cisr_encoder folder include the CISR encoder program, which store the sparse matrix in an efficient and economics way.
3. The vector_encoder folder transfer the matrix into vector to be transferred to the accelerator.

  The zip files include all the verilog code. Top.zip is the ultimate top module of the accelerator. The submodules are also included and introduced below:

1.  BVB.zip include the Bank Vector Buffer(BVB) module
This module include 32 Block RAMs. The vector we want to process is split up and stored in those 32 BRAMs. The index of the vector values are equal to the least 5 significant bits of the each BRAM's index. The inputs of the module is column ids from matrix fetcher. If there are n channels, it would be a n*32 crossbar. We use the column id as index to find the vector values we want. For each column id input, we first use the last 5 least significant bits of it to find the BRAM index we want to search within. Then we use the column id to find the vector value in this BRAM. The outputs, vector values, are sent into FIFOs and transmitted to the accumulators at last. This BVB module can process data from all channels in parallel. It can output data to all channels in the same clock cycle. 

2.  BVB_Simple.zip is just a simple version of the Bank Vector Buffer.
This one can't output data into multiple channels in one clock cycle. 

3.  CISR.zip send row index to the accumulator
This module get the row lengths of rows and the channel numbers as input. There are counters in the module, which count down from the row length numbers sent in down to 0. While counting down, it decode the row IDs of the rows which send the row lengths in. This uses a simple algorithm but need a complex combinational logic to finish. This module also works in parallel, which means that it sends data to all channels in every clock cycle. 

4. Channel_Accumulator.zip include the channel and accumulator
The channels include a multiplier wich do the multiplication of vector value from BVB and matrix value from matrix fetcher. Then it output the product to the accumulator. According to the multiple row IDs input from the CISR part, the accumulator accumulate the product and sum them up to find out the output vector value according to the row ID. Then output the value the output buffer.

5. FIFO.zip include the seperated channel, accumulator, and a testbench for FIFO. 
Parameterized number of channels, accumulators together form one big channel. The input and output channels become big vectors.

6. Image.zip include the VGA controller which is used to turn the output vector to a matrix and display on the monitor.
The matrix-times-vector product is a vector. We turn the vector back into matrix and store it in a block RAM. Then we take the values in the vector as pixels and output to the VGA port to display the matrix on the monitor.

7. Sparse_Storage is the sub-top module which includes the BVB_Simple, CISR, Channel_Accumulators. It's not the ultimate top module and it's just for testbench.


All modules are parameterized and all parameters in all modules are unified in one file, the definitions.vh. 
For instance, the parameter row_id_bits is the row ID number bit length.  Parametermult_out_bits is the bit length of the product of vector value and matrix value. matrix_val_bits and vec_val_bits are the bit lengths of one single matrix and vector element value.
