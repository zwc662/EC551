
def generate_coe(path, memory, addr, radix=16, bytes_per_row=8):
    """
    Creates a COE file
    """
    f = open(path, 'w')

    f.write("; Storing NN weights and biases, along with images\n")
    f.write("; Stored sequentially as W1, b1, W2, b2, W3, b3, img1, img2, img3...\n")
    f.write("; Positions of matrices and vectors are determined during synthesys,\n")
    f.write("; and should be parametrized in the design\n")
    f.write(";\n")
    f.write("; Addresses of values, columns, lengths, biases, for the 3 layers: \n")
    f.write("; " + str(addr) + "\n")
    f.write(";\n")
    f.write("; Memory stored as one byte values, written in hex format (radix = 16)\n")
    f.write("memory_initialization_radix = " + str(radix) + ";\n")
    f.write("memory_initialization_vector = \n")

    row_counter = 0
    for cell in memory:
        assert 0 <= cell 
        # assert cell <= 255

        # hex format
        f.write(format(cell, '04x') + " ")
        # dec format
        # f.write(str(twos_complement(cell)) + " ")

        row_counter += 1
        if row_counter == bytes_per_row:
            f.write("\n")
            row_counter = 0





