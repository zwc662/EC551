
def generate_coe(path, memory, labels, radix=16, bytes_per_row=8, ):
    """
    Creates a COE file
    """
    f = open(path, 'w')

    f.write("; Storing images with 1 appended to the end\n")
    f.write("; Each image is 28*28+1=785 bytes long\n")
    f.write(";\n")
    f.write("; Labels: " + str(labels) + "\n")
    f.write(";\n")
    f.write("memory_initialization_radix = " + str(radix) + ";\n")
    f.write("memory_initialization_vector = \n")

    row_counter = 0
    for cell in memory:
        assert 0 <= cell <= 255

        # hex format
        f.write(format(cell, '02x'))
        # dec format
        # f.write(str(twos_complement(cell)) + " ")

        row_counter += 1
        if row_counter == bytes_per_row:
            f.write("\n")
            row_counter = 0





