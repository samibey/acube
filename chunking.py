def partition_document(input_file):
    # open the input file and read its content
    with open(input_file, "r") as f:
        content = f.read()
    
    # split the content into lines using splitlines()
    lines = content.splitlines()
    
    # initialize the output file index and the partition size
    i = 0
    size = 0
    
    # loop through the lines
    for line in lines:
        # if the size is zero, create a new output file with the index i
        if size == 0:
            output_file = input_file[:-4] + "_" + str(i) + ".txt"
            f = open(output_file, "w")
        
        # write the line to the output file and add its length to the size
        f.write(line + "\n")
        size += len(line)
        
        # if the size is greater than or equal to 500, close the output file and reset the size
        if size >= 1000:
            f.close()
            size = 0
            i += 1
    
    # if there is still some content left in the last output file, close it
    if size > 0:
        f.close()

partition_document("a.txt")
