import ast, json

CSIZE = 800
## github
def partition_document(source, doc, target):
    # open the input file and read its content
    input_file = source + "/" + doc
    with open(input_file, "r") as f:
        content = f.read().rstrip()
    
    # split the content into lines using splitlines()
    lines = content.splitlines()
    
    # initialize the output file index and the partition size
    i = 0
    size = 0
    ck_st = ""
    ck_tx = ""
    
    # loop through the lines
    for line in lines:
        # if the size is zero, create a new output file with the index i
        if size == 0:
            output_file = target + "/" + doc[:-4] + "_" + str(i) + ".txt"
            f = open(output_file, "w")
            ck_tx = ""
        
        # write the line to the output file and add its length to the size
        f.write(line + "\r")
        size += len(line)
        ck_tx = ck_tx + line + "\r"
        
        # if the size is greater than or equal to CSIZE, close the output file and reset the size
        if size >= CSIZE:
            if (ck_st != ""):
                ck_st += ", "
            ck_st = ck_st + f"""{{
                'text': '''{ck_tx}''',
                'name': '{output_file}',
                'idx' : {i},
                'refd': '{doc}',
                'refl': 'none',
                'reft': 'text',
            }}"""
            f.close()
            size = 0
            i += 1 
    
    # if there is still some content left in the last output file, close it
    if size > 0:
        if (ck_st != ""):
                ck_st += ", "
        ck_st = ck_st + f"""{{
                'text': '''{ck_tx}''',
                'name': '{output_file}',
                'idx' : {i},
                'refd': '{doc}',
                'refl': 'none',
                'reft': 'text',
            }}"""
    f.close()
    #return f"{i} chunks were inserted"
    ck_st = "[" + ck_st + "]"
    ck_di = ast.literal_eval(ck_st)
    return ck_di

SOURCE = "./docs"
TARGET = "./chunks"
DOC = "career-hub.txt"
ck_di = partition_document(SOURCE, DOC, TARGET)

#print(ck_di[2]["text"])
print(len(ck_di))