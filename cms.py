from pinecone import Pinecone
from openai import OpenAI
import os, numpy as np
import json, uuid, ast, datetime, time
from prettytable import from_csv, PrettyTable

# embedding model
embedding_model = "text-embedding-ada-002"
#embedding_model = "text-embedding-3-large"

## github
vec_dim = 1536
CSIZE = 2000
SOURCE = "./docs"
TARGET = "./chunks"
KBC = "kbc.csv"
DOC = "osa_policies_forms.txt"

# id,file,old,refd,idx,wc,vid,refl,reft,date,dt
i_id    = 0
i_file  = 1
i_old   = 2
i_refd  = 3
i_idx   = 4
i_wc    = 5
i_vid   = 6
i_refl  = 7
i_reft  = 8
i_date  = 9
i_dt    = 10

# pinecone api key
pc_api_key = os.environ["PCNOS"]
#env = "gcp-starter"
#index_name = "aub-v3l-3072"
index_name = "a3-ada-1536"

# connect to pinecone database
pc = Pinecone(api_key=pc_api_key)

# connect to pinecone index
index = pc.Index(index_name)

#print(pinecone.describe_index(index_name))
print(index.describe_index_stats())

# connect to openAI using api_key
client = OpenAI(
   api_key=os.environ["OPENAI"],
 )

# -------------------------------------------------

# returns today's date for vector date stamp
def get_date():
  today = datetime.date.today()
  date_str = today.strftime("%y%m%d")
  return int(date_str)

# returns current datetime for vector datetime stamp
def get_datetime():
  dt = datetime.datetime.now()
  dt_str = dt.strftime ("%y%m%d%H%M%S")
  return int(dt_str)

# returns an almost unique id
def uid():
  uuid_str = str (uuid.uuid4 ())
  uuid_str = uuid_str.replace("-", "")
  return uuid_str

# read all vectors in db
def read_all(index):
    vec = np.ones(vec_dim)
    vec_list = vec.tolist()

    # returns all vectors excluding values
    r = index.query(vector=vec_list,
                    top_k=20,
                    include_metadata=True,
                    include_values=False)
    return r

# read all vectors in db
def del_all(r):
    list = r["matches"]

    for item in list:
        print (f"deleting vec id: {item['id']}")
        index.delete([item['id']])

#upsert semantic vector in Pinecone
def add_vec(di):
  name = di["name"]
  print(f"adding vector for {name}")
  res = client.embeddings.create(
                input=json.dumps(di["text"]),
                model=embedding_model
              )
  index.upsert([(di["id"], res.data[0].embedding, di),])
  return "success"

# partition document in sourse into chunks inserted in target
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
            output_file = target + "/" + doc[:-4] + "#" + str(i) + ".txt"
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
    ck_st = "[" + ck_st + "]"
    ck_di = ast.literal_eval(ck_st)
    return ck_di

# -----------------------------------------------------
# chunking and upserting
def part_and_ups():
  print(DOC)

  di = partition_document(SOURCE, DOC, TARGET)
  print(len(di))
  for i in range(0, len(di)):
      # insert object at index i
      res = add_vec(di[i])
      print(str(res) + ": " + str(i))
      #time.sleep(22)  

# extract doc and ck_id
def extract_strings(file_name):
    # Split the file name by "#"
    parts = file_name.split("#")

    # Extract the string before the "#"
    string_before = parts[0]

    # Extract the integer string between "#" and "."
    integer_string = parts[1].split(".")[0]

    # Return a list of the two strings
    return [string_before, integer_string]

#rename files in dir
def rename_files(dir):
  print (f"renaming files in {dir}")
  for file in os.listdir(dir):
    pair = extract_strings(file)
    doc = pair[0]
    idx = pair[1].zfill(4)
    nfile = doc + "#" + idx + ".txt"
    #print(f"file: {file}\t\t nfile: {nfile}")
    os.rename(os.path.join(dir, file), os.path.join(dir, nfile))

# count the words in a file
def count_words(file_name):
    with open(file_name, "r") as file:
        content = file.read()
        # Split the content by whitespace characters
        words = content.split()
        return len(words)

# creating the chunks knowledge base
def create_kbc():
  print("creating kbc")
  kb = KBC
  
  with open(kb) as fp:
          mtb = from_csv(fp)
        
  fnames = mtb.field_names
  if fnames[0] == "id":
    mtb.del_column("id")
  
  #print(mtb.field_names)
  #print(mtb)
  
  # Loop through the files in the directory
  for file in os.listdir(TARGET):
    #print(file)
    pair = extract_strings(file)
    old = 'y'
    refd = pair[0] + ".txt"
    idx = pair[1]
    wc = count_words(os.path.join(TARGET, file))
    vid="abc123"
    refl="https://aub.edu.lb"
    reft="text"
    date=140224
    dt=11140224
    lst=[file, old, refd, idx, wc, vid, refl, reft, date, dt]
    #print(f"row: {lst}")
    mtb.add_row(lst)
  
  mtb.sortby = "file"
  #mtb.rows.sort(key=lambda x: x[0])
  mtb.add_autoindex("id")
  #print(mtb)
  
  table_txt = mtb.get_csv_string()
  # Open a file in the current directory in write mode
  with open(kb, "w") as fp:
      # Write the table string to the file
      fp.write(table_txt)

# process kbc id
def process_kbc_id():
  print("processing kbc id")
  with open(KBC) as fp:
          mpt = from_csv(fp)
        
  fnames = mpt.field_names 
  if fnames[0] == "id":
    mpt.del_column("id")
    mpt.add_autoindex("id")

  mpt_txt = mpt.get_csv_string() 
  with open(KBC, "w") as fp:
    fp.write(mpt_txt)

  #print(mpt)
  
# update a field value in a PT row
def update_pt_field(mpt, ridx, fname, val):
  print(f"updating row at {ridx} field \"{fname}\" to {val}")
  
  trows = mpt.rows
  fnames = mpt.field_names
  fname = "City Name"

  # find fidx
  fidx = 0
  for field in fnames:
    #print (f"field = {field}")
    if field == fname:
      break
    fidx += 1
  print (f"fidx = {fidx}")

  if ridx < len(trows) and fidx < len(fname):
    trows[ridx][fidx] = val

# read a file into a string
def read_chunk(file_name):
  file_name = os.path.join(TARGET, file_name)
  #print(f"reading chunk: {file_name}")
  with open(file_name, "r") as file:
    content = file.read()
    return content

# compiling the vector knowledge base 
def process_kbv():
  print("processing knowledge base vectors")
  # id,file,old,refd,idx,wc,vid,refl,reft,date,dt
  
  with open(KBC) as fp:
    mpt = from_csv(fp)
  
  trows = mpt.rows
  for tr in trows:
    if tr[i_old] == "y":
      tr[i_old] = "n"
      chunk = read_chunk(tr[i_file])
      if tr[i_vid] == "abc123":
        tr[i_vid] = uid()
      tr[i_date] = get_date()
      tr[i_dt] = get_datetime()
      ck_st = f"""{{
            'id'  : '{tr[i_vid]}',
            'text': '''{chunk}''',
            'name': '{tr[i_file]}',
            'idx' : {int(tr[i_idx])},
            'refd': '{tr[i_refd]}',
            'refl': '{tr[i_refl]}',
            'reft': 'tr[i_reft]',
            'date': {tr[i_date]},
            'dt'  : {tr[i_dt]},
      }}"""
      ck_di = ast.literal_eval(ck_st)
      #print(ck_di["id"])
      status = add_vec(ck_di)
    #print(tr)
  
  mpt_txt = mpt.get_csv_string() 
  with open(KBC, "w") as fp:
    fp.write(mpt_txt)
  #print(mpt)
  
# -------------------------------------------------------

process_kbv()

# res = read_chunk("Contacts#0.txt")
# #print(res)
# #create_kbc()
# output_file = "Contacts#0.txt"
# i = 0
# doc = "Contacts.txt"
# link = "https://aub.edu.lb"
# ck_st = f"""{{
#             'id'  : '{uid()}',
#             'text': '''{res}''',
#             'name': '{output_file}',
#             'idx' : {i},
#             'refd': '{doc}',
#             'refl': '{link}',
#             'reft': 'text',
#             'date': {get_date()},
#             'dt'  : {get_datetime()},
#         }}"""

# ck_di = ast.literal_eval(ck_st)
# print(ck_di["id"])
# status = add_vec(ck_di)

