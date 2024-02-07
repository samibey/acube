from pinecone import Pinecone
from openai import OpenAI
import os, numpy as np
import json, uuid, ast, datetime, time

# embedding model
#embedding_model = "text-embedding-ada-002"
embedding_model = "text-embedding-3-large"

## github
vec_dim = 3072
CSIZE = 2000
SOURCE = "./docs"
TARGET = "./chunks"
DOC = "orientation.txt"

# pinecone api key
pc_api_key = os.environ["PCNOS"]
#env = "gcp-starter"
index_name = "aub-v3l-3072"

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
        
#upsert document in Pinecone
def add_doc(di):
  res = client.embeddings.create(
                input=json.dumps(di["text"]),
                model=embedding_model,
                dimensions=vec_dim
              )
  ds = {
        "text": di["text"],
        "name": di["name"],
        "idx" : di["idx"],
        "refd": di["refd"],
        "refl": di["refl"],
        "reft": di["reft"],
        "date": get_date(),
        "dt": get_datetime()
      }
  index.upsert([(uid(), res.data[0].embedding, ds),])
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
    ck_st = "[" + ck_st + "]"
    ck_di = ast.literal_eval(ck_st)
    return ck_di

# -------------------------------------------------

# # connect to pinecone database
# pinecone.init(
#     api_key=pc_api_key,
#     environment= env
# )

# # connect to pinecone index
# index = pinecone.Index(index_name=index_name)

#print(pinecone.describe_index(index_name))
#print(index.describe_index_stats())

# -------------------------------------------------

di = partition_document(SOURCE, DOC, TARGET)
print(len(di))
for i in range(0, len(di)):
    # insert object at index i
    res = add_doc(di[i])
    print(str(res) + ": " + str(i))
    #time.sleep(22)

