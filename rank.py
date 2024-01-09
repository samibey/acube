import pinecone, os, json, datetime, time
from openai import OpenAI
from fastapi import FastAPI
import gradio as gr

vec_dim = 1536
MAX_TOKENS = 600
TOP_K = 2
reqs = []

# pinecone api key
pc_api_key = os.environ["PINECONE"]
env = "gcp-starter"
index_name = "aub"

# connect to pinecone database
pinecone.init(
    api_key=pc_api_key,
    environment= env
)

# connect to pinecone index
index = pinecone.Index(index_name=index_name)

#print(pinecone.describe_index(index_name))
index.describe_index_stats()

# embedding model
embedding_model = "text-embedding-ada-002"

# LLM model
llm_model = "gpt-3.5-turbo-instruct"

# connect to openAI using api_key
client = OpenAI(
   api_key=os.environ["OPENAI"],
 )

def get_datetime():
  dt = datetime.datetime.now()
  #dt_str = dt.strftime ("%d%H%M%S")
  return dt

def slowit():
  dt = get_datetime()
  if (len(reqs)<3):
    #print("len(reqs)<3")
    reqs.append(dt)
    return dt
  if (len(reqs)==3 and (reqs[2] - reqs[0]).total_seconds()<=60):
    #print(f"(reqs[2] - reqs[0])<=60: {(reqs[2]-reqs[0]).total_seconds()}")
    time.sleep(22)
    dt=get_datetime()
    reqs.append(dt)
    reqs.pop(0)
    return dt
  if (len(reqs)>1 and (dt - reqs[2]).total_seconds()>60):
    #print("(dt - reqs[2])>60")
    reqs.clear()
    reqs.append(dt)
    return dt
  # print("unknown condition")
  # print (dt)
  # print (reqs)
  return dt

# rank offers
def rank_chunks(index, text)->str:
  print("ranking chunks: " + text)
  response = client.embeddings.create(
                input=json.dumps(text),
                model=embedding_model
              )
  embedding = response.data[0].embedding
  r = index.query(vector=embedding,
                  top_k=TOP_K,
                  include_values=False,
                  include_metadata=True)
  #print(r)
  context = ""
  i = int(r["matches"][0]["metadata"]["idx"])
  j = int(r["matches"][1]["metadata"]["idx"])
  # print(str(i) + " " + str(j))
  ti = r["matches"][0]["metadata"]["text"]
  tj = r["matches"][1]["metadata"]["text"]
  if (i<j):
    context = ti + tj
  else:
    context = tj + ti
  # print (context)
  return(context)

def qna(question, isCreative = True)->str:
    context = rank_chunks(index, question)
    
    if isCreative:
      verbose = "using as much details"
    else:
      verbose = ""
      
    utext = f"""
    you are a helpful assistant who specializes in helping AUB,
    American University of Beirut, students regarding university
    related questions.
    Answer the question {verbose} only based on the context below.
    If the question is not related to the context below,
    then answer 'This is not my specialty, please try again.'\n\n
    Context:
    {context} 
    
    
    Question: {question}
    Answer: 
    """
    #dt = slowit()
    
    completion = client.completions.create(
      model=llm_model,
      prompt = utext,
      temperature=1,
      max_tokens=MAX_TOKENS,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
    )
    answer = completion.choices[0].text.lstrip()
    #answer = answer.replace("\\n", "\n")
    print(answer)
    #answer = "hello world"
    return answer

iface = gr.Interface(
        fn=qna, 
        inputs=gr.components.Textbox(label='Question'),
        outputs=gr.components.Textbox(label='Answer'),
        allow_flagging='never')

app = FastAPI()

# to run locally: uvicorn rank:app --reload 
app = gr.mount_gradio_app(app, iface, path='/')