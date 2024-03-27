from pinecone import Pinecone
import os, json, datetime, time
from openai import OpenAI
from fastapi import FastAPI
import gradio as gr
## github
vec_dim = 1024
MAX_TOKENS = 500
TOP_K = 3
EPS = 4
reqs = []

no_answer = "no answer"

# pinecone api key
pc_api_key = os.environ["PCNOS"]
#env = "gcp-starter"
#index_name = "aub-ada-1536"
#index_name = "a3-ada-1536"
index_name = "a3-v3l-1024"

# connect to pinecone database
pc = Pinecone(api_key=pc_api_key)

# connect to pinecone index
index = pc.Index(index_name)

#print(pinecone.describe_index(index_name))
index.describe_index_stats()

# embedding model
# embedding_model = "text-embedding-ada-002"
embedding_model = "text-embedding-3-large"

# LLM model
# llm_model = "gpt-3.5-turbo-instruct"
llm_model = "gpt-3.5-turbo-1106"

# connect to openAI using api_key
client = OpenAI(
   api_key=os.environ["OPENAI"],
 )

#---------------------------------------------

def get_datetime():
  dt = datetime.datetime.now()
  #dt_str = dt.strftime ("%d%H%M%S")
  return dt

def slowit():
  print ("anti spam")

# calculate the % difference between 2 scores
def score_diff(s1, s2):
  return round(100*(s1-s2)/s1)

# rank offers
def rank_vectors(text)->str:
  print(f"ranking vectors: {text}")
  response = client.embeddings.create(
                input=text,
                model=embedding_model,
                dimensions=vec_dim
              )
  ebs = response.data[0].embedding
  r = index.query(vector=ebs,
                  top_k=TOP_K,
                  include_values=False,
                  include_metadata=True)
  #print(r)
  context = r["matches"][0]["metadata"]["text"]
  
  s0 = r["matches"][0]["score"]
  s1 = r["matches"][1]["score"]
  s2 = r["matches"][2]["score"]
  
  # if score_diff(s0, s1) < EPS:
  #   context = context + "\n" + r["matches"][1]["metadata"]["text"]
    
  # if score_diff(s0, s2) < EPS:
  #   context = context + "\n" + r["matches"][2]["metadata"]["text"]
    
  print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  x0 = r["matches"][0]["metadata"]["name"]
  x1 = r["matches"][1]["metadata"]["name"]
  x2 = r["matches"][2]["metadata"]["name"]
  print(f"chunk: {x0} {x1} {x2}")
  print(f"scores: {s0} {s1} {s2}")
  print (context)
  print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
  return(context)

# mapping LLM ignorance
def check_words(words, sentence):
  # Convert the sentence to lowercase
  sentence = sentence.lower()
  # Loop through each word in the list
  for word in words:
    # Convert the word to lowercase
    word = word.lower()
    # Check if the word is in the sentence
    if word not in sentence:
      # Return False if any word is missing
      return False
  # Return True if all words are present
  return True

# AI answering user's question
def qna(question)->str:
  context = rank_vectors(question)

  # sp = "You are a help assistant."
  sp = ""
  context = f"""
  \"\"\"
  {context}
  \"\"\"
  """

  up = f"""
  {context}

  assistant will carefully analyze the attached information.
  assistant will avoid unnecessary justifications in his answer.
  assistant answers the following question ONLY based on 
  the attached info without using any external sources: 
  \"{question}\".

  if the attached info is NOT enough to answer user's question,
  then assistant will be brief and answer only with \"no answer\".
  """
  # print(up)
  res = client.chat.completions.create(
    model=llm_model,
    messages=[
      {
        "role": "system",
        "content": sp
      },
      {
        "role": "user",
        "content": up
      }
    ],
    temperature=0.5,
    max_tokens=MAX_TOKENS,
    top_p=0.1,
    frequency_penalty=0,
    presence_penalty=0
  )
  # not provided in the attached information
  #res
  answer = res.choices[0].message.content.lstrip()
  fa = (answer.lower() == no_answer)
  lb = ['not', 'attached', 'information']
  fb = check_words(lb, answer)
  if (fa or fb):
    answer = "Please refer to the OSA webpage: https://www.aub.edu.lb/SAO/Pages/default.aspx"
  
  return answer
  
  
iface = gr.Interface(
       fn=qna, 
       inputs=gr.components.Textbox(label='Question'),
       outputs=gr.components.Textbox(label='Answer'),
       allow_flagging='never')

app = FastAPI()

# to run locally: uvicorn rank:app --reload 
app = gr.mount_gradio_app(app, iface, path='/')

#-------------------------------------------------------

# question = "who is the dean of student affairs ?"
# question = "how much is 1+2?"

# ans = qna(question)
# print(ans)