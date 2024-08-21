import os, json, datetime, time
from openai import OpenAI
from fastapi import FastAPI
import gradio as gr

from datetime import datetime
from pymongo import MongoClient
import cohere

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="cohere")

llm_model       = "gpt-4o-mini-2024-07-18"
embedding_model = "text-embedding-3-large"
vec_dim         = 1024
max_tokens      = 2000
top_k           = 10
reqs            = []
IDK             = "Please refer to the OSA webpage: https://www.aub.edu.lb/SAO/Pages/default.aspx"
IDK_LIMIT       = 0.1
RL              = 0.5

win_size = 400      # sliding window size
win_step = 200      # sliding window step
context_mult = 4    # context: 2*context_mult*win_size

# connect to openAI
oai = OpenAI(
    api_key = os.environ["OPENAI"],
 )

###############################################################################
def embed_text(text):
###############################################################################
  res = oai.embeddings.create(
                input=text,
                model=embedding_model,
                dimensions=vec_dim
              )
  return res.data[0].embedding
#------------------------------------------------------------------------------

res = embed_text("apple is juicy and delicious")

print(f"embeddings = {len(res)}\n")

completion = oai.chat.completions.create(
  model=llm_model,
  temperature=0.8,
  max_tokens=max_tokens,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  response_format={
    "type": "text"
  },
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "what is the capital of Turkey?"}
  ]
)
print(f"openai: {completion.choices[0].message.content}\n")

#------------------------------------------------------------------------------

mongo = MongoClient(os.environ["MONGODB"])

try:
    mongo.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!\n")
except Exception as e:
    print(e)

db = mongo.a3
docs = db.docs
vectors = db.vectors
qnas = db.qnas
logs = db.logs

#------------------------------------------------------------------------------

co = cohere.Client(os.environ["COHERE"])

###############################################################################
def search_similar_vectors(vectors, query_vec, top_k=top_k):
###############################################################################
    # Define the Atlas Search pipeline.
    pipeline = [
        {
            '$vectorSearch': {
                'index': 'vectors_index',
                'path': 'emb',
                'queryVector': query_vec,
                'numCandidates': 150,
                'limit': top_k
            }
        }, {
            '$project': {
                '_id': 1,
                'doc_name': 1,
                'score': {
                    '$meta': 'vectorSearchScore'
                },
                'idx': 1,
                'type': 1,
                'text': 1
            }
        }
    ]
    # Execute the search pipeline.
    results = list(vectors.aggregate(pipeline))

    return results

###############################################################################
def rerank(co, query, similar_vectors, top_n=top_k):
###############################################################################
  # print(f"\ncohere rerank: {query}\n")

  rr_docs = []
  for doc in similar_vectors:
    rr_docs.append(doc['text'])

  res = co.rerank(model="rerank-english-v3.0", query=query, documents=rr_docs, top_n=top_n, return_documents=True)

  # print the results
  # for doc in res.results:
  #     print(f"Doc {doc.index:03d} -> score: {doc.relevance_score:.5f} {similar_vectors[doc.index]['idx']:03d} {similar_vectors[doc.index]['type']}\t{similar_vectors[doc.index]['doc_name']}")

  return res.results

###############################################################################
def get_llm_context(mongo, vec_id):
###############################################################################
  db = mongo.a3
  docs = db.docs
  vectors = db.vectors

  vec = vectors.find_one({'_id': vec_id})
  # print(f"get_llm_context: {vec['doc_name']}\t{vec['idx']}\t{vec['doc_id']}")
  idx = vec['idx']
  
  # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  # print(f"CHUNK: {vec['text']}")
  # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

  type = vec['type']
  # if type == 'chunk':
       # win_size = 500
       # win_step = 200

  pvec = db.docs.find_one({'_id': vec['doc_id']})
  #print(pvec['name'])
  text = pvec['text']

  start_idx = idx * win_step
  start_context = max(0, start_idx - context_mult*win_size)
  if start_context == 0:
    delta = context_mult*win_size - start_idx
  else:
    delta = 0
  end_context = min(len(text), start_idx + delta + context_mult*win_size)

  return text[start_context:end_context]


#------------------------------------------------------------------------------

def qna(question="hello")->str:
       return ("hello qna")

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