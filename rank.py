import os, json
from openai import OpenAI
from fastapi import FastAPI
import gradio as gr
from bson import ObjectId 
from datetime import datetime
from pymongo import MongoClient
import cohere

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="cohere")

llm_model       = "gpt-4o-mini-2024-07-18"
embedding_model = "text-embedding-3-large"
llm_temp        = 0.8
vec_dim         = 1024
q_vec_dim       = 256
max_tokens      = 10000
top_k           = 10
reqs            = []
IDK             = "idk"
IDK             = "Please refer to the OSA webpage: https://www.aub.edu.lb/SAO/Pages/default.aspx"
NINO            = "nothing in, nothing out."
IDK_LIMIT       = 0.1
RL              = 0.5
Q_LIMIT         = 0.9
Q_ONE           = 0.999

win_size = 400      # sliding window size
win_step = 200      # sliding window step
context_mult = 4    # context: 2*context_mult*win_size

# system="You are AUB student assistant with access to some AUB documents and information. You use the information you are given to provide advice."

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

###############################################################################
def embed_query(query):
###############################################################################
  res = oai.embeddings.create(
                input=query,
                model=embedding_model,
                dimensions=q_vec_dim
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

  if (rr_docs == []):
    print(">>>>>>> black hole\n")
    return []
    
  res = co.rerank(model="rerank-english-v3.0", query=query, documents=rr_docs, top_n=top_n, return_documents=True)

  # print the results
  # for doc in res.results:
  #     print(f"Doc {doc.index:03d} -> score: {doc.relevance_score:.5f} {similar_vectors[doc.index]['idx']:03d} {similar_vectors[doc.index]['type']}\t{similar_vectors[doc.index]['doc_name']}")

  return res.results

###############################################################################
def get_llm_context(mongo, vec_id):
###############################################################################
  # print("\n------- get_llm_context\n")
  
  vec = vectors.find_one({'_id': vec_id})
  # print(f"get_llm_context: {vec['doc_name']}\t{vec['idx']}\t{vec['doc_id']}")
  idx = vec['idx']
  
  # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  # print(f"CHUNK: {vec['text']}")
  # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

  type = vec['type']
  # if type == 'chunk': # many win_size
       # win_size = 500
       # win_step = 200

  pvec = docs.find_one({'_id': vec['doc_id']})
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

###############################################################################
def ask_llm(query, f_res):
###############################################################################
  # print(f"\nask_llm: {query}\n")

  if query == "":
    return NINO

  scon = f'''
  you are an AI assistant for students of AUB (American University of Beirut).
  '''

  ucon = ""
  # unpack the tuple
  for i, (index, score, document) in enumerate(f_res): 
    ucon += f'''
    "\n{document}\n"
    '''

  ucon += f'''
  \n{query}\n
  '''
  # print("********************")
  # print(ucon)
  # print("********************")

  completion = oai.chat.completions.create(
  model=llm_model,
  temperature=llm_temp,
  max_tokens=max_tokens,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  response_format={
    "type": "text"
  },
  messages=[
      {"role": "system", "content": scon},
      {"role": "user", "content": ucon}
    ]
  )
  r = completion.choices[0].message.content
  print(f"total tokens: {completion.usage.total_tokens}\n")
  return(r)


###############################################################################
def llm_best_match(query, sq)->str:
###############################################################################
  print(f">>>>>>>> llm_best_match: {query}\n")

  if query == "":
    return NINO

  scon = f'''
  you are an AI assistant for students at AUB (American University of Beirut).
  '''

  ucon = ""

  q_str = ""
  for i, o in enumerate(sq):
    q_str += f"""
    {i+1} - {o['query']}\n
    """


  ucon += f"""
  given the following labeled queries:

  {q_str}

  which of them is a semantic match of the query "{query}"?

  if there is a match, provide only the label of the matched query, otherwise respond with 0.

  answer is:
  """

  # print("********************")
  # print(ucon)
  # print("********************")

  while True:
    completion = oai.chat.completions.create(
      model=llm_model,
      temperature=llm_temp,
      max_tokens=max_tokens,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      response_format={
        "type": "text"
      },
      messages=[
          {"role": "system", "content": scon},
          {"role": "user", "content": ucon}
        ]
    )
    r = completion.choices[0].message.content
    print(f">>>>>>>>>{r}<<<<<<<<")
    if len(r) <= 1:
      break

  # print(f"total tokens: {completion.usage.total_tokens}\n")

  r_i = int(r[0])
  if r_i == 0:
    x = ""
  else:
    x = sq[r_i-1]['query']

  print(f"llm_best_match: {r}\tans: {x}\n")

  return(r[0])


###############################################################################
def search_similar_queries(query_vec, top_k=5):
###############################################################################
    vectors = qnas

    # query_vec = embed_text(query)

    # Define the Atlas Search pipeline.
    pipeline = [
        {
            '$vectorSearch': {
                'index': 'query_index',
                'path': 'emb',
                'queryVector': query_vec,
                'numCandidates': 100,
                'limit': top_k
            }
        }, {
            '$project': {
                '_id': 1,
                'query': 1,
                'answer': 1,
                'score': {
                    '$meta': 'vectorSearchScore'
                }
            }
        }
    ]

    # Execute the search pipeline.
    results = list(vectors.aggregate(pipeline))

    return results


###############################################################################
def check_query(query, query_vec)->(str, str): # type: ignore
###############################################################################

  sq = search_similar_queries(query_vec)

  # filter most similar queries
  f_sq = [item for item in sq if item['score'] > Q_LIMIT]

  if not f_sq:
    return "", ""

  # print("Results from search_similar_queries:")
  # print(sq)

  # check for an exact match
  top_vec = sq[0]['score']
  if top_vec > Q_ONE:
    return sq[0]['answer'], sq[0]['_id']

  # print results
  # for doc in f_sq:
  #     print(f"{doc['_id']} score: {doc['score']:.5f} {doc['query']}")

  # find best semantic match via LLM
  idx_s = llm_best_match(query, f_sq)

  idx = int(idx_s)
  if idx == 0:
    return "", ""

  answer = f_sq[idx-1]['answer']
  id = f_sq[idx-1]['_id']

  return answer, id


###############################################################################
def qna(query)->str:
###############################################################################
  print(f"QnA: {query}\n")

  # Define the log document to be inserted
  log = {
      "query": f"{query}",
      "answer": "",
      "dts": datetime.now().strftime('%y%m%d%H%M%S'),
	    "qna_id": "",
      "rrv": []
  }
  rrv = {
          "vr": 1,
          "vid": "",
          "vds": 1.0,
          "vcs": 1.0,
          "vidx": 0,
          "type": "",
          "d_name": ""
        }

  # define the qna document
  qna_doc = {
      "query": query,
      "answer": "",
      "dts": datetime.now().strftime('%y%m%d%H%M%S'),
      "state": ""
  }
  
  if query == "":
    return NINO

  query_vec = embed_query(query)

  # check if query exists
  answer, id = check_query(query, query_vec)

  # if query exists, log & return answer
  if answer != "":
    log['answer'] = answer
    log['qna_id'] = id
    res = logs.insert_one(log)
    print(f"logging : {res.inserted_id}")
    return answer

  ############ main pipeline ############
  
  # define the qna document
  qna_doc = {
      "query": query,
      "answer": "",
      "dts": datetime.now().strftime('%y%m%d%H%M%S'),
      "emb": query_vec,
      "state": ""
  }
  
  query_vec = embed_text(query)

  sv = search_similar_vectors(vectors, query_vec)
  # print results
  # for doc in sv:
  #     print(f"{doc['_id']} score: {doc['score']:.5f} {doc['idx']:03d}\t{doc['doc_name']}")

  res = rerank(co, query, sv)
  # print results
  for doc in res:
    i = doc.index
    print(f"Doc {i:03d} {sv[i]['_id']} {sv[i]['score']:.5f} -> {doc.relevance_score:.5f} {sv[i]['idx']:03d} {sv[i]['type']}\t{sv[i]['doc_name']}")
    obj = {                             # re-ranked-vector
            'vr': i,                    # vector ranking (not rr)
            'vid': sv[i]['_id'],        # vector object_id
            'vds': sv[i]['score'],      # vector dense similarity
            'vcs': doc.relevance_score, # vector cross similarity
            'vidx': sv[i]['idx'],       # vector chunk idx w/r parent doc
            'type': sv[i]['type'],      # contingency 4 multi chunking 
            'd_name': sv[i]['doc_name'] # parent document name
        }
    log['rrv'].append(obj)

  # check if there is no answer
  if res[0].relevance_score < IDK_LIMIT:
    ans = IDK
    qna_doc['answer'] = ans
    res = qnas.insert_one(qna_doc)
    log['qna_id'] = res.inserted_id
    log['answer'] = ans
    res = logs.insert_one(log)
    # Print the ID of the inserted document
    print(f"logging : {res.inserted_id}")
    return IDK

  # f_res = [res[0]] + [item for item in res[1:] if item.relevance_score > RL]
  f_res = [(res[0].index, res[0].relevance_score, get_llm_context(mongo, sv[res[0].index]['_id']))] + \
  [(item.index, item.relevance_score, get_llm_context(mongo, sv[item.index]['_id'])) for item in res[1:] if item.relevance_score > RL]

    
  ans = ask_llm(query, f_res)
  # print(f"\nllm ans:\n {ans}\n")
  
  qna_doc['answer'] = ans
  res = qnas.insert_one(qna_doc)
  log['qna_id'] = res.inserted_id
  log['answer'] = ans
  
  # add the answer to the log
  log['answer'] = ans
  res = logs.insert_one(log)

  # Print the ID of the inserted document
  print(f"logging : {res.inserted_id}")

  return ans

#------------------------------------------------------------------------------

# def qna(question="hello")->str:
#        return ("hello qna")

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