from pinecone import Pinecone
import os, json, datetime, time
from openai import OpenAI
from fastapi import FastAPI
import gradio as gr
## github

from datetime import datetime
from pymongo import MongoClient
import cohere

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold



llm_model       = "gpt-4o-mini"
embedding_model = "text-embedding-3-large"
vec_dim         = 1024
max_tokens      = 500
top_k           = 10
reqs            = []

# connect to openAI
oai = OpenAI(
    api_key = os.environ["OPENAI"],
 )

# answer = "Please refer to the OSA webpage: https://www.aub.edu.lb/SAO/Pages/default.aspx"

def qna():
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