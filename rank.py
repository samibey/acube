import openai, pinecone, os, json
from openai.embeddings_utils import get_embedding
from fastapi import FastAPI
import gradio as gr

vec_dim = 1536
MAX_TOKENS = 600
TOP_K = 2

# pinecone api key
pc_api_key = os.environ["PINECONE"]
env = "us-west1-gcp-free"
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
openai.api_key = os.environ["OPENAI"]

# rank offers
def rank_chunks(index, text)->str:
  print("ranking chunks: " + text)
  embedding = get_embedding(
                json.dumps(text),
                engine=embedding_model
              )
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
    completion = openai.Completion.create(
      model=llm_model,
      temperature=1,
      max_tokens=MAX_TOKENS,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      prompt = utext,
    )
    answer = completion.choices[0].text.lstrip()
    #answer = answer.replace("\\n", "\n")
    return answer

iface = gr.Interface(
        fn=qna, 
        inputs=gr.components.Textbox(label='Question'),
        outputs=gr.components.Textbox(label='Answer'),
        allow_flagging='never')

app = FastAPI()

app = gr.mount_gradio_app(app, iface, path='/')