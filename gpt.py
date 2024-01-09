from fastapi import FastAPI
from openai import OpenAI
import gradio as gr
import os

# LLM model
llm_model = "gpt-3.5-turbo-instruct"

# Set your API key
client = OpenAI(
  api_key=os.environ["OPENAI"],
)
## github
def ask(input: str) -> str:
    completion = client.completions.create(
      model=llm_model,
      temperature=1,
      max_tokens=250,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      prompt = input,
    )
    return completion.choices[0].text.lstrip()

iface = gr.Interface(
        fn=ask, 
        inputs=gr.components.Textbox(label='Question'),
        outputs=gr.components.Textbox(label='Answer'),
        allow_flagging='never')

app = FastAPI()

# @app.get('/')
# async def root():
#     return 'acube app is running', 200

# to run locally: uvicorn gpt:app --reload 
app = gr.mount_gradio_app(app, iface, path='/')