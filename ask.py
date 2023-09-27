from fastapi import FastAPI
import gradio as gr
import openai, os

# LLM model
llm_model = "gpt-3.5-turbo-instruct"

# Set your API key
openai.api_key = os.environ["OPENAI_API_KEY"]

def ask(input: str) -> str:
    completion = openai.Completion.create(
      model=llm_model,
      temperature=1,
      max_tokens=1000,
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

app = gr.mount_gradio_app(app, iface, path='/')