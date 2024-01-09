from openai import OpenAI
import os

client = OpenAI(
   api_key=os.environ["OPENAI"],
 )

response = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt="what is the capital of Turkey?",
  temperature=1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(response.choices[0].text)