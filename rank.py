import openai, pinecone, os, json, numpy as np
from openai.embeddings_utils import get_embedding

# embedding model
embedding_model = "text-embedding-ada-002"

# rank offers
def rank_chunks(index, text):
  print("ranking offers")
  print(text)
  embedding = get_embedding(
                json.dumps(text),
                engine=embedding_model
              )
  r = index.query(vector=embedding,
                  top_k=10,
                  include_values=False,
                  include_metadata=True)
  return(r)

q = "what is HIP?"