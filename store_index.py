from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
text_chunk_01= [t.page_content for t in text_chunks]
embeddings = download_hugging_face_embeddings()






from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=PINECONE_API_KEY)

#connecting to host
index_name = "medicalbot"
index = pc.Index(
    name=index_name,
    host='https://medicalbot-30o0q9d.svc.aped-4627-b74a.pinecone.io ' # This is the full host URL for your index
)





#initializing pinecone.
import os
from pinecone import Pinecone
import numpy as np
from tqdm import tqdm

# Initialize Pinecone
api_key = PINECONE_API_KEY
pc = Pinecone(api_key=api_key)

# Connect to your index
index_name = "medicalbot"
try:
    index = pc.Index(index_name)
except Exception as e:
    print(f"Error connecting to index '{index_name}': {e}")
    exit(1)

# Prepare your data
# Assuming you have your text chunks in a list called 'text_chunks'
# and your embeddings in a numpy array called 'embeddings'
# data = []
# for i, (text, embedding) in enumerate(zip(text_chunk_01, embeddingss)):
#     data.append((str(i), embedding.tolist(), {"text": text}))

# # # Upsert data in batches
# batch_size = 100
# for i in tqdm(range(0, len(data), batch_size)):
#     batch = data[i:i+batch_size]
#     try:
#         index.upsert(vectors=batch)
#     except Exception as e:
#         print(f"Error upserting batch starting at index {i}: {e}")
#         continue

print("Upsert complete!")