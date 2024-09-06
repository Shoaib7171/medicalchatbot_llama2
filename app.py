from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

#Initializing pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone as PineconeClient

pc = PineconeClient(api_key='e4baa522-e803-4e7c-9636-6f45be6b8578')

prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

index_name = "medicalbot"
index = pc.Index(
    name=index_name,
    host='https://medicalbot-30o0q9d.svc.aped-4627-b74a.pinecone.io ' # This is the full host URL for your index
)



# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Pinecone vector store
index_name = "medicalbot"
index = pc.Index(index_name)
vectorstore = Pinecone(index, embeddings.embed_query, "text")

# Create the retriever
retriever = vectorstore.as_retriever(search_kwargs={'k': 2})


#promt and llm
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model='model\llama-2-7b-chat.ggmlv3.q4_0.bin',
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

chain_type_kwargs = {}  # Add any specific chain type arguments here
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs
)



@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)