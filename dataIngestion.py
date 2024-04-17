import os
from dotenv import load_dotenv
from langchain_community.vectorstores.pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from PyPDF2 import PdfReader

load_dotenv()

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

path=os.path.join(os.curdir,"./data/")
docs=os.listdir(path)

def get_text_chunks(docs):
    text=""
    for doc in docs:
        pdf=PdfReader(os.path.join(path,doc))
        for page in pdf.pages:
            text+=page.extract_text()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks=text_splitter.split_text(text)
    return chunks

chunks=get_text_chunks(docs)
index_name="basic"
# vectorstore=Pinecone.from_texts(chunks,embedding=embeddings,index_name=index_name)