from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
import chainlit as cl
from dotenv import load_dotenv

load_dotenv()

# Vectorstore
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

vectorstore=Pinecone.from_existing_index('basic',embedding=embeddings)
retriever=vectorstore.as_retriever()

# Query Transform Prompt and Chat Prompt
query_transform_prompt = ChatPromptTemplate.from_template("""chat-history: {chat_history}\n Given the above conversation, generate a search query to search for relevant information based on the chat history and the last question from a vectorstore. Just return the search query and nothing else""")

SYSTEM_TEMPLATE = """
Elaborate Answer the question based only on given context
<context>
{context}
</context>
question:{question}
Elaborate the answer related to the question from the context in english, unless stated otherwise.
Dont state about the context in the answer.
Answer:
"""
question_answering_prompt = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)


# Memory Chain using LCEL
def create_memory_chain():
    llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3, convert_system_message_to_human=True, verbose=True)
    query_transforming_retriever_chain = (query_transform_prompt | llm | StrOutputParser() | retriever).with_config(run_name="chainlit")
    document_chain =create_stuff_documents_chain(llm, question_answering_prompt)
    return RunnablePassthrough.assign(context=query_transforming_retriever_chain,).assign(answer=document_chain,)

# Run the chain
@cl.step
def run_chain(question):
    
    chat_history=cl.user_session.get("chat_history")
    print(chat_history)
    chain=cl.user_session.get("chain")
    chat_history+=f"""\nHuman: {question}"""
    res=chain.invoke({"chat_history":chat_history, 'question':question})['answer']
    chat_history+=f"""\nAssistant: {res}"""
    cl.user_session.set("chat_history", chat_history)
    return res

@cl.on_chat_start
def initialize_chat():
    chat_history=""
    chain=create_memory_chain()
    cl.user_session.set("chat_history", chat_history)
    cl.user_session.set("chain", chain)
    print("A new chat session has started!")

@cl.on_message
async def on_message(msg: cl.Message):
    await cl.Message(content=run_chain(msg.content,)).send()