#https://python.langchain.com/docs/integrations/document_transformers/beautiful_soup/

from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"


import os
from os.path import join, dirname
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import PGVector
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

load_dotenv('.env')

OPENAI_API_KEY= os.environ.get('OPENAI_API_KEY')
PGVECTOR_CONNECTION_STRING = os.environ.get('PGVECTOR_CONNECTION_STRING')
USER_AGENT = os.environ.get('USER_AGENT')
print(f'PGVECTOR_CONNECTION_STRING -  {PGVECTOR_CONNECTION_STRING}')

import warnings
warnings.filterwarnings('ignore')

from IPython.display import display, Markdown

#load base page
hn_loader = AsyncChromiumLoader(["https://news.ycombinator.com/"])
html = hn_loader.load()
#extract links
bs = BeautifulSoup(html[0].page_content, 'html.parser')
urls = [link.get('href') for link in bs.select('td.title .titleline > a')]

# Load documents from the URLs
#TODO - use AsyncChromiumLoader
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs_list)
print(f"Text split into {len(chunks)} chunks")

#inti db
collection_name = "hn-rag"
pgvector_db = PGVector.from_documents(
    collection_name=f"{collection_name}",
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    use_jsonb=True
)
print("Database loaded")

local_model = "llama3.2:latest"
#local_model = "granite3-dense:8b"
llm = ChatOllama(model=local_model)

query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant.  Your task is to generate 2 
    different versions of the give user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on user question, your
    goal is to help users overcome some of the limitations of distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}"""
)
print("prompt created")

pg_retriever = MultiQueryRetriever.from_llm(
    pgvector_db.as_retriever(),
    llm,
    prompt=query_prompt
)

print("retriever created")

template = """Answer the question on on the following context: 
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
print("template created")

pg_chain = (
    {"context": pg_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("chain created")

def chat(question):
    """
    Chat with the page using the chaing, removed markdown for cli
    """
    return display(pg_chain.invoke(question))

chat('Can you summaraize the provided articles with the title of the document at the top?')