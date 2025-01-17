from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore


import os 
from dotenv import load_dotenv

load_dotenv()
open_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
# print(open_api_key,'\n',pinecone_api_key)

# Create embeddings for each chunk
embeddings = OpenAIEmbeddings()

pc = Pinecone(api_key=pinecone_api_key) 
index_name = 'research-paper-on-vehicle-rag-index'
index = pc.Index(index_name)
# index.describe_index_stats()
# vector_store = PineconeVectorStore(index=index, embedding=embeddings)

loader = DirectoryLoader('', glob="./paper/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Create embeddings for each chunk
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())

# Define a retriever from the vector DB
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Get a prompt from LangChain hub
prompt = hub.pull("rlm/rag-prompt")

# Define the LLM object using OpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# The function to combine multiple document into one
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define whole chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("What is Crash Pulse of a Vehicle? How to obtain Crash Pulse?")
print(result)

