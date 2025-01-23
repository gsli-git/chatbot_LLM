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

from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationalRetrievalChain

from .models import db, ChatMessage

# pc = Pinecone()

import os 
from dotenv import load_dotenv

load_dotenv()
open_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
print(open_api_key[:25],'\n',pinecone_api_key[:25])

# Create embeddings for each chunk
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

pc = Pinecone(api_key=pinecone_api_key) 
index_name = 'research-paper-on-vehicle-rag-index'
index = pc.Index(index_name)
# index.describe_index_stats()
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

# Define a retriever from the vector DB
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Get a prompt from LangChain hub
prompt = hub.pull("rlm/rag-prompt")
# prompt = ChatPromptTemplate.from_template(template)


# Define the LLM object using OpenAI
llm = ChatOpenAI(streaming=True, model_name="gpt-3.5-turbo", temperature=0)

# The function to combine multiple document into one
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# Define whole chain
retrieval_chain = (
    {
        # "context": retriever | format_docs, 
        "context": retriever.with_config(run_name="Docs"),
        "question": RunnablePassthrough()        
        }
    | prompt
    | llm
    | StrOutputParser()
)

# result = retrieval_chain.invoke("What is Crash Pulse of a Vehicle? How to obtain Crash Pulse?")
# print(result)

def call_chat(question):
    answer = ""
    for chunk in retrieval_chain.stream(question):
        answer += chunk
        yield {"token": chunk}

    chat_message = ChatMessage(user_id=1, question=question, answer=answer)
    db.session.add(chat_message)
    db.session.commit()