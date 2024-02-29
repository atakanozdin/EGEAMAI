from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import os
import pandas as pd
from langchain.document_loaders import CSVLoader 
from langchain.indexes import VectorstoreIndexCreator 
from langchain.chains import RetrievalQA 
from langchain.llms import OpenAI 

from langchain_community.document_loaders import DataFrameLoader

os.environ["OPENAI_API_KEY"] = "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"
embedding_function = OpenAIEmbeddings()

def call_csv_db(embedding_function):
    collection_name_1 = input("What is the collection name: ")
    csv_db = Chroma(persist_directory=f"./chroma_db/{collection_name_1}", embedding_function=embedding_function)
    
    retriever = csv_db.as_retriever()
    return csv_db, retriever

csv_db, retriever = call_csv_db(embedding_function)

def DataAnalystRun():
    template = """You are a virtual mechanical engineer and you have academic knowledge about additive manufacturing. 
    Answer the question based only on the following context:
    {context}

    Question: {question}

    You can refer to more than one academic essays.
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    # RESPONSE
    query = input("What is your question? ")
    print("RESPONSE: ")
    print(chain.invoke(query))

    # CONTEXT
    docs = csv_db.similarity_search(query)
    print("CONTEXT: ")
    print(docs[0].page_content)


DataAnalystRun()
# First Query: I'd like to make a machine learning model about porosity and lacks of surface. Which data can I use?
# Second Query: I'd like to learn melt pool temperature and melt pool geometry result of the my works. Which academical essay and its data should I use?
