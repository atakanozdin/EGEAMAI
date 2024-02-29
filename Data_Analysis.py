from langchain.document_loaders import CSVLoader 
from langchain.indexes import VectorstoreIndexCreator 
from langchain.chains import RetrievalQA 
from langchain.llms import OpenAI 
import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader

os.environ["OPENAI_API_KEY"] = "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"

df = pd.read_excel('EgeAMAI1.xlsx')

# Load the documents 
loader = DataFrameLoader(df, page_content_column="Content")
# Create an index using the loaded documents 
index_creator = VectorstoreIndexCreator() 
docsearch = index_creator.from_loaders([loader])

# Create a question-answering chain using the index 
chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), 
                                    input_key="question")

# Pass a query to the chain 
query =input("What is your question? ")  #"Is there any work about porosity. Sort of the all works." 
response = chain({"question": query})

print(response['result'])