import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = "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"
embedding_function = OpenAIEmbeddings()

def test_vectordb():
    collection_name_1 = input("What is the collection name: ")
    vector_db = Chroma(persist_directory=f"./chroma_db/{collection_name_1}", embedding_function=embedding_function)

    retriever = vector_db.as_retriever(lambda_val=0.025, k=7, filter=None)
    return retriever

retriever = test_vectordb()
print(retriever)