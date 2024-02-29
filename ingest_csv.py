from langchain_community.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
import pandas as pd

os.environ["OPENAI_API_KEY"] = "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"

embedding_function = OpenAIEmbeddings()

# Register a new Chroma VectorDB
def ingest_csv_db():
    # Ask for collection name twice for confirmation
    while True:
        collection_name_1 = input("Enter the collection name: ")
        collection_name_2 = input("Confirm the collection name: ")

        if collection_name_1 == collection_name_2:
            break
        else:
            print("Collection names do not match. Please enter again.")

    print("Starting ingestion...")

    df = pd.read_excel('EgeAMAI1.xlsx')

    loader = DataFrameLoader(df, page_content_column="Content")
    documents = loader.load()

    vectorstore = Chroma.from_documents(documents, embedding=embedding_function, 
                                        persist_directory=f"./chroma_db/{collection_name_1}")
    print("Ingestion complete...")

ingest_csv_db()