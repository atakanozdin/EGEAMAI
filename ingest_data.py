import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFDirectoryLoader

os.environ["OPENAI_API_KEY"] = "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"

embedding_function = OpenAIEmbeddings()

# Register a new Chroma VectorDB
def ingest_db():
    DOCUMENTS_DIR = "D:/EgeAMAI/Documents"

    # Ask for collection name twice for confirmation
    while True:
        collection_name_1 = input("Enter the collection name: ")
        collection_name_2 = input("Confirm the collection name: ")

        if collection_name_1 == collection_name_2:
            break
        else:
            print("Collection names do not match. Please enter again.")

    print("Starting ingestion...")
    # Load raw documents from the directory
    loader = PyPDFDirectoryLoader(DOCUMENTS_DIR)
    docs = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    print('Split docs:', splits)
    
    # Create and store the embeddings in ChromaDB
    embedding_function = OpenAIEmbeddings()

    vectorstore = Chroma.from_documents(splits, embedding=embedding_function, 
                                        persist_directory=f"./chroma_db/{collection_name_1}")
    print("Ingestion complete...")

ingest_db()