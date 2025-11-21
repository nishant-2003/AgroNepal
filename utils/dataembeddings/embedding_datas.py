
import chromadb
from openai import OpenAI
import os
from chromadb.utils import embedding_functions
import numpy as np
from typing import List
from dotenv import load_dotenv
from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from utils.splitprocessing.splitprocessor import SplittingData
from utils.datasetloader.loader import Datasetloader

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")



def create_and_persist(document_chunks:list,persist_directory:str,collection_name:str):
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store=Chroma.from_documents(
        documents=document_chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    return vector_store

# main execution

if __name__=="__main__":
    current_script_directory=Path(__file__).parent
    PROJECT_ROOT=current_script_directory.parent.parent 
    dotenv_path=PROJECT_ROOT /".env"
    print(f"The paths is {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in the environemnt")
    
    dataset_path=PROJECT_ROOT / "Datasets" / "files"
    file_directory=Datasetloader(dataset_path)
    loaded_files=file_directory.smart_loader()
    loader=SplittingData(loaded_files)
    document_chunks=loader.splitter()
    CHROMA_PERSISTENT_DIRECTORY=PROJECT_ROOT /"StoredEmbeddings"
    COLLECTION_NAME="Agricultural_dataset_vectors"
    vector_store=create_and_persist(
        document_chunks,
        CHROMA_PERSISTENT_DIRECTORY,
        COLLECTION_NAME
    )
   # retrieved_docs=vector_store.similarity_search("what is the minimum coffee support price of Fresh cherry",k=2)
    #for doc in retrieved_docs:
     #print(f"contents: {doc.page_content}\n metadata: {doc.metadata}")
# Define the same paths and names used during creation
'''
# Define the same paths and names used during creation
CHROMA_PERSIST_DIRECTORY = "./StoredEmbeddings"
MY_COLLECTION_NAME = "Agricultural_dataset_vectors"

# Initialize the embedding model (must be the same one used for storage)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store from the persistent directory and specify the collection
print(f"Loading vector store from '{CHROMA_PERSIST_DIRECTORY}' with collection '{MY_COLLECTION_NAME}'...")
vector_store = Chroma(
    persist_directory=CHROMA_PERSIST_DIRECTORY,
    embedding_function=embedding_model,
    collection_name=MY_COLLECTION_NAME
)

# Now your vector_store is ready to be used for queries
print("Vector store loaded successfully!")
results = vector_store.similarity_search("what is the minimum coffee support price of Fresh cherry")
print(results)

'''
 
    


