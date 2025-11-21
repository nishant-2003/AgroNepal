import chromadb
from openai import OpenAI
import os
from chromadb.utils import embedding_functions
import numpy as np
from typing import List
import json
from dotenv import load_dotenv
from pathlib import Path

current_script_directory=Path(__file__).parent
PROJECT_ROOT=current_script_directory.parent.parent

DB_PATH=PROJECT_ROOT / "StoredEmbeddings"
DATASET_PATH=PROJECT_ROOT / "Datasets" / "question-answerpair.json"

load_dotenv()
if not os.getenv("OPEN_AI_API_KEY"):
    raise ValueError("OPEN_AI_API_KEY not found in the environemnt")

os.environ["OPEN_AI_API_KEY"]=os.getenv("OPEN_AI_API_KEY")
PERSISTENT_CLIENT = chromadb.PersistentClient(path=str(DB_PATH))
COLLECTION_NAME = "My_question_answer_pair_embeddings"
client = OpenAI(api_key=os.environ["OPEN_AI_API_KEY"])
QA_PAIRS = []
try:
    with open(DATASET_PATH, 'r', encoding="utf-8") as file:
        QA_PAIRS = json.load(file)
except:
    pass

def generate_embedding(text_to_embed: str) -> List[float]:
    response = client.embeddings.create(
        input=text_to_embed,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def return_cosine_similarity_and_results(query_text:str,threshold:int)->str:
    """It uses L2 distance"""
    openai_ef=embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPEN_AI_API_KEY"],
        model_name="text-embedding-3-small"
    )
    try:
        collection = PERSISTENT_CLIENT.get_collection(
            name=COLLECTION_NAME,
            embedding_function=openai_ef
        )
    except Exception as e:
        print(f"Error accessing collection: {e}. Ensure the collection exists.")
        return 

    query_embedding_list = generate_embedding(query_text)
    query_embedding = np.array(query_embedding_list)

    query_result = collection.query(
        query_texts=[query_text],
        n_results=2,
        include=['documents', 'embeddings', 'distances']
    )
    if query_result and query_result['documents'] and query_result['documents'][0]:
         for i in range(len(query_result['documents'][0])):
            retrieved_embedding=np.array(query_result['embeddings'][0][i])
            cosine_similarity=np.dot(query_embedding,retrieved_embedding)
            if cosine_similarity > threshold:
                retrieved_id = int(query_result['ids'][0][i])
                retrieved_answer = ""
                if retrieved_id < len(QA_PAIRS):
                    retrieved_answer = QA_PAIRS[retrieved_id]['answer']
                    return f"{retrieved_answer}"
    return f"None"
    
         
