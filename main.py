### The code is written in 11-october-2025
## First processing datasets and pdfs
#from utils.datasetloader.loader import Datasetloader
#from utils.splitprocessing.splitprocessor import SplittingData
#test=Datasetloader("C:\\Users\\KIIT\\Downloads\\dataset")
#var1=test.smart_loader()
#print(var1[63].metadata["source"])

#from utils.QAembeddings import a_embeddings
#print(a_embeddings.return_cosine_similarity_and_results("Why is crop rotation important in farming?",0.8))
import os
from dotenv import load_dotenv
from typing import List, TypedDict

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph,END
from utils.QAembeddings import a_embeddings
from utils.datasetloader.loader import Datasetloader
#from utils.splitprocessing.splitprocessor import SplittingData
import chromadb
from langchain_chroma import Chroma
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

load_dotenv()
'''
try:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OpenAI API key not found please set in .env file")
    if "TAVILY_API_KEY" not in os.environ:
        raise ValueError("Tavily API key not found please set in .env file")
except:
    print("Fine")
    '''
llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)
class GraphState(TypedDict):
    query:str
    documents:List[str]
    generation:str
    decision:str

embeddings=OpenAIEmbeddings(model="text-embedding-3-small")

def write_query(state:GraphState):
    query:str
    documents:List[str]
    generation:str

def check_query(state:GraphState):
    query=state["query"]
    print(type(query))
    generated_answer=a_embeddings.return_cosine_similarity_and_results(query,0.8)
    if (generated_answer!='None'):
        return {"decision":"q_a_result_generation",
                "generation":generated_answer}
    else:
        return {"decision":"retrieve_documents"}
    
def route_decision(state:GraphState):
    decision=state["decision"]
    if decision=="generate_from_q/a_dataset":
        return f"q_a_result_generation"
    else:
        return f"retrieve_documents"


#cautious expert prompt
def q_a_result_generation(state:GraphState):
    query=state["query"]
    initial_answer=state["generation"]
    prompt=PromptTemplate(
        template="""You are 'NepAgri' agricultural scientist and educator. Your primary goal is to provide accurate, well-grounded explanations without making unverified claims.
            Given the following:
    Question: "{question}"
    Answer: "{initial_answer}"

    Generate the following sections based on the Q&A pair:

1.  **Detailed Scientific Explanation:**
    * Explain the core principles (e.g., biological, chemical, soil science) that make the answer correct.
    * Clearly describe the mechanism or process involved.

2.  **Related Agricultural Concepts:**
    * List and briefly define key terms, techniques, or scientific principles that are directly related to the question and answer.

3.  **Practical Applications & Broader Context:**
    * Describe how this knowledge is applied in practical farming, crop management, or livestock operations.
    * Mention broader fields of study or related challenges in agriculture (e.g., sustainability, soil health, pest management) where this concept is important.

Strictly adhere to these critical rules:
* Your explanation must be based on widely accepted agricultural science and verifiable best practices.
* When connecting to other topics, use cautious phrasing like "This principle is also relevant in..." or "A related area for farmers to consider is...".
* DO NOT invent data, specific crop yields, chemical application rates, or unverified pest/disease names.
* **Crucially:** If the topic involves advice that could be dependent on local conditions (like soil type, climate, or regulations), you MUST add a disclaimer: "This is a general explanation. Farmers should always consult with a local agricultural extension service for advice tailored to their specific region and conditions."

""",
    input_variables=["question","answer"]
    )
    
    chain=prompt | llm |StrOutputParser()
    response=chain.invoke({"question":query,"answer":initial_answer})
    return {"generation":response}

def retrieve_documents(state:GraphState):
    query=state["query"]
    CHROMA_PERSIST_DIRECTORY="./StoredEmbeddings"
    MY_COLLECTION_NAME="Agricultural_dataset_vectors"

    vector_store=Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=MY_COLLECTION_NAME
    )
    documents=vector_store.similarity_search(query,k=3)
    doc_contents=[doc.page_content for doc in documents]

    
    return {"documents":documents}


def prompt_documents(state:GraphState):
    query=state["query"]
    documents=state["documents"]
    prompt=PromptTemplate(
        template="""
You are 'NepAgri', an expert AI agricultural assistant. Your sole purpose is to provide accurate, safe, and evidence-based answers by analyzing documents.

Your task is to answer the user's question based strictly and exclusively on the information within the 'CONTEXT' section provided below.

**Core Directives:**
1.  **Answer ONLY from the provided CONTEXT.** Do not use any external knowledge, make assumptions, or guess.
2.  **If the CONTEXT does not contain a clear answer to the question, you MUST respond with the exact phrase:** 'I cannot find a definitive answer for your query in the provided documents. For this issue, it is best to consult a local agricultural extension officer.' Do not try to rephrase or offer partial advice.
3.  **Cite your source.** If the context contains a 'Document ID' or a source name, mention it in your answer.
4.  **Keep the answer concise** and directly address the user's question.

--- CONTEXT FROM DOCUMENT(S) ---
{context}
--- END OF CONTEXT ---

--- USER'S QUESTION ---
{query}
--- END OF QUESTION ---

--- YOUR FACTUAL RESPONSE ---
""",
    input_variables=["documents","query"]
    )


    chain=prompt | llm | StrOutputParser()
    response=chain.invoke({"context":documents,"query":query})
    return {"generation":response}




workflow=StateGraph(GraphState)
workflow.add_node("check_query",check_query)
workflow.add_node("q_a_result_generation",q_a_result_generation)
workflow.add_node("route_decision",route_decision)
workflow.add_node("retrieve_documents",retrieve_documents)
workflow.add_node("prompt_documents",prompt_documents)

workflow.set_entry_point("check_query")
workflow.add_conditional_edges(
    "check_query",
    lambda state:state["decision"],
    {
        "q_a_result_generation":"q_a_result_generation",
        "retrieve_documents":"retrieve_documents"
    }
)

#workflow.add_edge("check_query","q_a_result_generation")
workflow.add_edge("q_a_result_generation",END)
#workflow.add_edge("check_query","retrieve_documents")
workflow.add_edge("retrieve_documents","prompt_documents")
workflow.add_edge("prompt_documents",END)
app=workflow.compile()
        
query=input("Enter your question related to agriculture:")
input_local={"query":query}

for output in app.stream(input_local):
    for key, value in output.items():
       pass
print(f"{value["generation"]}") 


png_data=app.get_graph().draw_mermaid_png()
output_filename="graph_visualization.png"

with open(output_filename,"wb") as f:
    f.write(png_data)


