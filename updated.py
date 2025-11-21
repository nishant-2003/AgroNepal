### The code is written in 11-october-2025
import os
from dotenv import load_dotenv
from typing import List, TypedDict

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain import hub # For pulling the ReAct prompt

from utils.QAembeddings import a_embeddings
from utils.datasetloader.loader import Datasetloader
#from utils.splitprocessing.splitprocessor import SplittingData
import chromadb
from langchain_chroma import Chroma
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)
import logging
from typing import Type

import asyncio
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

from tavily import TavilyClient

# --- Global Clients ---
openai_async_client = AsyncOpenAI()
tavily_client = TavilyClient()
load_dotenv()
assemblyai_api = os.environ.get("ASSEMBLYAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

full_transcript = []

# --- 1. AssemblyAI Streaming Transcriber Functions ---

def on_begin(self: Type[StreamingClient], event: BeginEvent):
    print(f"Session started: {event.id}")


def on_turn(self: Type[StreamingClient], event: TurnEvent):
    print(f"\r{event.transcript}", end="", flush=True)

    if event.end_of_turn:
        print()
        
        if event.transcript:
            full_transcript.append(event.transcript)

        cleaned_transcript = event.transcript.strip().lower().rstrip(".,!?")
        
        if cleaned_transcript == "okay that's it":
            print("\n--- 'Okay that's it' detected. Stopping session. ---")
            self.disconnect(terminate=True)

        if not event.turn_is_formatted:
            params = StreamingSessionParameters(
                format_turns=True,
            )
            self.set_params(params)


def on_terminated(self: Type[StreamingClient], event: TerminationEvent):
    print("\n" + "="*50)
    print("--- FULL QUERY ---")
    
    final_text = " ".join(full_transcript)
    print(final_text)
    
    print("="*50)
    
    print(
        f"Session terminated: {event.audio_duration_seconds} seconds of audio processed"
    )


def on_error(self: Type[StreamingClient], error: StreamingError):
    print(f"Error occurred: {error}")
    
    if full_transcript:
        print("\n--- Partial query (before error) ---")
        final_text = " ".join(full_transcript)
        print(final_text)
        print("="*50)

# --- 2. ReAct Agent Tools ---
# We define the functions your agent can use as tools.

@tool
def high_confidence_qa_tool(query: str) -> str:
    """
    Checks a high-confidence, pre-computed Q&A database for a direct answer.
    Use this first for any query. It returns a string answer if a high-confidence
    match (cosine similarity > 0.8) is found, otherwise it returns 'None'.
    """
    print(f"--- Calling Tool: high_confidence_qa_tool ---")
    generated_answer = a_embeddings.return_cosine_similarity_and_results(query, 0.8)
    if generated_answer != 'None':
        print(f"--- Tool Result: Found high-confidence answer. ---")
        return generated_answer
    else:
        print(f"--- Tool Result: No high-confidence answer found. ---")
        return "None"

@tool
def local_vector_store_tool(query: str) -> str:
    """
    Searches the local Chroma vector database ('Agricultural_dataset_vectors')
    for relevant documents. Use this tool when the high_confidence_qa_tool fails.
    It returns document snippets if found, or 'sorry' if no relevant docs are found.
    """
    print(f"--- Calling Tool: local_vector_store_tool ---")
    CHROMA_PERSIST_DIRECTORY = "./StoredEmbeddings"
    MY_COLLECTION_NAME = "Agricultural_dataset_vectors"

    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=MY_COLLECTION_NAME
    )
    documents = vector_store.similarity_search(query, k=3)
    doc_contents = [doc.page_content for doc in documents]
    
    if not doc_contents:
        print(f"--- Tool Result: No documents found in RAG. ---")
        return "sorry"

    # Use the 'sorry' prompt logic from your original code to validate docs
    prompt = PromptTemplate(
        template="""
You are an AI assistant. Your task is to answer the user's question based strictly
and exclusively on the information within the 'CONTEXT'.
If the CONTEXT does not contain a clear answer, you MUST respond with the exact phrase: 'sorry'.
Otherwise, synthesize a concise answer from the context.

--- CONTEXT FROM DOCUMENT(S) ---
{context}
--- END OF CONTEXT ---

--- USER'S QUESTION ---
{query}
--- END OF QUESTION ---
""",
    input_variables=["context", "query"]
    )
    chain = prompt | llm | StrOutputParser()
    response_answer = chain.invoke({"context": "\n---\n".join(doc_contents), "query": query})
    
    print(f"--- Tool Result (RAG): {response_answer[:50]}... ---")
    return response_answer

@tool
def web_search_tool(query: str) -> str:
    """
    Use this tool as a last resort if both the high_confidence_qa_tool and
    local_vector_store_tool fail to provide an answer.
    It searches the web using Tavily for a relevant answer and validates it.
    """
    print(f"--- Calling Tool: web_search_tool ---")
    EXCLUDE_DOMAINS = [
        "reddit.com", "facebook.com", "twitter.com", "pintrest.com", "amazon.com"
    ]
    
    try:
        expert_response = tavily_client.search(
            query=query,
            search_depth="advanced",
            exclude_domains=EXCLUDE_DOMAINS,
            auto_parameters=True,
            max_results=3
        )
        
        # Get the content of the first good result
        ans = ""
        results = expert_response.get('results', [])
        if results:
            ans = results[0].get('content', '')
        
        if not ans:
            print(f"--- Tool Result (Web): No content found. ---")
            return "No information found on the web."

        # Validate the web result
        validation_prompt = PromptTemplate(
            template="""You are an expert fact-checker. Your job is to validate a retrieved piece of information (CONTEXT) against a user's (QUESTION).
The context may be from a web search and could be low-quality.

QUESTION:
{query}

CONTEXT:
{answer}

Read the CONTEXT and the QUESTION, then provide your verdict:
Verdict: [CORRECT] if the context directly and accurately answers the question.
Verdict: [INCORRECT] if the context is irrelevant, low-quality, or does not answer the question.
""",
            input_variables=["query", "answer"]
        )
        validation_chain = validation_prompt | llm | StrOutputParser()
        verdict = validation_chain.invoke({"query": query, "answer": ans})

        if "CORRECT" in verdict:
            print(f"--- Tool Result (Web): Found and validated answer. ---")
            return ans
        else:
            print(f"--- Tool Result (Web): Found info, but failed validation. ---")
            return "I found some information on the web, but I could not verify its quality or relevance. Please try rephrasing your question."
            
    except Exception as e:
        print(f"--- Tool Error (Web): {e} ---")
        return "There was an error during the web search."

# --- 3. Agent & Graph Setup ---

# Define the tools the agent can use
tools = [high_confidence_qa_tool, local_vector_store_tool, web_search_tool]

# Define the system prompt for the ReAct agent
# This prompt explains the agent's persona and how it should prioritize tools.
HUB_PROMPT = """
You are 'AgroNepal', an expert AI agricultural assistant. You are helpful, respectful, and strictly factual.
You must answer the user's query by following a precise set of rules.

Your Goal: Find the best possible answer to the user's question.

Your Tools: You have three tools available to you, listed in order of preference.
1.  `high_confidence_qa_tool`: A private database of known questions.
2.  `local_vector_store_tool`: A private library of agricultural documents.
3.  `web_search_tool`: A general web search.

Your Action Plan (Strict):
1.  First, ALWAYS use the `high_confidence_qa_tool` first.
2.  If and ONLY IF `high_confidence_qa_tool` returns 'None', you must then reflect and try the `local_vector_store_tool`.
3.  If and ONLY IF `local_vector_store_tool` returns 'sorry' (meaning it found no documents or the documents were irrelevant), you must then reflect and use the `web_search_tool`.
4.  If the `web_search_tool` also fails, apologize and state that you cannot find the answer.
5.  Once you have a valid answer from any tool, provide it as your Final Answer.

Your Thought Process:
You must show your work at each step. Use the "Thought" section to explain your plan, which tool you are choosing, and why.
When a tool fails, your Thought must explain that it failed and how you are correcting your plan by moving to the next tool.

Begin.
"""

# Pull the base ReAct prompt and insert our custom instructions
prompt = hub.pull("hwchase17/react")
prompt = prompt.partial(
    instructions=HUB_PROMPT,
    tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
    tool_names=", ".join([tool.name for tool in tools]),
)

# Create the agent
agent = create_react_agent(llm, tools, prompt)

# Create the agent executor (the "orchestrator")
app = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 4. Main Execution Block ---

if __name__ == "__main__":
    if assemblyai_api == "YOUR_ASSEMBLYAI_API_KEY_HERE":
        print("Error: Please replace the default API key with your actual AssemblyAI API key.")
        quit()
        
    client = StreamingClient(
        StreamingClientOptions(
            api_key=assemblyai_api,
            api_host="streaming.assemblyai.com",
        )
    )

    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    client.connect(
        StreamingParameters(
            sample_rate = 16000,
            format_turns = True
        )
    )

    DEVICE_ID = 2  # <-- IMPORTANT: Change this to your microphone's device index

    try:
        try:
            print(f"\n--- Listening with device ID {DEVICE_ID} ---")
            print("Speak into your microphone. Say 'okay that's it' or press Ctrl+C to stop.\n")

            client.stream(
                aai.extras.MicrophoneStream(
                    sample_rate=16000,
                    device_index=DEVICE_ID
                )
            )

        finally:
            client.disconnect(terminate=True)

    except KeyboardInterrupt:
        print("\n--- Stopping transcriber (Ctrl+C pressed) ---")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Make sure you have set the correct DEVICE_ID number.")
        client.disconnect(terminate=True)
    
    # Process the final transcript
    final_query = " ".join(full_transcript)
    
    if not final_query:
        print("No transcript captured. Exiting.")
        quit()

    print(f"\n--- Sending final query to agent: '{final_query}' ---")
    
    # Run the agent orchestrator
    user_query=input("enter query")
    #input_local = {"input": final_query}
    input_local={"input":user_query}
    generation = ""
    
    try:
        # We use invoke (not stream) because we only care about the final answer
        # The agent's internal streaming is handled by `verbose=True`
        response = app.invoke(input_local, {"recursion_limit": 6})
        generation = response.get("output", "I'm sorry, I was unable to process your request.")
        
    except Exception as e:
        print(f"\n--- Agent execution error: {e} ---")
        generation = "I'm sorry, an error occurred while I was thinking."

    print("\n" + "="*50)
    print(f"--- FINAL ANSWER ---")
    print(generation)
    print("="*50)

    # --- 5. Text-to-Speech Output ---
    
    async def audio_output() -> None:
        print("\n--- Starting Text-to-Speech ---")
        try:
            async with openai_async_client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="coral",
                input=generation,
                # instructions="Speak in a cheerful and positive tone of female", # Instructions not supported by gpt-4o-mini-tts
                response_format="pcm",
            ) as response:
                await LocalAudioPlayer().play(response)
            print("--- Audio playback finished ---")
        except Exception as e:
            print(f"--- Audio playback error: {e} ---")

    if generation:
        asyncio.run(audio_output())
    else:
        print("No generation to speak.")

    # --- 6. Graph Visualization (Optional) ---
    # Note: ReAct agent graphs are complex loops. This will show the agent's structure.
    try:
        png_data = app.get_graph().draw_mermaid_png()
        output_filename = "agent_graph_visualization.png"
        with open(output_filename, "wb") as f:
            f.write(png_data)
        print(f"\n--- Agent graph saved to {output_filename} ---")
    except Exception as e:
        print(f"--- Could not save graph visualization: {e} ---")

