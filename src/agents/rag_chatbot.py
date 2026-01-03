
import chromadb
import getpass
import logging
import logging.config
import os
import yaml


from dotenv import load_dotenv

from langchain_ollama.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from typing_extensions import List, TypedDict

# set up environment and loggers
load_dotenv()
logging_config_path = os.environ["LOGGER_FILE_PATH"]
with open(logging_config_path) as fp:
    logger_conf = yaml.safe_load(fp)
logging.config.dictConfig(logger_conf)
logger = logging.getLogger(__name__)



# GENERATIVE LLM
llm = ChatOllama(model="qwen3:4b")

# EMBEDDING MODEL
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3",
                                   model_kwargs={'device': 'cpu'},
                                   encode_kwargs={'normalize_embeddings': True})



# VECTOR STORE

# Initialize the persistent client with the correct directory
persistent_client = chromadb.PersistentClient(path="./data/chroma_db")

# List collections to verify the existing ones
collections = persistent_client.list_collections()
print("Available collections:", collections)

# Ensure the collection name matches the existing one
vector_store = Chroma(
    client=persistent_client,
    collection_name='documents',  # Replace 'documents' with the actual collection name if different
    embedding_function=embeddings,
)

# Prompt
prompt = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""


@tool(response_format="content_and_artifact")
def retrieve(query):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k = 10)

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    
    
    
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you don't know."
        "For the answers you know provide the sources along with the url at the end"
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}


# state graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)




