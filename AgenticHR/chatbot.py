"""
This page is set-up for the agentic chatbot feature.
"""

import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool

from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma   # pip install -U langchain-chroma

from langgraph.checkpoint.postgres import PostgresSaver

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

##################################################
# State
##################################################

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


##################################################
# LLM
##################################################

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=300,
)


##################################################
# Vector DB
##################################################

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="RAG_pipeline/PolicyVB",
    embedding_function=embeddings
)


##################################################
# Tools
##################################################

@tool
def retriever_tool(
    query: str,
    k: int = 5,
    search_type: str = "mmr"
) -> str:
    """
    Retrieve relevant HR policy documents from the vector database.
    """

    retriever = vector_db.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": k,
            "fetch_k": 50,
        },
    )

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found."

    return "\n\n".join(
        f"Source: {doc.metadata.get('source','Unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in docs
    )


##################################################
# Tool Binding
##################################################

tools = [retriever_tool]

llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)


##################################################
# Chat Node
##################################################

def chat_node(state: ChatState):

    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }


##################################################
# Database
##################################################

DB_URI = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5442/langgraph"
)


##################################################
# Graph
##################################################

def build_graph(checkpointer):

    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")

    graph.add_conditional_edges(
        "chat_node",
        tools_condition,
    )

    graph.add_edge(
        "tools",
        "chat_node",
    )

    return graph.compile(
        checkpointer=checkpointer
    )


##################################################
# Main
##################################################

if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "user_001",
            "checkpoint_ns": "askhr"
        }
    }

    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:

        print("Connected to PostgreSQL...")

        # Safe to call every startup
        checkpointer.setup()

        print("Checkpoint tables are ready.")

        chatbot = build_graph(checkpointer)

        print("\nAI: How can I help you today?\n")

        while True:

            user_input = input("You: ")

            if user_input.lower() in ("quit", "exit", "break"):
                break

            result = chatbot.invoke(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=config
            )

            print("\nAI:", result["messages"][-1].content)