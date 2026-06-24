"""
This page is set-up for the agentic chatbot feature.
"""

from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool

from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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
    persist_directory=r"RAG_pipeline/PolicyVB",
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
            "fetch_k": 50
        }
    )

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found."

    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Content: {doc.page_content}"
            for doc in docs
        ]
    )

    return context


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
    response = llm_with_tools.invoke(state["messages"])

    return {
        "messages": [response]
    }


##################################################
# Graph
##################################################

def build_graph():

    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")

    graph.add_conditional_edges(
        "chat_node",
        tools_condition
    )

    graph.add_edge("tools", "chat_node")

    return graph.compile()


##################################################
# Testing
##################################################

# if __name__ == "__main__":

#     chatbot = build_graph()

#     result = chatbot.invoke(
#         {
#             "messages": [
#                 HumanMessage(
#                     content="What is the leave policy?"
#                 )
#             ]
#         }
#     )

#     for msg in result["messages"]:
#         print("\n")
#         print(msg)