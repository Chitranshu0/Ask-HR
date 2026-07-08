"""
Ask-HR backend — LangGraph + SQLite.

Graph design
------------
    START -> chat_node -> tools_condition -> tools -> chat_node -> END
                        -> (no tool call)  -> END

There is no separate router_node. chat_node is the only decision point: the
LLM itself picks a tool based on the system prompt below.

  - retriever_tool          -> policy / information questions (RAG)
  - request_hr_approval     -> anything needing a human decision (leave,
                                resignation, salary, transfer, etc.)

request_hr_approval calls interrupt() internally. When it fires, the graph
pauses mid-tool-call; the whole thread (state["messages"]) is already visible
to whoever resumes it, so the human reviewing the request sees full context
for free. Once resumed with Command(resume={...}), the tool returns the
HR decision as its ToolMessage output, execution falls through the same
tools -> chat_node -> END edge as the RAG path, and chat_node writes the
final approved/rejected reply to the employee.

Memory
------
Short-term : LangGraph SqliteSaver checkpointer (per-thread message state,
             file-based so it survives restarts).
Long-term  : Separate SQLite DB (askhr_memory.db) holding the thread
             registry, the HR decision audit log, and cached summaries.
"""

import os
import time
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langgraph.checkpoint.memory import MemorySaver 
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

load_dotenv(override=True)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
LLM_MODEL = os.getenv("ASKHR_LLM_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE = float(os.getenv("ASKHR_LLM_TEMPERATURE", "0.0")) # Lowered to 0.0 for stable tool calling
LLM_MAX_TOKENS = int(os.getenv("ASKHR_LLM_MAX_TOKENS", "1200"))

EMBEDDING_MODEL = os.getenv("ASKHR_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DB_DIR = os.getenv("ASKHR_VECTOR_DB_DIR", "AgenticHR/RAG_pipeline/PolicyVB")


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #
llm = ChatGroq(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
)


# --------------------------------------------------------------------------- #
# Vector store + retriever tool (RAG)
# --------------------------------------------------------------------------- #
_embeddings = None
_vector_db = None


def get_vector_db():
    global _embeddings, _vector_db
    if _vector_db is not None:
        return _vector_db
    _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    _vector_db = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=_embeddings,
    )
    return _vector_db


@tool
def retriever_tool(query: str, k: int = 5, search_type: str = "mmr") -> str:
    """
    Retriever the content from the vector db
    """
    vector_db = get_vector_db()
    if vector_db is None:
        return "The knowledge base is temporarily unavailable. Please try again shortly."

    retriever = vector_db.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, "fetch_k": max(50, k * 5)},
    )
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found."

    return "\n\n".join(
        f"Source: {list(doc.metadata.get('source', 'Unknown').split('/'))[-1].split('.')[0]}\nContent: {doc.page_content}"
        for doc in docs
    )


# --------------------------------------------------------------------------- #
# HITL tool
# --------------------------------------------------------------------------- #
@tool
def request_hr_approval(request_summary: str) -> str:
    """
    Escalate an employee request to a human HR reviewer for approval or
    rejection. Use this for anything requiring a human decision: leave
    requests, resignation, salary issues, promotions, transfers, etc.
    """
    decision = interrupt(
        {
            "type": "hr_approval",
            "request": request_summary,
            "instruction": "Approve or reject this employee request.",
        }
    )

    approved = bool(decision.get("approved"))
    comment = decision.get("comment") or ("Approved" if approved else "No reason provided")

    status = "APPROVED" if approved else "REJECTED"
    return f"{status} by HR.\nComment: {comment}"


tools = [retriever_tool, request_hr_approval]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

# Cleaned up prompt to prevent Llama formatting leakage
CHAT_SYSTEM_PROMPT = SystemMessage(
    content="""You are Ask-HR, an internal corporate HR assistant. 

Your job is to answer questions by calling tools or replying directly based on tool outputs.

1. For informational questions (e.g., company identity, company name, policy details, leave balance rules, notice period, holidays, benefits, timings), you MUST execute 'retriever_tool'.
2. For action requests needing human decision-making (e.g., leave requests, resignation, salary changes), call 'request_hr_approval'.

Never fabricate company guidelines. When tools return information, summarize it naturally for the employee."""
)


# --------------------------------------------------------------------------- #
# Core node
# --------------------------------------------------------------------------- #
def chat_node(state: ChatState) -> Dict[str, list]:
    messages = state["messages"]
    
    # Ensure the System Prompt remains at the head of the message stack
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [CHAT_SYSTEM_PROMPT] + [m for m in messages if not isinstance(m, SystemMessage)]
    else:
        messages = [CHAT_SYSTEM_PROMPT] + [m for m in messages[1:] if not isinstance(m, SystemMessage)]

    try:
        response = llm_with_tools.invoke(messages)
    except Exception as exc:
        response = AIMessage(
            content="I'm having trouble reaching the assistant right now. Please try again in a moment."
        )
    return {"messages": [response]}


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def build_graph(checkpointer):
    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    return graph.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------- #
# CLI test runner
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    thread_id = "cli_test_001"
    config = {"configurable": {"thread_id": thread_id}}

    checkpointer = MemorySaver()
    chatbot = build_graph(checkpointer)
    print("\n🤖 Ask-HR (Stateless): How can I help you today? (type 'exit' to quit)\n")

    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in ("quit", "exit", "break"):
            break

        result = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

        if result.get("__interrupt__"):
            request = result["__interrupt__"][0].value
            print("\n==============================")
            print("🚨 HR Approval Required")
            print("==============================")
            print("Employee Request:", request["request"])

            decision_input = input("\nApprove? (yes/no): ")
            comment = input("Comment: ")

            result = chatbot.invoke(
                Command(
                    resume={
                        "approved": decision_input.lower() == "yes",
                        "comment": comment,
                    }
                ),
                config=config,
            )

        print("\n🤖 AI:", result["messages"][-1].content)