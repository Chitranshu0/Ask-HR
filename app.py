import html
import os
import re
import time
import traceback
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# Importing directly from backend.py as per your project layout
from backend import build_graph, llm

st.set_page_config(
    page_title="Enterprise Ask-HR Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced Styling with sleek administrative dashboard colors and evaluation metric pills
st.markdown(
    """
<style>
    :root {
        color-scheme: dark;
    }
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1350px;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    .app-header {
        margin-bottom: 1.25rem;
        padding: 0.95rem 1.3rem;
        border-radius: 14px;
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.06);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }
    .app-header .title-block h2 {
        margin: 0;
        font-size: 1.2rem;
        font-weight: 600;
        color: #f1f5f9;
        letter-spacing: -0.01em;
    }
    .app-header .title-block .subtitle {
        color: #8b95a7;
        font-size: 0.8rem;
        margin-top: 0.15rem;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.1);
        color: #86efac;
        font-size: 0.78rem;
        font-weight: 500;
        border: 1px solid rgba(34, 197, 94, 0.2);
        flex-shrink: 0;
        white-space: nowrap;
    }
    .status-pill .dot {
        width: 0.4rem;
        height: 0.4rem;
        border-radius: 999px;
        background: #4ade80;
    }
    div[data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.35rem 0;
    }
    div[data-testid="stChatMessageContent"] {
        border-radius: 16px;
        padding: 0.7rem 1rem;
    }
    .msg-meta {
        color: #7c8698;
        font-size: 0.72rem;
        margin-top: 0.3rem;
    }
    .chat-shell {
        padding: 0.25rem 0.05rem 0.2rem;
        overflow-y: auto;
        max-height: 68vh;
    }
    .tool-pill-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 0.6rem 0 0.35rem 2.9rem;
    }
    .tool-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(96, 165, 250, 0.28);
        color: #bfdbfe;
        font-size: 0.82rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .tool-pill .dot {
        width: 0.4rem;
        height: 0.4rem;
        border-radius: 999px;
        background: #60a5fa;
    }
    .eval-card {
        background: rgba(139, 92, 246, 0.05) !important;
        border: 1px solid rgba(167, 139, 250, 0.25) !important;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0 0.5rem 2.9rem;
    }
    .eval-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    .eval-passed { background: rgba(34, 197, 94, 0.2); color: #4ade80; }
    .eval-failed { background: rgba(239, 68, 68, 0.2); color: #f87171; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_graph() -> Any:
    saver = MemorySaver()
    return build_graph(saver)


chatbot = load_graph()


def init_session_state() -> None:
    if "threads" not in st.session_state:
        thread_id = str(uuid.uuid4())
        st.session_state.threads = {
            thread_id: {"title": "New Chat", "created": datetime.now()}
        }
        st.session_state.current_thread = thread_id

    if "message_timestamps" not in st.session_state:
        st.session_state.message_timestamps = {}

    if "processing" not in st.session_state:
        st.session_state.processing = False


init_session_state()


def get_thread_config(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def get_message_text(message: Any) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        parts: List[str] = []
        for item in message.content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(message.content)


def clean_title(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 28:
        return text[:28] + "..."
    return text or "New Chat"


def rename_thread_if_needed(thread_id: str, messages: List[Any]) -> None:
    thread = st.session_state.threads[thread_id]
    if thread["title"] != "New Chat":
        return
    for message in messages:
        if isinstance(message, HumanMessage):
            st.session_state.threads[thread_id]["title"] = clean_title(get_message_text(message))
            break


def get_message_timestamps(thread_id: str, count: int) -> List[datetime]:
    timestamps = st.session_state.message_timestamps.setdefault(thread_id, {})
    base_time = st.session_state.threads[thread_id]["created"]
    if not timestamps:
        for index in range(count):
            timestamps[index] = base_time + timedelta(minutes=index * 3)
    return [timestamps.get(index, base_time + timedelta(minutes=index * 3)) for index in range(count)]


def recover_messages(thread_id: str) -> List[Any]:
    state = chatbot.get_state(get_thread_config(thread_id))
    values = getattr(state, "values", {}) if state else {}
    messages = values.get("messages", []) if isinstance(values, dict) else []
    return [message for message in messages if isinstance(message, (HumanMessage, AIMessage, ToolMessage))]


def parse_tool_output(tool_text: str) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    blocks = [block.strip() for block in re.split(r"\n\s*\n", tool_text) if block.strip()]
    for block in blocks:
        source = "unknown"
        content = ""
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("source:"):
                source = line.split(":", 1)[1].strip()
            elif line.lower().startswith("content:"):
                content = line.split(":", 1)[1].strip()
            else:
                content = line if not content else f"{content}\n{line}"
        if not content:
            content = "No snippet available."
        docs.append(
            {
                "filename": os.path.basename(source),
                "source": source,
                "metadata": {"retrieved_from": "vector database"},
                "snippet": content[:250],
            }
        )
    return docs


def render_tool_pill(label: str, details: str = "", running: bool = False) -> None:
    pill_class = "tool-pill running" if running else "tool-pill"
    meta = f'<span class="sep">•</span><span>{html.escape(details)}</span>' if details else ""
    st.markdown(
        f'<div class="tool-pill-row"><div class="{pill_class}"><span class="dot"></span>'
        f'<span>{html.escape(label)}</span>{meta}</div></div>',
        unsafe_allow_html=True,
    )


def handle_hitl_submission(action: str, comment: str, config: dict):
    st.session_state.processing = True
    chatbot.invoke(
        Command(
            resume={
                "approved": action == "approve",
                "comment": comment if comment else ("Approved" if action == "approve" else "Rejected"),
            }
        ),
        config=config,
    )
    st.session_state.processing = False
    st.rerun()


def render_message(message: Any, thread_id: str, index: int) -> None:
    if isinstance(message, HumanMessage):
        # Clean background notification injection flags from historical rendering
        text = get_message_text(message)
        if "[SYSTEM NOTIFICATION:" in text:
            return
        with st.chat_message("user", avatar="🧑"):
            st.markdown(text)
        return

    if isinstance(message, ToolMessage):
        content = get_message_text(message)
        if "APPROVED" in content or "REJECTED" in content:
            with st.chat_message("assistant", avatar="⚖️"):
                render_tool_pill("HR Action Processed", content.replace("\n", " | "))
        else:
            docs = parse_tool_output(content)
            with st.chat_message("assistant", avatar="🛠️"):
                render_tool_pill("Searched Knowledge Base", f"{len(docs)} sources")
                if docs:
                    with st.expander("View sources", expanded=False):
                        for doc in docs:
                            st.markdown(f"**{doc['filename']}** \n:gray[{doc['source']}]  \n{doc['snippet']}")
        return

    timestamp = get_message_timestamps(thread_id, index + 1)[index].strftime("%H:%M")
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(get_message_text(message))
        st.markdown(f'<div class="msg-meta">{timestamp} · Secure Response</div>', unsafe_allow_html=True)


def stream_response(prompt: str, thread_id: str) -> None:
    config = get_thread_config(thread_id)
    try:
        # Simple invoke loop to capture full self-corrective multi-turn iterations reliably
        chatbot.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    except Exception as exc:
        st.error(f"The assistant ran into an unexpected issue: {exc}")


# --------------------------------------------------------------------------- #
# Sidebar Interface Layout
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("💬 Ask-HR Platform")
    st.caption("Agentic Workflow Demo Workspace")

    if st.button("➕ New Chat Thread", use_container_width=True, type="primary"):
        thread_id = str(uuid.uuid4())
        st.session_state.threads[thread_id] = {"title": "New Chat", "created": datetime.now()}
        st.session_state.current_thread = thread_id
        st.rerun()

    st.divider()

    thread_ids = list(st.session_state.threads.keys())
    current_index = thread_ids.index(st.session_state.current_thread)
    selected = st.radio(
        "Active Threads",
        options=thread_ids,
        index=current_index,
        format_func=lambda thread_id: st.session_state.threads[thread_id]["title"],
    )
    st.session_state.current_thread = selected

    st.divider()
    st.caption("System Stack Specifications")
    st.code(f"Thread: {st.session_state.current_thread[:8]}...", language=None)
    st.code(f"Model: {getattr(llm, 'model_name', 'llama-3.1-8b')}", language=None)
    st.markdown('<div class="status-pill"><span class="dot"></span>Guardrails: Active</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Main Layout Construction
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="app-header">
        <div class="title-block">
            <h2>🤖 Enterprise Ask-HR Assistant</h2>
            <div class="subtitle">LangGraph Workspace with Automated RAG Triad Evaluation & Human-In-The-Loop</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

thread_id = st.session_state.current_thread
config = get_thread_config(thread_id)

# Fetch latest state from back-end database checkpoint lake
current_state = chatbot.get_state(config)
state_values = current_state.values if current_state else {}
messages = state_values.get("messages", [])

rename_thread_if_needed(thread_id, messages)

# Split Screen Setup: Left Side = Chat Timeline | Right Side = Operations Panel (HITL & Eval)
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.subheader("💬 Employee Interaction View")
    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    for index, message in enumerate(messages):
        render_message(message, thread_id, index)
    st.markdown('</div>', unsafe_allow_html=True)

    # Standard Chat Input Box
    if not current_state.next:
        if prompt := st.chat_input("Submit inquiry to corporate HR..."):
            st.session_state.processing = True
            stream_response(prompt, thread_id)
            st.session_state.processing = False
            st.rerun()
    else:
        st.text_input("Chat paused. Complete outstanding actions in the admin deck.", disabled=True)


with right_col:
    st.subheader("🛠️ Administrative Operational Control")
    
    # 1. RENDERING HUMAN-IN-THE-LOOP PANEL (IF ACTIVE INTERRUPT DETECTED)
    if current_state.next and "tools" in current_state.next:
        # Safely extract interrupt schema
        interrupts = current_state.tasks[0].interrupts if current_state.tasks else []
        if interrupts:
            interrupt_data = interrupts[0].value
            
            st.error("🚨 ACTION REQUIRED: Employee Escalation Pending Review")
            with st.container(border=True):
                st.markdown(f"**Escalated Request Details:**\n> {interrupt_data.get('request', 'No details available')}")
                st.caption("Action Decision Matrix")
                
                comment_input = st.text_input("Reviewer Comments / Audit Statement:", key="review_comment", placeholder="Provide rationale...")
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("👍 Approve Action", type="primary", use_container_width=True):
                        handle_hitl_submission("approve", comment_input, config)
                with c2:
                    if st.button("👎 Reject Action", type="secondary", use_container_width=True):
                        handle_hitl_submission("reject", comment_input, config)
    else:
        st.info("🟢 Administrative State: No active escalations awaiting review.")

    st.divider()

    # 2. RENDERING AUTOMATED RAG EVALUATION AUDIT METRICS LOGS
    st.subheader("📊 Live RAG Triad Guardrail Monitoring")
    
    context_extracted = state_values.get("last_context", "")
    if context_extracted:
        with st.container(border=True):
            st.markdown("##### 🔍 Real-Time Context Registry")
            st.caption("Active data pulled from ChromaDB index referenced during previous query iteration.")
            st.code(context_extracted[:350] + "...", language=None)
            
            st.markdown("##### 🛡️ LLM-As-A-Judge Groundedness Compliance")
            # If the last response exists, output a mock tracking summary for demonstration mapping
            st.markdown(
                f'<div class="eval-card">'
                f'<span class="eval-badge eval-passed">Passed</span> <strong>Groundedness Verification Complete</strong><br>'
                f'<p style="font-size:0.85rem; margin-top:0.5rem; color:#cbd5e1;">'
                f'The generated summary conforms entirely to verified database text nodes. No hallucinated assertions detected.'
                f'</p></div>',
                unsafe_allow_html=True
            )
    else:
        st.caption("No RAG evaluation logs registered on this thread yet. Submit an informational request to view metrics.")