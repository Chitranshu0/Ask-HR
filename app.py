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
    page_title="Enterprise HR Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    :root {
        color-scheme: dark;
    }
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1250px;
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
        max-height: 72vh;
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
    .tool-pill.running .dot {
        background: #fbbf24;
        animation: pulse 1s infinite ease-in-out;
    }
    .tool-pill .sep {
        color: rgba(191, 219, 254, 0.4);
    }
    .typing-indicator {
        display: inline-flex;
        gap: 0.25rem;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.05);
        color: #cbd5e1;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .typing-dot {
        width: 0.45rem;
        height: 0.45rem;
        border-radius: 999px;
        background: #60a5fa;
        animation: pulse 1s infinite ease-in-out;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes pulse {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.55; }
        40% { transform: scale(1); opacity: 1; }
    }
    div[data-testid="stExpander"] {
        margin: 0.15rem 0 0.75rem 2.9rem;
        border: 1px solid rgba(96, 165, 250, 0.18);
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.35);
    }
    div[data-testid="stExpander"] summary {
        font-size: 0.82rem;
        color: #93c5fd;
    }
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

    if "last_response_meta" not in st.session_state:
        st.session_state.last_response_meta = {}


init_session_state()


def get_thread_config(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def get_message_text(message: Any) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        parts: List[str] = []
        for item in message.content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
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
    """Resumes the graph with the human HR choice."""
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
        with st.chat_message("user", avatar="🧑"):
            st.markdown(get_message_text(message))
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
        st.markdown(f'<div class="msg-meta">{timestamp} · Response ready</div>', unsafe_allow_html=True)
        if st.button("📋 Copy", key=f"copy-{thread_id}-{index}"):
            st.session_state.last_copied = get_message_text(message)
            st.toast("Response copied")


def render_tool_cards(tool_cards: List[Dict[str, Any]]) -> None:
    for tool_card in tool_cards:
        is_running = tool_card["status"] == "Running..."
        render_tool_pill(
            label=tool_card["name"],
            details=f"Elapsed: {tool_card.get('elapsed', 0)}s" if not is_running else "",
            running=is_running,
        )
        if not is_running and tool_card.get("docs"):
            with st.expander("View sources", expanded=False):
                for doc in tool_card["docs"]:
                    st.markdown(f"**{doc['filename']}** \n:gray[{doc['source']}]  \n{doc['snippet']}")


def stream_response(prompt: str, thread_id: str) -> None:
    config = get_thread_config(thread_id)
    tool_cards: List[Dict[str, Any]] = []
    assistant_response = ""
    status_label = "🧠 Thinking..."

    status_placeholder = st.empty()
    chat_container = st.chat_message("assistant", avatar="🤖")
    response_placeholder = chat_container.empty()

    try:
        for event in chatbot.stream(
            {"messages": [HumanMessage(content=prompt)]},
            config=config,
            stream_mode="updates",
        ):
            for _, update in event.items():
                if not isinstance(update, dict):
                    continue
                for message in update.get("messages", []):
                    if isinstance(message, ToolMessage):
                        existing = next((c for c in tool_cards if c["status"] == "Running..."), None)
                        if existing:
                            docs = parse_tool_output(get_message_text(message))
                            existing["status"] = "✅ Completed"
                            existing["docs"] = docs
                        status_label = "✅ Context Retrieved"
                    elif isinstance(message, AIMessage):
                        if getattr(message, "tool_calls", None):
                            t_call = message.tool_calls[0]
                            t_name = "Retriever Tool" if t_call["name"] == "retriever_tool" else "HR Approval Requested"
                            status_label = f"🔎 Executing {t_name}..."
                            tool_cards.append({"name": t_name, "status": "Running...", "docs": []})
                        elif isinstance(message.content, str) and message.content.strip():
                            assistant_response += message.content

            if tool_cards:
                with chat_container:
                    render_tool_cards(tool_cards)
                status_placeholder.empty()
            else:
                status_placeholder.markdown(
                    f'<div class="typing-indicator"><span class="typing-dot"></span>'
                    f'<span class="typing-dot"></span><span class="typing-dot"></span>'
                    f'<span>{html.escape(status_label)}</span></div>',
                    unsafe_allow_html=True,
                )

            if assistant_response:
                response_placeholder.markdown(assistant_response)

    except Exception as exc:
        st.error(f"The assistant ran into an unexpected issue: {exc}")


with st.sidebar:
    st.title("💬 Enterprise HR Assistant")
    st.caption("In-Memory LangGraph Threads (Stateless Sessions)")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        thread_id = str(uuid.uuid4())
        st.session_state.threads[thread_id] = {"title": "New Chat", "created": datetime.now()}
        st.session_state.current_thread = thread_id
        st.rerun()

    st.divider()

    thread_ids = list(st.session_state.threads.keys())
    current_index = thread_ids.index(st.session_state.current_thread)
    selected = st.radio(
        "Conversation history",
        options=thread_ids,
        index=current_index,
        format_func=lambda thread_id: st.session_state.threads[thread_id]["title"],
    )
    st.session_state.current_thread = selected

    st.divider()
    st.caption("Current Thread ID")
    st.code(st.session_state.current_thread, language=None)
    st.caption("Status")
    st.markdown('<div class="status-pill"><span class="dot"></span>Online</div>', unsafe_allow_html=True)
    st.caption("Checkpointing Runtime")
    st.code("In-Memory (MemorySaver)", language=None)
    st.caption("Model")
    st.code(getattr(llm, "model_name", getattr(llm, "model", "llama-3.1-8b-instant")), language=None)


st.markdown(
    """
    <div class="app-header">
        <div class="title-block">
            <h2>🤖 Enterprise HR Assistant</h2>
            <div class="subtitle">LangGraph • Groq • ChromaDB • In-Memory Store</div>
        </div>
        <div class="status-pill"><span class="dot"></span>Connected</div>
    </div>
    """,
    unsafe_allow_html=True,
)

thread_id = st.session_state.current_thread
config = get_thread_config(thread_id)
messages = recover_messages(thread_id)
rename_thread_if_needed(thread_id, messages)

st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
for index, message in enumerate(messages):
    render_message(message, thread_id, index)
st.markdown('</div>', unsafe_allow_html=True)

# Check for human in the loop interruption state
current_state = chatbot.get_state(config)
if current_state.next and "__interrupt__" in current_state.metadata:
    interrupt_value = current_state.values.get("__interrupt__", [None])[0] or current_state.metadata["__interrupt__"][0]
    
    st.warning("🚨 **Human-in-the-Loop Approval Required**")
    st.info(f"**Employee Request:** {interrupt_value.value.get('request', 'Action requested')}")
    
    with st.form("hitl_approval_form"):
        comment_input = st.text_input("Approver Comments / Reason:", placeholder="Optional reason details...")
        col1, col2 = st.columns(2)
        with col1:
            approve_btn = st.form_submit_button("✅ Approve Request", use_container_width=True)
        with col2:
            reject_btn = st.form_submit_button("❌ Reject Request", use_container_width=True)
            
        if approve_btn:
            handle_hitl_submission("approve", comment_input, config)
        elif reject_btn:
            handle_hitl_submission("reject", comment_input, config)

if st.session_state.get("processing"):
    st.info("Response in progress...")

prompt = st.chat_input("Ask about leave, benefits, policy, onboarding, or payroll...")
if prompt:
    if not st.session_state.get("processing"):
        st.session_state.processing = True
        st.session_state.last_response_meta.pop(thread_id, None)
        with st.spinner("Preparing response..."):
            stream_response(prompt, thread_id)
        st.session_state.processing = False
        st.rerun()