# Ask-HR

Ask-HR is an agentic AI employee support assistant designed to make HR information easy to access through natural conversation. The project combines LangGraph, Retrieval-Augmented Generation (RAG), ChromaDB, HuggingFace embeddings, Groq LLMs, and PostgreSQL checkpointing to answer employee questions using company HR policies and internal knowledge.

## Project Agenda

The goal of Ask-HR is to reduce dependency on manual HR support for repeated employee queries. Employees should be able to ask questions such as leave rules, attendance expectations, reimbursement process, onboarding steps, benefits, work-from-home policy, or company handbook details and receive quick, context-aware answers from verified company documents.

This project is being built as a foundation for a broader HR assistant that can eventually support policy retrieval, leave management, workflow automation, employee record lookup, and conversational task handling.

## Problem It Solves

In many organizations, HR teams repeatedly answer the same questions across policies, benefits, leave balances, attendance rules, onboarding, travel, reimbursements, and company processes. Employees often need to search through multiple documents or wait for HR replies.

Ask-HR solves this by:

- Providing a single conversational interface for HR-related questions.
- Retrieving answers from company policy documents instead of relying only on model memory.
- Keeping responses grounded in source documents using a vector database.
- Supporting conversation state through LangGraph checkpointing.
- Preparing the base for future HR workflow automation.

## Core Highlights

- **Agentic chatbot:** Built with LangGraph using a graph-based chat flow.
- **RAG pipeline:** Retrieves relevant HR policy content from a Chroma vector database.
- **Policy knowledge base:** Uses company HR markdown documents such as leave policy, attendance policy, benefits policy, employee handbook, travel policy, reimbursement policy, and FAQs.
- **Synthetic HR datasets:** Includes generated CSV data for employees, departments, projects, attendance, leave requests, leave balances, benefits, assets, skills, training records, and performance reviews.
- **LLM integration:** Uses Groq's `llama-3.1-8b-instant` model through LangChain.
- **Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` through HuggingFace embeddings.
- **Persistent memory:** Uses PostgreSQL with LangGraph checkpointing to maintain threaded conversations.
- **Local database setup:** Includes Docker Compose configuration for a PostgreSQL service.

## Current Architecture

```text
Employee Query
     |
     v
LangGraph Chatbot
     |
     +-- LLM decides whether HR document retrieval is needed
     |
     +-- Retriever Tool
           |
           v
       Chroma Vector DB
           |
           v
       Relevant HR Policy Context
     |
     v
LLM generates grounded response
     |
     v
PostgreSQL stores conversation checkpoint
```

## Repository Structure

```text
Ask-HR/
+-- AgenticHR/
|   +-- chatbot.py                 # Main LangGraph chatbot CLI
|   +-- docker-compose.yml          # PostgreSQL service for LangGraph checkpoints
|   +-- RAG_pipeline/
|       +-- rag.py                  # Retriever helper for policy vector search
|       +-- test.py                 # Basic retriever test
|       +-- PolicyVB/               # Chroma vector database
+-- Company Info/
|   +-- docs/                       # HR policy and handbook markdown files
|   +-- data/                       # Synthetic HR CSV datasets
|   +-- scripts/generate_hr_kb.py   # HR knowledge base generator
|   +-- GENERATE.md                 # Data generation notes
+-- requirements.txt
+-- LICENSE
+-- README.md
```

## Progress Completed So Far

- Created the initial project structure for Ask-HR.
- Added HR policy documents for:
  - Employee handbook
  - Leave policy
  - Attendance policy
  - Work-from-home policy
  - Reimbursement policy
  - Travel policy
  - Benefits policy
  - Promotion policy
  - Performance review policy
  - Onboarding guide
  - Code of conduct
  - FAQs
  - Company announcements
- Generated synthetic company HR datasets, including employees, departments, attendance, leave balances, leave requests, projects, benefits, skills, assets, training records, and performance reviews.
- Added a script to regenerate the HR knowledge base.
- Built a Chroma vector database for policy retrieval.
- Implemented a RAG retriever function for querying HR policy content.
- Implemented the LangGraph chatbot flow.
- Added a retriever tool that the LLM can call during conversation.
- Integrated Groq LLM support through LangChain.
- Added PostgreSQL checkpointing for persistent conversation threads.
- Added Docker Compose configuration for running PostgreSQL locally.
- Added basic retriever testing files and notebooks for experimentation.

## Features Available Now

- Ask HR policy questions through the CLI chatbot.
- Retrieve relevant company policy context using semantic search.
- Return source-aware HR policy snippets to the LLM.
- Maintain a conversation thread using LangGraph checkpoints.
- Regenerate sample HR documents and datasets when needed.

## Planned Work

- Add a FastAPI backend for serving the chatbot as an API.
- Build a frontend interface for employees and HR admins.
- Add leave balance lookup and leave request workflows.
- Add employee-specific access control and authentication.
- Improve answer formatting with clearer source citations.
- Add automated tests for the chatbot graph and retriever behavior.
- Add ingestion scripts for rebuilding the vector database from markdown documents.
- Expand HR workflows beyond document Q&A, such as approvals, onboarding tasks, and ticket creation.

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file with the required API key:

```env
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://postgres:postgres@localhost:5442/langgraph
```

If `DATABASE_URL` is not set, the chatbot uses:

```text
postgresql://postgres:postgres@localhost:5442/langgraph
```

### 4. Start PostgreSQL

```powershell
cd AgenticHR
docker compose up -d
```

### 5. Run the chatbot

From the `AgenticHR` directory:

```powershell
python chatbot.py
```

Type `exit`, `quit`, or `break` to stop the chatbot.

## Regenerating the HR Knowledge Base

The project includes a synthetic HR data generator.

```powershell
cd "Company Info"
python scripts/generate_hr_kb.py
```

This regenerates:

- `Company Info/data/`
- `Company Info/docs/`

## Tech Stack

- Python
- LangChain
- LangGraph
- Groq
- HuggingFace sentence-transformers
- ChromaDB
- PostgreSQL
- Docker Compose
- Pandas
- Faker

## Project Status

Ask-HR is currently in the prototype stage. The main RAG chatbot flow is implemented and can retrieve HR policy information from a local vector database. The next major step is to turn the working CLI prototype into a complete application with API endpoints, frontend UI, stronger workflow automation, and production-ready testing.
