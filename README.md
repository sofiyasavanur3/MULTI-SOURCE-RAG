Multi-Source RAG System (Enterprise Knowledge Assistant)

Live App:
https://multi-source-rag-uerfb5u3b39mhjhipt5o.streamlit.app

An enterprise-grade Retrieval-Augmented Generation (RAG) system that answers questions using internal documents with source citations, built using LlamaIndex + OpenAI and deployed on Streamlit Cloud.


🔍 What Problem This Solves
LLMs don’t know company-specific data.
This system lets users ask natural language questions over PDFs, CSVs, and web content — and get grounded answers, not hallucinations.

🧠 How It Works (High Level)
Documents are ingested and chunked
Chunks are converted into embeddings
Relevant chunks are retrieved using vector + keyword search
Results are re-ranked
An LLM generates an answer with citations

✨ Key Capabilities
Multi-source ingestion (PDF, CSV, Web)
Hybrid retrieval (Vector + BM25)
Source-aware answers (citations)
Streamlit web interface
Cloud-ready deployment

🛠️ Tech Stack
Python 3.11
LlamaIndex (RAG orchestration)
OpenAI GPT-3.5 / GPT-4
Streamlit (UI + deployment)
Hybrid Retrieval (Vector + BM25)

🎯 Why This Matters
Reduces hallucinations
Scales to enterprise documents
Reusable RAG architecture
Production-ready deployment

👩‍💻 Author
Sofiya Savaanur
GitHub: https://github.com/sofiyasavaanur3

🔹 RAG Query Flow (Step-by-Step)
User Question
      │
      ▼
Convert Question → Embedding
      │
      ▼
Retrieve Top-K Chunks
(Vector + BM25)
      │
      ▼
Re-Rank Results
      │
      ▼
Send Context to LLM
      │
      ▼
Generate Answer
(with citations)

🔹 Why Hybrid Retrieval (Interview Gold)
Vector Search  → Semantic meaning
BM25 Search   → Exact keywords
Hybrid Fusion → Higher accuracy


Details:

Vector search understands meaning.
BM25 catches exact terms.
Hybrid retrieval reduces missed answers.

🔹 Deployment Architecture
GitHub Repository
       │
       ▼
Streamlit Cloud
       │
       ├── Python Runtime (3.11)
       ├── Dependency Install
       ├── Secrets Management
       │
       ▼
Live Web App


No Docker. No servers. Low operational cost.