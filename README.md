Live App:
👉 https://multi-source-rag-uerfb5u3b39mhjhipt5o.streamlit.app

An enterprise-grade Retrieval-Augmented Generation (RAG) system that answers questions using internal documents with source citations, built with LlamaIndex + OpenAI and deployed on Streamlit Cloud.

🔍 What Problem This Solves

Large Language Models do not know company-specific or private data.

This system allows users to ask natural language questions over:

PDF documents

CSV / structured data

Web content

and receive grounded answers with citations, instead of hallucinated responses.

🧠 How It Works (High Level)

Documents are ingested and chunked

Text chunks are converted into embeddings

Relevant chunks are retrieved using vector + keyword search

Retrieved results are re-ranked for relevance

An LLM generates a final answer with source citations

✨ Key Capabilities

Multi-source ingestion (PDF, CSV, Web)

Hybrid retrieval (Vector + BM25)

Source-aware answers with citations

Interactive Streamlit web interface

Cloud-ready deployment

🛠️ Tech Stack

Python 3.11

LlamaIndex (RAG orchestration)

OpenAI GPT-3.5 / GPT-4

Streamlit (UI + deployment)

Hybrid Retrieval (Vector + BM25)

🏗️ System Architecture

This project follows a modular Retrieval-Augmented Generation (RAG) architecture designed for enterprise knowledge systems.

High-Level Components

Frontend (Streamlit)

File upload

Question input

Answer display with citations

Ingestion Layer

PDF parsing

CSV loading

Web content extraction

Text chunking and metadata tagging

Embedding Layer

Converts text chunks into vector embeddings using OpenAI

Index Storage

Persistent vector storage

Prevents re-indexing on every restart

Retrieval Layer

Vector similarity search

BM25 keyword search

Hybrid fusion of results

Re-Ranking Layer

Scores retrieved chunks by relevance

LLM Generation

Generates grounded answers

Includes source citations

🔄 RAG Query Flow

User asks a question

Question is converted into an embedding

Relevant document chunks are retrieved using:

Vector search (semantic)

BM25 search (keyword)

Results are merged and re-ranked

Top-ranked context is sent to the LLM

LLM generates an answer with citations

🔍 Why Hybrid Retrieval

Vector Search understands semantic meaning
Example: “CEO” ≈ “Chief Executive Officer”

BM25 Search captures exact keywords
Example: invoice numbers, codes, dates

Hybrid Retrieval combines both approaches to reduce missed answers
Result: higher accuracy and better recall

🚀 Deployment Architecture

Source code hosted on GitHub

Deployed using Streamlit Cloud

Python 3.11 runtime

Secure secrets management via Streamlit Secrets

No Docker, no servers, low operational overhead

🎯 Why This Matters

Reduces hallucinations

Scales to enterprise document collections

Reusable and extensible RAG architecture

Production-ready deployment

👩‍💻 Author

Sofiya Savaanur
GitHub: https://github.com/sofiyasavaanur3