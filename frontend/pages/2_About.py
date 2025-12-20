"""
About Page

WHY?
- Explains what the system does
- Architecture overview
- Credits and documentation
"""

import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About Multi-Source RAG")

st.markdown("""
## 🎯 What is this?

This is an **Advanced Multi-Source RAG (Retrieval-Augmented Generation)** system 
that combines multiple data sources with sophisticated retrieval strategies to 
provide accurate, cited answers to your questions.

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────┐
│                  User Interface                     │
│              (Streamlit Web App)                    │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│             Ingestion Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │   PDFs   │  │   Web    │  │   CSVs   │        │
│  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│              Vector Index (FAISS)                   │
│         (Searchable Document Embeddings)            │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│            Retrieval Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Vector  │  │   BM25   │  │  Hybrid  │        │
│  │  Search  │  │  Search  │  │  Fusion  │        │
│  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│              Re-Ranking Layer                       │
│         (Score and re-order results)                │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│            Generation Layer (LLM)                   │
│         (Generate answer with citations)            │
└─────────────────────────────────────────────────────┘
```

## 🔑 Key Features

### 📚 Multi-Source Ingestion
- **PDFs**: Research papers, reports, manuals
- **Websites**: Documentation, blogs, articles
- **CSVs**: Structured data, databases

### 🔍 Advanced Retrieval
- **Vector Search**: Semantic understanding
- **BM25 Search**: Keyword matching
- **Hybrid Fusion**: Best of both worlds
- **Re-Ranking**: Quality optimization

### 🤖 Smart Generation
- Accurate answers with citations
- Source tracking
- Context-aware responses

## 🛠️ Tech Stack

- **Framework**: LlamaIndex
- **LLM**: OpenAI GPT-3.5/4
- **Vector DB**: FAISS
- **Frontend**: Streamlit
- **Language**: Python 3.11+

## 📖 How to Use

1. **Upload Data**: Add PDFs, CSVs, or website URLs
2. **Build Index**: Click "Build Knowledge Base"
3. **Ask Questions**: Type your question
4. **Get Answers**: See answer with source citations

## 💡 Tips

- Use **Hybrid mode** for best results
- Upload multiple sources for comprehensive answers
- Check sources to verify information
- Clear history to start fresh conversations

## 👨‍💻 Author

Built as a learning project to demonstrate advanced RAG techniques.

**GitHub**: [Your GitHub URL]

## 📄 License

MIT License - Feel free to use and modify!

---

*Version 1.0 - December 2024*
""")

# Fun stats
st.markdown("---")
st.subheader("📊 Fun Stats")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Lines of Code", "~2,000+")
with col2:
    st.metric("Components", "12")
with col3:
    st.metric("Retrieval Modes", "3")