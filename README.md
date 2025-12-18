# 🚀 Advanced Multi-Source RAG for Enterprise Knowledge Base

An enterprise-grade Retrieval-Augmented Generation (RAG) system that ingests data from multiple sources (PDFs, websites, structured databases), uses advanced retrieval techniques, and provides accurate answers with source citations.

## 🎯 Features

- **Multi-Source Ingestion**: PDFs, websites, CSV/databases
- **Advanced Retrieval**: Vector search, sentence window retrieval, graph-based knowledge retrieval
- **Intelligent Fusion**: Combines and re-ranks results from multiple retrievers
- **Source Citations**: Every answer includes references to source documents
- **Web Interface**: User-friendly Streamlit/Gradio interface
- **Production-Ready**: Deployable on AWS EC2/Azure

## 🛠️ Tech Stack

- **Framework**: LlamaIndex, LangChain
- **LLM**: OpenAI GPT-3.5/4 (or Claude)
- **Vector Database**: FAISS / Pinecone
- **Frontend**: Streamlit
- **Backend**: Python + FastAPI
- **Deployment**: Docker + AWS/Azure

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/multi-source-rag.git
cd multi-source-rag
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## 🚀 Quick Start

### Basic Usage
```bash
python src/simple_rag.py
```

### Run Web Interface
```bash
streamlit run frontend/app.py
```

## 📊 Project Structure
```
Multi-Source-RAG/
├── data/              # Data sources
├── src/               # Source code
│   ├── ingestion/     # Data ingestion modules
│   ├── retrieval/     # Retrieval strategies
│   ├── ranking/       # Re-ranking logic
│   └── api/           # Backend API
├── frontend/          # Web interface
├── storage/           # Vector stores
└── tests/             # Unit tests
```

## 🎓 Key Concepts

### What is RAG?
Retrieval-Augmented Generation combines the power of large language models with your own data, allowing the AI to answer questions based on your specific documents.

### Multi-Source Approach
Instead of relying on a single data source, this system integrates:
- PDFs (research papers, reports)
- Websites (documentation, blogs)
- Structured data (CSV files, databases)

### Advanced Retrieval
- **Vector Search**: Semantic similarity search
- **Sentence Window**: Context-aware retrieval
- **Graph-Based**: Knowledge graph traversal

## 📈 Roadmap

- [x] Basic RAG implementation
- [x] PDF ingestion
- [ ] Website scraping
- [ ] CSV/database integration
- [ ] Multiple retrieval strategies
- [ ] Fusion and re-ranking
- [ ] Web interface
- [ ] Deployment on cloud

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📝 License

MIT License

## 👨‍�💻 Author

Your Name - [GitHub Profile](https://github.com/YOUR_USERNAME)

## 🙏 Acknowledgments

- Built with [LlamaIndex](https://www.llamaindex.ai/)
- Inspired by enterprise RAG systems