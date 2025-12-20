"""
Multi-Source RAG Web Interface

WHY STREAMLIT?
- Fast to build (Python only, no HTML/CSS/JS)
- Beautiful UI out of the box
- Perfect for ML/AI demos
- Easy to deploy

FEATURES:
- File upload (PDF, CSV)
- URL ingestion
- Chat interface
- Source citations
- Multiple retrieval modes
- Knowledge base statistics
"""

import streamlit as st
import sys
from pathlib import Path
import os
from dotenv import load_dotenv
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment
load_dotenv()

# Import our modules
from src.ingestion.unified_manager import UnifiedIngestionManager
from src.retrieval.advanced_query_engine import AdvancedQueryEngine

# Page configuration
# WHY? Sets browser tab title, icon, layout
st.set_page_config(
    page_title="Multi-Source RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
# WHY? Make it look professional!
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    """
    Initialize Streamlit session state
    
    WHY SESSION STATE?
    - Streamlit reruns on every interaction
    - Session state preserves data between reruns
    - Like variables that persist
    
    WHAT WE STORE:
    - ingestion_manager: Handles data ingestion
    - query_engine: Handles queries
    - chat_history: Conversation history
    - indexed: Whether index is built
    """
    if 'ingestion_manager' not in st.session_state:
        st.session_state.ingestion_manager = UnifiedIngestionManager()
    
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'indexed' not in st.session_state:
        st.session_state.indexed = False
    
    if 'temp_dirs' not in st.session_state:
        st.session_state.temp_dirs = []


def save_uploaded_file(uploaded_file, target_dir):
    """
    Save uploaded file to disk
    
    WHY?
    - Streamlit gives us file in memory
    - Our ingesters need files on disk
    - Save temporarily, then process
    
    Parameters:
    - uploaded_file: Streamlit UploadedFile object
    - target_dir: Where to save
    
    Returns: Path to saved file
    """
    target_path = Path(target_dir) / uploaded_file.name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return target_path


def sidebar_data_ingestion():
    """
    Sidebar for data ingestion
    
    WHY SIDEBAR?
    - Keeps main area clean for chat
    - All data management in one place
    - Easy to find and use
    
    FEATURES:
    - File upload
    - URL input
    - Build index button
    - Statistics display
    """
    st.sidebar.title("📚 Data Sources")
    
    # File Upload Section
    st.sidebar.subheader("📄 Upload Files")
    
    uploaded_files = st.sidebar.file_uploader(
        "Choose PDF or CSV files",
        type=['pdf', 'csv'],
        accept_multiple_files=True,
        help="Upload documents to add to your knowledge base"
    )
    
    # URL Input Section
    st.sidebar.subheader("🌐 Add Websites")
    
    url_input = st.sidebar.text_area(
        "Enter URLs (one per line)",
        help="Add website URLs to scrape and add to knowledge base",
        height=100
    )
    
    # Build Index Button
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🔨 Build Knowledge Base", type="primary", use_container_width=True):
     with st.spinner("Building knowledge base..."):
        try:
            # IMPORTANT: Create a NEW ingestion manager to clear old data
            st.session_state.ingestion_manager = UnifiedIngestionManager()
            
            # Create temporary directories
            temp_pdf_dir = tempfile.mkdtemp()
            temp_csv_dir = tempfile.mkdtemp()
            st.session_state.temp_dirs.extend([temp_pdf_dir, temp_csv_dir])
            
            # Track what we're processing
            pdf_count = 0
            csv_count = 0
            
            # Process uploaded files
            if uploaded_files:
                st.sidebar.info(f"📤 Processing {len(uploaded_files)} uploaded file(s)...")
                for uploaded_file in uploaded_files:
                    if uploaded_file.name.endswith('.pdf'):
                        file_path = save_uploaded_file(uploaded_file, temp_pdf_dir)
                        pdf_count += 1
                        st.sidebar.success(f"✓ Saved: {uploaded_file.name}")
                    elif uploaded_file.name.endswith('.csv'):
                        file_path = save_uploaded_file(uploaded_file, temp_csv_dir)
                        csv_count += 1
                        st.sidebar.success(f"✓ Saved: {uploaded_file.name}")
            
            # Parse URLs
            urls = []
            if url_input.strip():
                urls = [url.strip() for url in url_input.strip().split('\n') if url.strip()]
                st.sidebar.info(f"🌐 Found {len(urls)} URL(s) to process")
            
            # IMPORTANT: Only process what user uploaded/entered
            total_items = pdf_count + csv_count + len(urls)
            
            if total_items == 0:
                st.sidebar.warning("⚠️ Please upload files or add URLs first!")
                return
            
            st.sidebar.info(f"📊 Processing: {pdf_count} PDFs, {csv_count} CSVs, {len(urls)} URLs")
            
            # Ingest PDFs from uploaded files ONLY
            if pdf_count > 0:
                st.sidebar.info("📄 Reading PDFs...")
                added = st.session_state.ingestion_manager.add_pdfs(temp_pdf_dir)
                st.sidebar.success(f"✓ Added {added} PDF chunks")
            
            # Ingest CSVs from uploaded files ONLY
            if csv_count > 0:
                st.sidebar.info("📊 Reading CSVs...")
                added = st.session_state.ingestion_manager.add_csvs(temp_csv_dir)
                st.sidebar.success(f"✓ Added {added} CSV rows")
            
            # Ingest URLs
            if urls:
                st.sidebar.info("🌐 Scraping websites...")
                added = st.session_state.ingestion_manager.add_urls(urls)
                st.sidebar.success(f"✓ Added {added} web pages")
            
            # Get all documents
            documents = st.session_state.ingestion_manager.get_documents()
            
            if len(documents) == 0:
                st.sidebar.error("❌ No content extracted. Please check your files/URLs.")
                return
            
            st.sidebar.info(f"📚 Total: {len(documents)} documents")
            
            # Show sample of what was indexed
            st.sidebar.info("📝 Sample from first document:")
            st.sidebar.code(documents[0].text[:200] + "...")
            
            # Build index
            st.sidebar.info("🔨 Building vector index (this may take a minute)...")
            st.session_state.query_engine = AdvancedQueryEngine(storage_dir="storage_streamlit")
            st.session_state.query_engine.load_or_create_index(documents)
            st.session_state.indexed = True
            
            st.sidebar.success(f"✅ Knowledge base built with {len(documents)} documents!")
            st.balloons()
            
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
            import traceback
            st.sidebar.error("Full error:")
            st.sidebar.code(traceback.format_exc())
    
    # Display Statistics
    if st.session_state.indexed:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Statistics")
        
        stats = st.session_state.ingestion_manager.get_statistics()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Total Docs", stats['total_documents'])
        with col2:
            st.metric("Unique Content", stats['unique_content_hashes'])
        
        # Document breakdown
        st.sidebar.write("**By Source:**")
        for source, count in stats['by_source'].items():
            if count > 0:
                st.sidebar.write(f"• {source.upper()}: {count}")


def main_chat_interface():
    """
    Main chat interface
    
    WHY CHAT FORMAT?
    - Natural interaction
    - Shows conversation history
    - Familiar to users (like ChatGPT)
    
    FEATURES:
    - Question input
    - Retrieval mode selector
    - Answer display
    - Source citations
    - Conversation history
    """
    st.title("🤖 Multi-Source RAG System")
    st.markdown("Ask questions about your uploaded documents and websites!")
    
    # Check if indexed
    if not st.session_state.indexed:
        st.info("👈 Upload files or add URLs in the sidebar, then click 'Build Knowledge Base' to get started!")
        
        # Show demo info
        st.markdown("---")
        st.subheader("🎯 What is this?")
        st.markdown("""
        This is an **Advanced Multi-Source RAG (Retrieval-Augmented Generation)** system that:
        
        - 📄 **Ingests** data from PDFs, CSVs, and websites
        - 🔍 **Searches** using multiple strategies (vector + keyword)
        - 🤖 **Generates** accurate answers with source citations
        - 🎯 **Re-ranks** results for maximum accuracy
        
        **How to use:**
        1. Upload PDF or CSV files in the sidebar
        2. Add website URLs (optional)
        3. Click "Build Knowledge Base"
        4. Ask questions!
        """)
        return
    
    # Retrieval mode selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_input(
            "Ask a question:",
            placeholder="e.g., Who is the CEO?",
            key="question_input"
        )
    
    with col2:
        retrieval_mode = st.selectbox(
            "Mode:",
            ["hybrid", "vector", "bm25"],
            help="""
            - **Hybrid**: Best results (combines vector + keyword)
            - **Vector**: Semantic search (understands meaning)
            - **BM25**: Keyword search (exact matches)
            """
        )
    
    # Query button
    if st.button("🔍 Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question!")
            return
        
        with st.spinner(f"Searching using {retrieval_mode} mode..."):
            try:
                # Query the engine
                result = st.session_state.query_engine.query(
                    question,
                    mode=retrieval_mode,
                    verbose=False
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'mode': retrieval_mode,
                    'sources': result['sources'],
                    'time': result['total_time']
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("💬 Conversation History")
        
        # Display in reverse order (newest first)
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                # Question
                st.markdown(f"**❓ Question:** {chat['question']}")
                st.markdown(f"*Mode: {chat['mode'].upper()} | Time: {chat['time']:.2f}s*")
                
                # Answer
                st.markdown(f"**🤖 Answer:**")
                st.info(chat['answer'])
                
                # Sources
                if chat['sources']:
                    with st.expander(f"📚 View {len(chat['sources'])} source(s)"):
                        for source in chat['sources']:
                            source_name = (
                                source['metadata'].get('filename') or
                                source['metadata'].get('url') or
                                source['metadata'].get('source_file') or
                                'Unknown'
                            )
                            
                            score_text = f" (Score: {source['score']:.3f})" if source['score'] else ""
                            
                            st.markdown(f"**Source {source['id']}: {source_name}**{score_text}")
                            st.text(source['text'])
                            st.markdown("---")
                
                st.markdown("---")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()


def cleanup_temp_dirs():
    """
    Clean up temporary directories
    
    WHY?
    - Free disk space
    - Remove uploaded files
    - Clean exit
    """
    for temp_dir in st.session_state.get('temp_dirs', []):
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


# Main app
def main():
    """
    Main application entry point
    
    WHY THIS STRUCTURE?
    - Initialize state first
    - Sidebar for data management
    - Main area for chat
    - Cleanup on exit
    """
    # Initialize
    init_session_state()
    
    # Layout
    sidebar_data_ingestion()
    main_chat_interface()
    
    # Cleanup (called when app closes)
    import atexit
    atexit.register(cleanup_temp_dirs)


if __name__ == "__main__":
    main()