"""
Vector Index & Retrieval System

WHY THIS EXISTS:
- Converts documents to searchable vectors
- Stores in efficient vector database (FAISS)
- Enables semantic search (meaning-based, not keyword)
- Provides query interface for RAG

HOW IT WORKS:
1. Take documents from ingestion
2. Create embeddings (text → vectors)
3. Store in FAISS index
4. Query: question → find similar vectors → retrieve documents
"""

"""
Vector Index & Retrieval System

WHY THIS EXISTS:
- Converts documents to searchable vectors
- Stores in efficient vector database (FAISS)
- Enables semantic search (meaning-based, not keyword)
- Provides query interface for RAG
"""

import os
from dotenv import load_dotenv

# Load environment variables FIRST
# WHY? OpenAI library needs API key before making any calls
load_dotenv()

# Verify API key exists
# WHY? Better error message than cryptic authentication error
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "❌ OPENAI_API_KEY not found!\n"
        "Please check your .env file contains:\n"
        "OPENAI_API_KEY=sk-your-key-here"
    )

from typing import List, Optional
from llama_index import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorIndexManager:
    """
    Manages vector indexing and retrieval
    
    WHY USE A CLASS?
    - Encapsulates index creation and querying
    - Manages storage and loading
    - Configurable parameters
    - Reusable across different document sets
    
    ANALOGY:
    Like a librarian who:
    1. Catalogs books (creates index)
    2. Remembers where everything is (stores index)
    3. Finds books when you ask (queries)
    """
    
    def __init__(
        self,
        storage_dir: str = "storage",
        model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
        temperature: float = 0.1
    ):
        """
        Initialize the vector index manager
        
        Parameters:
        - storage_dir: Where to save the index (default: "storage")
          WHY? Avoid recreating index every time (slow + costs money!)
        - model: LLM for generating answers (default: gpt-3.5-turbo)
          WHY? GPT-3.5 is fast and cheap, GPT-4 is better but expensive
        - embedding_model: Model for creating vectors
          WHY? text-embedding-ada-002 is OpenAI's best embedding model
        - temperature: Randomness (0=deterministic, 1=creative)
          WHY? Low temperature (0.1) for factual answers
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Configure LlamaIndex settings globally
        # WHY GLOBAL? Applied to all operations (indexing, querying)
        Settings.llm = OpenAI(
            model=model,
            temperature=temperature
        )
        Settings.embed_model = OpenAIEmbedding(
            model=embedding_model
        )
        
        self.index = None
        self.query_engine = None
        
        logger.info(f"🎯 Vector Index Manager initialized")
        logger.info(f"   • Storage: {storage_dir}")
        logger.info(f"   • LLM: {model}")
        logger.info(f"   • Embeddings: {embedding_model}")
    
    
    def create_index(
        self,
        documents: List[Document],
        persist: bool = True
    ) -> VectorStoreIndex:
        """
        Create vector index from documents
        
        WHY?
        - Converts all documents to searchable vectors
        - One-time process (unless documents change)
        - Can be saved and reloaded
        
        WHAT HAPPENS:
        1. For each document:
           - Text → Embedding API → Vector (1536 numbers)
        2. Store all vectors in FAISS index
        3. Save to disk (if persist=True)
        
        COST:
        - ~$0.0001 per 1000 tokens
        - 6 documents (~6000 chars) ≈ $0.0005
        
        Parameters:
        - documents: List of Document objects
        - persist: Save to disk? (default True)
        
        Returns: VectorStoreIndex
        """
        if not documents:
            raise ValueError("No documents provided for indexing")
        
        logger.info(f"🔨 Creating vector index from {len(documents)} documents...")
        logger.info(f"⏳ This will take ~{len(documents) * 0.5:.0f} seconds...")
        
        # Create the index
        # WHY VectorStoreIndex? It handles embeddings + FAISS automatically
        self.index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True  # WHY? User feedback for long operations
        )
        
        # Save to disk if requested
        # WHY? Avoid recreating (saves time + money) next time
        if persist:
            self.index.storage_context.persist(
                persist_dir=str(self.storage_dir)
            )
            logger.info(f"💾 Index saved to {self.storage_dir}")
        
        logger.info(f"✅ Index created successfully!")
        return self.index
    
    
    def load_index(self) -> VectorStoreIndex:
        """
        Load existing index from disk
        
        WHY?
        - Faster than recreating (no API calls!)
        - Free (no embedding costs)
        - Instant startup
        
        WHEN TO USE:
        - Index already exists
        - Documents haven't changed
        - Want to save time/money
        
        Returns: Loaded VectorStoreIndex
        """
        if not (self.storage_dir / "docstore.json").exists():
            raise FileNotFoundError(
                f"No index found in {self.storage_dir}. "
                f"Create one first with create_index()"
            )
        
        logger.info(f"📂 Loading index from {self.storage_dir}...")
        
        # Load the storage context
        # WHY? Contains index + metadata + embeddings
        storage_context = StorageContext.from_defaults(
            persist_dir=str(self.storage_dir)
        )
        
        # Load the index
        self.index = load_index_from_storage(storage_context)
        
        logger.info(f"✅ Index loaded successfully!")
        return self.index
    
    
    def get_or_create_index(
        self,
        documents: Optional[List[Document]] = None
    ) -> VectorStoreIndex:
        """
        Smart method: Load if exists, create if doesn't
        
        WHY?
        - Convenience! One call handles both cases
        - Saves money by loading when possible
        - Creates when necessary
        
        LOGIC:
        1. Try to load existing index
        2. If not found AND documents provided → create new
        3. If not found AND no documents → error
        
        Parameters:
        - documents: Documents to index (only needed if creating new)
        
        Returns: VectorStoreIndex
        """
        try:
            # Try loading first
            # WHY? Faster and free!
            return self.load_index()
        except FileNotFoundError:
            # Index doesn't exist
            if documents:
                logger.info("No existing index found. Creating new one...")
                return self.create_index(documents)
            else:
                raise ValueError(
                    "No existing index found and no documents provided. "
                    "Provide documents to create a new index."
                )
    
    
    def create_query_engine(
        self,
        similarity_top_k: int = 3,
        response_mode: str = "compact"
    ):
        """
        Create query engine for asking questions
        
        WHY?
        - Handles the entire RAG pipeline:
          Question → Search → Retrieve → Generate Answer
        - Configurable retrieval parameters
        - Returns formatted responses with citations
        
        Parameters:
        - similarity_top_k: How many documents to retrieve (default 3)
          WHY 3? Balance between context and relevance
          - Too few (1): Might miss important info
          - Too many (10): Includes irrelevant info, confuses LLM
        - response_mode: How to combine retrieved docs (default "compact")
          WHY "compact"? Efficiently combines multiple chunks
          Options: "compact", "tree_summarize", "simple_summarize"
        
        Returns: Query engine
        """
        if not self.index:
            raise ValueError(
                "No index available. Create or load an index first."
            )
        
        logger.info(f"🔍 Creating query engine...")
        logger.info(f"   • Retrieving top {similarity_top_k} documents")
        logger.info(f"   • Response mode: {response_mode}")
        
        # Create query engine
        # WHY as_query_engine? Convenience method that sets up everything
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode=response_mode
        )
        
        logger.info(f"✅ Query engine ready!")
        return self.query_engine
    
    
    def query(self, question: str, verbose: bool = True) -> dict:
        """
        Ask a question and get an answer
        
        WHY?
        - Main interface for RAG system
        - Returns answer + source information
        - Easy to use
        
        WHAT HAPPENS:
        1. Question → Embedding (vector)
        2. Search index for similar vectors
        3. Retrieve top K documents
        4. Send to LLM: "Based on these docs, answer: ..."
        5. LLM generates answer
        6. Return answer + sources
        
        Parameters:
        - question: User's question
        - verbose: Print detailed info (default True)
        
        Returns: Dict with answer and metadata
        """
        if not self.query_engine:
            raise ValueError(
                "Query engine not created. Call create_query_engine() first."
            )
        
        if verbose:
            logger.info(f"❓ Question: {question}")
        
        # Query the engine
        # WHY .query()? Runs the entire RAG pipeline
        response = self.query_engine.query(question)
        
        # Extract information
        # WHY? User wants answer + sources for verification
        result = {
            'question': question,
            'answer': str(response),
            'source_nodes': response.source_nodes if hasattr(response, 'source_nodes') else []
        }
        
        if verbose:
            logger.info(f"✅ Answer: {result['answer'][:200]}...")
            logger.info(f"📚 Used {len(result['source_nodes'])} source(s)")
        
        return result
    
    
    def query_with_sources(self, question: str) -> dict:
        """
        Query and return formatted answer with citations
        
        WHY?
        - User-friendly output
        - Shows source documents
        - Builds trust (verifiable answers)
        
        Returns: Dict with answer and formatted sources
        """
        result = self.query(question, verbose=False)
        
        # Format sources
        # WHY? Clean presentation of where answer came from
        sources = []
        for i, node in enumerate(result['source_nodes'], 1):
            source_info = {
                'id': i,
                'text': node.text[:200] + "...",  # Preview
                'metadata': node.metadata,
                'score': node.score if hasattr(node, 'score') else None
            }
            sources.append(source_info)
        
        return {
            'question': question,
            'answer': result['answer'],
            'sources': sources
        }


def demo_vector_index():
    """
    Demonstrates vector indexing and querying
    
    WHY?
    - Shows complete workflow
    - Tests with real data
    - Educational example
    """
    print("\n" + "="*60)
    print("🎯 VECTOR INDEX & RETRIEVAL DEMO")
    print("="*60 + "\n")
    
    # Import the unified manager
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.ingestion.unified_manager import UnifiedIngestionManager
    
    # Step 1: Ingest documents
    print("📥 STEP 1: Ingesting documents...")
    manager = UnifiedIngestionManager()
    counts = manager.add_all_sources()
    documents = manager.get_documents()
    print(f"✅ Ingested {len(documents)} documents\n")
    
    # Step 2: Create vector index
    print("🔨 STEP 2: Creating vector index...")
    index_manager = VectorIndexManager(storage_dir="storage")
    index_manager.create_index(documents, persist=True)
    print()
    
    # Step 3: Create query engine
    print("🔍 STEP 3: Creating query engine...")
    index_manager.create_query_engine(similarity_top_k=3)
    print()
    
    # Step 4: Ask questions!
    print("💬 STEP 4: Asking questions...")
    print("="*60 + "\n")
    
    questions = [
        "Who is the CEO?",
        "What is TechCorp?",
        "How many employees work in Engineering?",
        "What's the CTO's salary?"
    ]
    
    for question in questions:
        print(f"❓ {question}")
        result = index_manager.query_with_sources(question)
        print(f"🤖 {result['answer']}\n")
        
        if result['sources']:
            print(f"   📚 Sources:")
            for source in result['sources']:
                source_name = (
                    source['metadata'].get('filename') or
                    source['metadata'].get('url') or
                    source['metadata'].get('source_file') or
                    'Unknown'
                )
                print(f"      • {source_name}")
        print()
    
    print("="*60)
    print("✅ Demo complete!")
    print(f"💾 Index saved to 'storage/' directory")
    print(f"🔄 Next time, it will load from disk (much faster!)")


if __name__ == "__main__":
    demo_vector_index()