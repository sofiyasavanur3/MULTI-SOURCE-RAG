"""
Advanced Query Engine with Multiple Retrieval Strategies

WHY THIS EXISTS:
- Integrates vector + BM25 + hybrid retrieval
- Provides multiple query modes
- Better accuracy than single-strategy
- Production-ready RAG system

FEATURES:
- Multiple retrieval strategies
- Configurable fusion weights
- Source tracking
- Performance comparison
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.schema import Document, QueryBundle
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from pathlib import Path
import logging
import time

# Load environment
load_dotenv()

# Import our custom retrievers
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedQueryEngine:
    """
    Advanced query engine with multiple retrieval strategies
    
    WHY THIS CLASS?
    - Manages different retrieval modes
    - Compares performance
    - Easy switching between strategies
    - Production-ready interface
    
    MODES:
    1. Vector only (semantic)
    2. BM25 only (keyword)
    3. Hybrid (best of both)
    """
    
    def __init__(
        self,
        storage_dir: str = "storage",
        model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
        temperature: float = 0.1
    ):
        """
        Initialize advanced query engine
        
        Parameters:
        - storage_dir: Where vector index is stored
        - model: LLM for generation
        - embedding_model: Model for embeddings
        - temperature: Response randomness
        """
        self.storage_dir = Path(storage_dir)
        
        # Configure LlamaIndex
        Settings.llm = OpenAI(model=model, temperature=temperature)
        Settings.embed_model = OpenAIEmbedding(model=embedding_model)
        
        self.index = None
        self.nodes = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self.hybrid_retriever = None
        
        logger.info("🎯 Advanced Query Engine initialized")
        logger.info(f"   Model: {model}")
        logger.info(f"   Storage: {storage_dir}")
    
    
    def load_or_create_index(
        self,
        documents: Optional[List[Document]] = None
    ):
        """
        Load existing index or create new one
        
        WHY?
        - Saves time and money by loading when possible
        - Creates when necessary
        - Sets up all retrievers
        
        Parameters:
        - documents: Only needed if creating new index
        """
        # Try to load existing index
        if (self.storage_dir / "docstore.json").exists():
            logger.info(f"📂 Loading existing index from {self.storage_dir}...")
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.storage_dir)
            )
            self.index = load_index_from_storage(storage_context)
            logger.info("✅ Index loaded!")
        else:
            if not documents:
                raise ValueError(
                    "No existing index found and no documents provided. "
                    "Provide documents to create new index."
                )
            
            logger.info(f"🔨 Creating new index from {len(documents)} documents...")
            self.index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True
            )
            self.index.storage_context.persist(persist_dir=str(self.storage_dir))
            logger.info("✅ Index created and saved!")
        
        # Extract nodes for BM25
        # WHY? BM25 needs access to all document chunks
        logger.info("📊 Extracting nodes for BM25...")
        self.nodes = list(self.index.docstore.docs.values())
        logger.info(f"✅ Extracted {len(self.nodes)} nodes")
        
        # Initialize all retrievers
        self._setup_retrievers()
    
    
    def _setup_retrievers(self, similarity_top_k: int = 5):
        """
        Set up all retrieval strategies
        
        WHY SEPARATE METHOD?
        - Clean initialization
        - Can reconfigure easily
        - All retrievers in one place
        
        Parameters:
        - similarity_top_k: Number of results per retriever
        """
        logger.info("🔧 Setting up retrievers...")
        
        # 1. Vector retriever (semantic search)
        # WHY? Understands meaning and context
        self.vector_retriever = self.index.as_retriever(
            similarity_top_k=similarity_top_k
        )
        logger.info("   ✅ Vector retriever ready")
        
        # 2. BM25 retriever (keyword search)
        # WHY? Good for exact terms, names, numbers
        self.bm25_retriever = BM25Retriever(
            nodes=self.nodes,
            similarity_top_k=similarity_top_k
        )
        logger.info("   ✅ BM25 retriever ready")
        
        # 3. Hybrid retriever (fusion)
        # WHY? Combines strengths of both
        self.hybrid_retriever = HybridRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.5, 0.5],  # Equal weight to both
            similarity_top_k=similarity_top_k
        )
        logger.info("   ✅ Hybrid retriever ready")
        
        logger.info("✅ All retrievers initialized!")
    
    
    def query(
        self,
        question: str,
        mode: str = "hybrid",
        similarity_top_k: int = 5,
        verbose: bool = True
    ) -> Dict:
        """
        Query with specified retrieval mode
        
        MODES:
        - "vector": Semantic search only
        - "bm25": Keyword search only
        - "hybrid": Fusion of both (recommended!)
        
        WHY DIFFERENT MODES?
        - Flexibility for different use cases
        - Performance comparison
        - Testing and debugging
        
        Parameters:
        - question: User's question
        - mode: Retrieval strategy ("vector", "bm25", "hybrid")
        - similarity_top_k: Number of results
        - verbose: Print detailed info
        
        Returns: Dict with answer, sources, timing
        """
        if not self.index:
            raise ValueError("Index not loaded. Call load_or_create_index() first.")
        
        # Select retriever based on mode
        # WHY DICT? Clean way to map mode to retriever
        retrievers = {
            "vector": self.vector_retriever,
            "bm25": self.bm25_retriever,
            "hybrid": self.hybrid_retriever
        }
        
        if mode not in retrievers:
            raise ValueError(f"Invalid mode: {mode}. Choose from: {list(retrievers.keys())}")
        
        retriever = retrievers[mode]
        
        if verbose:
            logger.info(f"🔍 Querying with mode: {mode.upper()}")
            logger.info(f"❓ Question: {question}")
        
        # Time the retrieval
        # WHY? Performance monitoring
        start_time = time.time()
        
        # Retrieve relevant documents
        query_bundle = QueryBundle(query_str=question)
        retrieved_nodes = retriever.retrieve(query_bundle)
        
        retrieval_time = time.time() - start_time
        
        if verbose:
            logger.info(f"⏱️  Retrieval time: {retrieval_time:.2f}s")
            logger.info(f"📚 Retrieved {len(retrieved_nodes)} nodes")
        
        # Generate answer using retrieved context
        # WHY? This is the "generation" part of RAG
        start_time = time.time()
        
        # Build context from retrieved nodes
        # WHY? LLM needs the relevant text to answer
        context = "\n\n".join([
            f"[Source {i+1}]: {node.node.text}"
            for i, node in enumerate(retrieved_nodes)
        ])
        
        # Create prompt for LLM
        # WHY THIS PROMPT? Encourages accurate, cited answers
        prompt = f"""Based on the following context, answer the question. 
If the answer is not in the context, say "I don't have enough information to answer this question."
Always cite which source(s) you used.

Context:
{context}

Question: {question}

Answer:"""
        
        # Get LLM response
        llm = Settings.llm
        response = llm.complete(prompt)
        
        generation_time = time.time() - start_time
        
        if verbose:
            logger.info(f"⏱️  Generation time: {generation_time:.2f}s")
            logger.info(f"✅ Answer generated!")
        
        # Format sources
        # WHY? User wants to verify information
        sources = []
        for i, node in enumerate(retrieved_nodes):
            source_info = {
                'id': i + 1,
                'text': node.node.text[:200] + "...",
                'score': node.score if hasattr(node, 'score') else None,
                'metadata': node.node.metadata
            }
            sources.append(source_info)
        
        return {
            'question': question,
            'answer': str(response),
            'mode': mode,
            'sources': sources,
            'num_sources': len(sources),
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': retrieval_time + generation_time
        }
    
    
    def compare_modes(
        self,
        question: str,
        modes: List[str] = ["vector", "bm25", "hybrid"]
    ) -> Dict[str, Dict]:
        """
        Compare different retrieval modes on same question
        
        WHY?
        - Shows which mode works best
        - Educational comparison
        - Helps tune your system
        
        EXAMPLE OUTPUT:
        {
            "vector": {"answer": "...", "time": 0.5},
            "bm25": {"answer": "...", "time": 0.3},
            "hybrid": {"answer": "...", "time": 0.8}
        }
        
        Parameters:
        - question: Question to test
        - modes: List of modes to compare
        
        Returns: Dict with results for each mode
        """
        logger.info(f"🔬 Comparing modes for: {question}")
        
        results = {}
        for mode in modes:
            logger.info(f"\n--- Testing {mode.upper()} mode ---")
            result = self.query(question, mode=mode, verbose=False)
            results[mode] = result
        
        return results


def demo_advanced_query_engine():
    """
    Demonstrates advanced query engine with all modes
    
    WHY?
    - Shows complete system in action
    - Compares different strategies
    - Proves hybrid is better!
    """
    print("\n" + "="*70)
    print("🎯 ADVANCED QUERY ENGINE DEMO")
    print("="*70 + "\n")
    
    # Import unified manager
    from src.ingestion.unified_manager import UnifiedIngestionManager
    
    # Step 1: Ingest documents
    print("📥 STEP 1: Ingesting documents...")
    manager = UnifiedIngestionManager()
    counts = manager.add_all_sources()
    documents = manager.get_documents()
    print(f"✅ Ingested {len(documents)} documents\n")
    
    # Step 2: Initialize advanced engine
    print("🔧 STEP 2: Setting up advanced query engine...")
    engine = AdvancedQueryEngine(storage_dir="storage")
    engine.load_or_create_index(documents)
    print()
    
    # Step 3: Test different modes
    print("🔬 STEP 3: Testing different retrieval modes...")
    print("="*70 + "\n")
    
    # Test questions
    # WHY THESE? Show different strengths
    test_cases = [
        {
            "question": "Who is the CEO?",
            "why": "Tests: Exact name matching"
        },
        {
            "question": "What does the company do?",
            "why": "Tests: Semantic understanding"
        },
        {
            "question": "How many people work in Engineering?",
            "why": "Tests: Number + keyword matching"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        question = test["question"]
        
        print(f"📝 TEST CASE {i}: {question}")
        print(f"   Purpose: {test['why']}\n")
        
        # Compare all modes
        results = engine.compare_modes(question)
        
        # Display results
        for mode in ["vector", "bm25", "hybrid"]:
            result = results[mode]
            print(f"   [{mode.upper()}]")
            print(f"   Answer: {result['answer'][:150]}...")
            print(f"   Time: {result['total_time']:.2f}s")
            print(f"   Sources: {result['num_sources']}")
            print()
        
        print("-" * 70 + "\n")
    
    # Step 4: Recommendation
    print("="*70)
    print("💡 RECOMMENDATION:")
    print("   Use HYBRID mode for best results!")
    print("   - Combines semantic understanding (vector)")
    print("   - With exact matching (BM25)")
    print("   - Reciprocal Rank Fusion ensures best results first")
    print("="*70 + "\n")


if __name__ == "__main__":
    demo_advanced_query_engine()