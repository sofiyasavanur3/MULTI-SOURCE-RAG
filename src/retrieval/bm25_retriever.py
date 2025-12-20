"""
BM25 Keyword-Based Retrieval

WHY THIS EXISTS:
- Complements vector search with keyword matching
- Better for exact terms, names, numbers
- Traditional IR algorithm (proven effective)
- Works without embeddings (no API cost!)

WHAT IS BM25?
- "Best Match 25" - a ranking algorithm
- Scores documents by keyword relevance
- Considers term frequency and document length
- Used by search engines before vector search

WHEN TO USE:
- User asks for specific names, dates, numbers
- Exact phrase matching needed
- Complement to semantic search
"""

from typing import List, Dict
from llama_index.core.schema import Document, NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """
    BM25-based keyword retriever
    
    WHY INHERIT BaseRetriever?
    - Compatible with LlamaIndex ecosystem
    - Can be used with query engines
    - Standard interface for all retrievers
    
    HOW BM25 WORKS:
    1. Tokenize documents (split into words)
    2. Build keyword index
    3. For query: score each doc by keyword overlap
    4. Rank by score
    
    EXAMPLE:
    Query: "CEO Priya"
    Doc 1: "CEO is Priya Sharma" → High score (both words!)
    Doc 2: "Company founded 2020" → Low score (no matches)
    """
    
    def __init__(
        self,
        nodes: List,
        similarity_top_k: int = 3,
        tokenizer=None
    ):
        """
        Initialize BM25 retriever
        
        Parameters:
        - nodes: List of document nodes to search
        - similarity_top_k: How many results to return
        - tokenizer: Function to split text (default: simple split)
        
        WHY NODES NOT DOCUMENTS?
        - LlamaIndex uses "nodes" (chunks with metadata)
        - More flexible than raw documents
        """
        self._nodes = nodes
        self._similarity_top_k = similarity_top_k
        
        # Default tokenizer: lowercase and split by spaces
        # WHY? Simple but effective for most cases
        self._tokenizer = tokenizer or (lambda text: text.lower().split())
        
        # Tokenize all documents
        # WHY? BM25 needs to analyze word frequencies
        logger.info(f"📊 Tokenizing {len(nodes)} documents for BM25...")
        self._corpus = [
            self._tokenizer(node.get_content()) 
            for node in nodes
        ]
        
        # Build BM25 index
        # WHY? Pre-compute statistics for fast querying
        logger.info("🔨 Building BM25 index...")
        self._bm25 = BM25Okapi(self._corpus)
        logger.info("✅ BM25 retriever initialized!")
    
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve documents for a query
        
        WHY THIS METHOD?
        - Required by BaseRetriever interface
        - Called automatically by query engine
        
        HOW IT WORKS:
        1. Tokenize query
        2. Score all documents
        3. Sort by score
        4. Return top K
        
        Parameters:
        - query_bundle: Contains query text
        
        Returns: List of scored nodes
        """
        query_text = query_bundle.query_str
        logger.info(f"🔍 BM25 searching for: {query_text}")
        
        # Tokenize query
        query_tokens = self._tokenizer(query_text)
        
        # Score all documents
        # WHY? BM25 algorithm computes relevance scores
        scores = self._bm25.get_scores(query_tokens)
        
        # Get top K indices
        # WHY argsort? Finds indices of highest scores
        top_indices = np.argsort(scores)[::-1][:self._similarity_top_k]
        
        # Create NodeWithScore objects
        # WHY? Standard format for LlamaIndex retrievers
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's some match
                node_with_score = NodeWithScore(
                    node=self._nodes[idx],
                    score=float(scores[idx])
                )
                results.append(node_with_score)
        
        logger.info(f"✅ BM25 found {len(results)} results")
        return results


def demo_bm25():
    """
    Demonstrates BM25 retrieval
    
    WHY?
    - Shows how BM25 differs from vector search
    - Educational comparison
    - Tests keyword matching
    """
    print("\n" + "="*60)
    print("📊 BM25 KEYWORD RETRIEVAL DEMO")
    print("="*60 + "\n")
    
    # Create sample documents
    # WHY THESE? Show BM25's strength with exact terms
    from llama_index.core.schema import TextNode
    
    docs = [
        TextNode(text="Priya Sharma is the CEO of TechCorp with salary $200,000"),
        TextNode(text="Raj Kumar is the CTO of TechCorp with salary $180,000"),
        TextNode(text="TechCorp was founded in 2020 and has 150 employees"),
        TextNode(text="The company revenue is $10 million in 2024"),
    ]
    
    # Initialize retriever
    retriever = BM25Retriever(docs, similarity_top_k=2)
    
    # Test queries
    queries = [
        "CEO salary",
        "CTO Raj",
        "2024 revenue",
        "founded when"
    ]
    
    print("🔍 Testing BM25 keyword matching:\n")
    
    for query in queries:
        print(f"❓ Query: {query}")
        
        from llama_index.core.schema import QueryBundle
        query_bundle = QueryBundle(query_str=query)
        results = retriever._retrieve(query_bundle)
        
        if results:
            print(f"   Top result (score: {results[0].score:.2f}):")
            print(f"   {results[0].node.text[:100]}...")
        else:
            print("   No results found")
        print()
    
    print("="*60)


if __name__ == "__main__":
    demo_bm25()


# **Save it!**

# ---

# ## 🧠 Understanding BM25

# ### **What is BM25?**

# **BM25 = "Best Matching 25"**
# - Traditional information retrieval algorithm
# - Used by search engines (Elasticsearch, etc.)
# - Scores documents by keyword relevance

# ### **How It Works:**

# **Formula (simplified):**
# ```
# Score = Σ (keyword_frequency × inverse_document_frequency)
# ```

# **Example:**
# ```
# Query: "CEO salary"
# Doc 1: "CEO is Priya, salary $200k" 
#        → "CEO" appears 1x, "salary" appears 1x
#        → Score: HIGH

# Doc 2: "Founded in 2020"
#        → No matches
#        → Score: 0