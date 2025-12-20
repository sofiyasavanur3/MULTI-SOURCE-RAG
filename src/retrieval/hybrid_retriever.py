"""
Hybrid Retrieval - Fusion of Multiple Strategies

WHY THIS EXISTS:
- Combines strengths of different retrievers
- Vector search + Keyword search = Best of both worlds
- Reciprocal Rank Fusion for smart combining
- Higher accuracy and recall

WHAT IS RECIPROCAL RANK FUSION (RRF)?
- Algorithm to merge ranked lists
- Gives weight to position in each list
- Better than simple concatenation

EXAMPLE:
Vector results:   [Doc A, Doc C, Doc B]
BM25 results:     [Doc B, Doc A, Doc D]
Fused results:    [Doc A, Doc B, Doc C, Doc D]
                  (A appears high in both!)
"""

from typing import List, Dict
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Combines multiple retrievers using Reciprocal Rank Fusion
    
    WHY HYBRID?
    - Vector search: Good for concepts, meaning
    - BM25: Good for exact terms, numbers
    - Together: Comprehensive coverage!
    
    ANALOGY:
    Like asking two experts the same question:
    - One expert knows concepts (vector)
    - Other expert knows facts (BM25)
    - Combine their answers for best result
    """
    
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        weights: List[float] = None,
        similarity_top_k: int = 5
    ):
        """
        Initialize hybrid retriever
        
        Parameters:
        - retrievers: List of retrievers to combine
        - weights: Importance of each retriever (default: equal)
        - similarity_top_k: Final number of results
        
        WHY WEIGHTS?
        - Sometimes one retriever is more important
        - Example: [0.7, 0.3] = Vector 70%, BM25 30%
        - Default: Equal weights [0.5, 0.5]
        """
        self._retrievers = retrievers
        self._similarity_top_k = similarity_top_k
        
        # Default: equal weights
        # WHY? Usually both methods are equally valuable
        if weights is None:
            weights = [1.0 / len(retrievers)] * len(retrievers)
        
        if len(weights) != len(retrievers):
            raise ValueError("Number of weights must match number of retrievers")
        
        self._weights = weights
        
        logger.info(f"🔀 Hybrid Retriever initialized with {len(retrievers)} retrievers")
        logger.info(f"   Weights: {weights}")
    
    
    def _reciprocal_rank_fusion(
        self,
        results_list: List[List[NodeWithScore]],
        k: int = 60
    ) -> List[NodeWithScore]:
        """
        Fuse multiple result lists using RRF algorithm
        
        WHY RRF?
        - Simple and effective
        - Doesn't require score calibration
        - Used in production systems
        
        HOW IT WORKS:
        For each document:
        score = Σ (weight / (k + rank))
        
        Where:
        - rank = position in list (1st, 2nd, 3rd...)
        - k = constant (usually 60)
        - weight = importance of this retriever
        
        EXAMPLE:
        Doc A in List 1 at rank 1: score = 1/(60+1) = 0.016
        Doc A in List 2 at rank 3: score = 1/(60+3) = 0.016
        Total for Doc A: 0.016 + 0.016 = 0.032
        
        Parameters:
        - results_list: List of result lists from each retriever
        - k: RRF constant (higher = less weight to rank)
        
        Returns: Fused and re-ranked results
        """
        # Dictionary to accumulate scores
        # WHY DEFAULTDICT? Automatically initializes new entries
        node_scores = defaultdict(float)
        node_map = {}  # Store actual nodes
        
        # Process each retriever's results
        for weight, results in zip(self._weights, results_list):
            for rank, node_with_score in enumerate(results, start=1):
                node_id = node_with_score.node.node_id
                
                # Reciprocal rank score
                # WHY 1/(k+rank)? Higher ranks get higher scores
                rrf_score = weight * (1.0 / (k + rank))
                
                node_scores[node_id] += rrf_score
                node_map[node_id] = node_with_score.node
        
        # Sort by fused score
        # WHY SORT? We want best results first
        sorted_nodes = sorted(
            node_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create result list
        fused_results = []
        for node_id, score in sorted_nodes[:self._similarity_top_k]:
            fused_results.append(
                NodeWithScore(
                    node=node_map[node_id],
                    score=score
                )
            )
        
        return fused_results
    
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve using all retrievers and fuse results
        
        PROCESS:
        1. Query each retriever
        2. Collect all results
        3. Fuse using RRF
        4. Return top K
        
        Parameters:
        - query_bundle: Query information
        
        Returns: Fused and ranked results
        """
        query_text = query_bundle.query_str
        logger.info(f"🔍 Hybrid retrieval for: {query_text}")
        
        # Query all retrievers
        # WHY SEPARATE? Each retriever uses different strategy
        all_results = []
        for i, retriever in enumerate(self._retrievers):
            logger.info(f"   Querying retriever {i+1}/{len(self._retrievers)}...")
            results = retriever._retrieve(query_bundle)
            all_results.append(results)
            logger.info(f"   Found {len(results)} results")
        
        # Fuse results
        logger.info("🔀 Fusing results with RRF...")
        fused = self._reciprocal_rank_fusion(all_results)
        
        logger.info(f"✅ Hybrid retrieval complete: {len(fused)} final results")
        return fused


def demo_hybrid():
    """
    Demonstrates hybrid retrieval
    
    WHY?
    - Shows how fusion improves results
    - Compares individual vs combined
    """
    print("\n" + "="*60)
    print("🔀 HYBRID RETRIEVAL DEMO")
    print("="*60 + "\n")
    
    # This will be more impressive with real data
    # For now, shows the concept
    print("Hybrid retrieval combines:")
    print("  1. Vector search (semantic understanding)")
    print("  2. BM25 search (keyword matching)")
    print("  3. Reciprocal Rank Fusion (smart combining)")
    print()
    print("Result: Best of both worlds! 🎯")
    print()
    print("Next: We'll integrate this with your real RAG system!")
    print("="*60)


if __name__ == "__main__":
    demo_hybrid()