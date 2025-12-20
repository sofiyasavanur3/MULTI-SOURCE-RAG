"""
Re-Ranking Module

WHY THIS EXISTS:
- Initial retrieval casts a wide net
- Re-ranking finds the BEST results
- Uses a specialized model to score relevance
- Final quality boost before LLM generation

WHAT IS RE-RANKING?
- Takes retrieved documents (say 10)
- Scores each for relevance to query
- Re-orders by score
- Returns top K (say 3)

WHY NOT JUST RETRIEVE 3 INITIALLY?
- Initial retrieval is fast but imprecise
- Re-ranking is slow but very accurate
- Strategy: Retrieve 10 fast → Re-rank to get best 3
"""

from typing import List, Tuple
from llama_index.core.schema import NodeWithScore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleReranker:
    """
    Simple re-ranking based on multiple signals
    
    WHY SIMPLE?
    - No additional API costs (unlike Cohere reranker)
    - Fast and effective
    - Uses signals already available
    
    SIGNALS USED:
    1. Original retrieval score
    2. Text length (not too short, not too long)
    3. Keyword overlap with query
    
    NOTE: For production, consider:
    - Cohere rerank API (very good but costs money)
    - Cross-encoder models (good, free, but slower)
    - This simple approach (fast, free, decent)
    """
    
    def __init__(
        self,
        score_weight: float = 0.5,
        length_weight: float = 0.2,
        keyword_weight: float = 0.3,
        ideal_length: int = 500
    ):
        """
        Initialize re-ranker
        
        Parameters:
        - score_weight: Weight for original retrieval score
        - length_weight: Weight for text length score  
        - keyword_weight: Weight for keyword overlap
        - ideal_length: Ideal text length in characters
        
        WHY THESE WEIGHTS?
        - score_weight high: Trust the retrievers
        - length_weight medium: Prefer complete answers
        - keyword_weight medium: Ensure relevance
        """
        self.score_weight = score_weight
        self.length_weight = length_weight
        self.keyword_weight = keyword_weight
        self.ideal_length = ideal_length
        
        # Normalize weights
        # WHY? Ensure they sum to 1.0
        total = score_weight + length_weight + keyword_weight
        self.score_weight /= total
        self.length_weight /= total
        self.keyword_weight /= total
        
        logger.info("🎯 Simple Reranker initialized")
        logger.info(f"   Weights - Score: {self.score_weight:.2f}, "
                   f"Length: {self.length_weight:.2f}, "
                   f"Keyword: {self.keyword_weight:.2f}")
    
    
    def _score_length(self, text: str) -> float:
        """
        Score based on text length
        
        WHY?
        - Too short: Incomplete information
        - Too long: Includes irrelevant info
        - Just right: Complete and focused
        
        FORMULA:
        - Score = 1.0 - |length - ideal| / ideal
        - Closer to ideal = higher score
        
        EXAMPLE:
        - ideal = 500
        - text = 450 chars → score = 1 - 50/500 = 0.9 ✅
        - text = 100 chars → score = 1 - 400/500 = 0.2 ❌
        - text = 1000 chars → score = 1 - 500/500 = 0.0 ❌
        
        Returns: Score between 0 and 1
        """
        length = len(text)
        diff = abs(length - self.ideal_length)
        score = max(0.0, 1.0 - (diff / self.ideal_length))
        return score
    
    
    def _score_keyword_overlap(self, query: str, text: str) -> float:
        """
        Score based on keyword overlap
        
        WHY?
        - Ensure document actually relates to query
        - Simple but effective
        
        HOW:
        - Extract keywords from query
        - Count how many appear in text
        - Normalize by query length
        
        EXAMPLE:
        Query: "CEO salary 2024"
        Text: "The CEO earns a salary of $200k in 2024"
        Keywords in text: CEO, salary, 2024 = 3/3 = 1.0 ✅
        
        Returns: Score between 0 and 1
        """
        # Simple tokenization
        # WHY lowercase? "CEO" and "ceo" should match
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        # Count overlapping keywords
        overlap = len(query_words.intersection(text_words))
        
        # Normalize by query length
        # WHY? Longer queries naturally have more overlaps
        if len(query_words) == 0:
            return 0.0
        
        score = overlap / len(query_words)
        return score
    
    
    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_k: int = 3
    ) -> List[NodeWithScore]:
        """
        Re-rank nodes by combined score
        
        PROCESS:
        1. For each node, calculate:
           - Original score
           - Length score
           - Keyword score
        2. Combine with weights
        3. Sort by combined score
        4. Return top K
        
        Parameters:
        - query: Original query
        - nodes: Retrieved nodes with scores
        - top_k: Number of results to return
        
        Returns: Re-ranked nodes (best first)
        """
        logger.info(f"🔄 Re-ranking {len(nodes)} nodes...")
        
        reranked = []
        
        for node in nodes:
            text = node.node.text
            
            # Get original score
            # WHY? Trust the retriever's judgment
            original_score = node.score if hasattr(node, 'score') and node.score else 0.5
            
            # Calculate additional scores
            length_score = self._score_length(text)
            keyword_score = self._score_keyword_overlap(query, text)
            
            # Combined score
            # WHY WEIGHTED? Different signals have different importance
            combined_score = (
                self.score_weight * original_score +
                self.length_weight * length_score +
                self.keyword_weight * keyword_score
            )
            
            # Create new NodeWithScore with combined score
            # WHY NEW? Don't modify original
            reranked_node = NodeWithScore(
                node=node.node,
                score=combined_score
            )
            reranked.append(reranked_node)
        
        # Sort by combined score (highest first)
        # WHY REVERSE? Higher scores are better
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        # Return top K
        result = reranked[:top_k]
        
        logger.info(f"✅ Re-ranked to top {len(result)} nodes")
        return result


def demo_reranker():
    """
    Demonstrates re-ranking
    
    WHY?
    - Shows how re-ranking improves results
    - Educational example
    """
    print("\n" + "="*60)
    print("🎯 RE-RANKING DEMO")
    print("="*60 + "\n")
    
    from llama_index.core.schema import TextNode
    
    # Create sample nodes
    # WHY THESE? Show different score patterns
    nodes = [
        NodeWithScore(
            node=TextNode(text="CEO Priya Sharma leads TechCorp"),
            score=0.8
        ),
        NodeWithScore(
            node=TextNode(text="The company was founded in 2020 and has grown rapidly with products in AI"),
            score=0.7
        ),
        NodeWithScore(
            node=TextNode(text="CEO"),  # Too short!
            score=0.9
        ),
        NodeWithScore(
            node=TextNode(text="Priya Sharma is the Chief Executive Officer of TechCorp, overseeing all operations"),
            score=0.6
        ),
    ]
    
    query = "Who is the CEO?"
    
    print(f"Query: {query}\n")
    print("Original scores:")
    for i, node in enumerate(nodes, 1):
        print(f"{i}. Score: {node.score:.2f} - {node.node.text[:50]}...")
    
    # Re-rank
    reranker = SimpleReranker()
    reranked = reranker.rerank(query, nodes, top_k=3)
    
    print("\nAfter re-ranking:")
    for i, node in enumerate(reranked, 1):
        print(f"{i}. Score: {node.score:.3f} - {node.node.text[:50]}...")
    
    print("\n💡 Notice how the very short result dropped in rank!")
    print("="*60)


if __name__ == "__main__":
    demo_reranker()