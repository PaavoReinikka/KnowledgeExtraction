"""
Utilities for hybrid retrieval fusion, such as Reciprocal Rank Fusion (RRF).
This is crucial for combining lexical (BM25) and semantic results.
"""

from typing import List, Dict, Union

def reciprocal_rank_fusion(
    lexical_results: List[Union[str, tuple]],
    semantic_results: List[Union[str, tuple]],
    k: int = 60,
    sort = False
) -> List[tuple]:
    """
    Combines two lists of top results into one using Reciprocal Rank Fusion (RRF).
    Documents that appear high in both lists will have the highest final score.

    Args:
        lexical_results: List of docs or (doc, score) from keyword search.
        semantic_results: List of docs or (doc, score) from vector search.
        k: Smoothing factor (default 60).

    Returns:
        List of tuples (doc, rrf_score) sorted by highest rank.
    """
    rrf_scores = {}
    
    # Sort results if they are tuples (doc, score) to ensure rank is correct
    if sort and lexical_results and isinstance(lexical_results[0], tuple):
        lexical_results = sorted(lexical_results, key=lambda x: x[1], reverse=True)
    
    if sort and semantic_results and isinstance(semantic_results[0], tuple):
        semantic_results = sorted(semantic_results, key=lambda x: x[1], reverse=True)
    
    # Process lexical ranking
    for rank, item in enumerate(lexical_results):
        doc = item[0] if isinstance(item, tuple) else item
        rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
        
    # Process semantic ranking
    for rank, item in enumerate(semantic_results):
        doc = item[0] if isinstance(item, tuple) else item
        rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
        
    # Sort and return
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
