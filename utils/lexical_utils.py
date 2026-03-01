"""
Lexical retrieval utilities, focusing on BM25 (Best Match 25) for keyword-based search.
BM25 is a robust alternative to TF-IDF and is often used as a baseline for keyword search.
"""

from typing import List, Optional
import numpy as np
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """
    A simple wrapper for BM25 retrieval using rank_bm25.
    """
    def __init__(self, corpus: List[str]):
        """
        Initialize the BM25 retriever with a corpus of documents.
        The documents are tokenized by space as a simple default.
        """
        self.corpus = corpus
        self.tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def get_relevant_documents(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Retrieves the top_k most relevant documents for a given query.
        Returns a list of dictionaries with 'text' and 'score'.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": self.corpus[idx],
                "score": float(scores[idx]),
                "index": int(idx)
            })
        return results

    def get_scores(self, query: str) -> np.ndarray:
        """
        Returns all scores for a given query, useful for hybrid fusion.
        """
        tokenized_query = query.lower().split()
        return self.bm25.get_scores(tokenized_query)
