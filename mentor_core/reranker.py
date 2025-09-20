#!/usr/bin/env python3
"""
BGE Reranker implementation for enhanced search result ranking
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Try importing reranker dependencies
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class BGEReranker:
    """BGE Reranker v2 for enhancing search result ranking"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """Initialize BGE reranker with specified model"""
        self.model_name = model_name
        self.model = None
        self.available = False
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = CrossEncoder(model_name)
                self.available = True
                print(f"✅ BGE Reranker loaded: {model_name}")
            except Exception as e:
                print(f"⚠️  BGE Reranker failed to load: {e}")
                self.available = False
        else:
            print("⚠️  sentence-transformers not available, BGE reranker disabled")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank documents using BGE reranker v2
        
        Args:
            query: Search query text
            documents: List of document dictionaries with 'content' field
            top_k: Number of top results to return (None = return all)
            
        Returns:
            List of reranked documents with 'rerank_score' field added
        """
        if not self.available or not documents:
            # Fallback: return documents as-is
            return documents[:top_k] if top_k else documents
        
        try:
            # Prepare query-document pairs
            pairs = []
            for doc in documents:
                content = doc.get('content', '')
                # Truncate content to reasonable length for reranker
                content = content[:2000] if len(content) > 2000 else content
                pairs.append([query, content])
            
            # Get reranker scores
            scores = self.model.predict(pairs)
            
            # Add scores to documents
            reranked_docs = []
            for doc, score in zip(documents, scores):
                doc_copy = doc.copy()
                doc_copy['rerank_score'] = float(score)
                # Update similarity_score to rerank_score for unified ranking
                doc_copy['original_score'] = doc_copy.get('similarity_score', 0.0)
                doc_copy['similarity_score'] = float(score)
                reranked_docs.append(doc_copy)
            
            # Sort by rerank score (descending)
            reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            return reranked_docs[:top_k] if top_k else reranked_docs
            
        except Exception as e:
            print(f"Warning: BGE reranking failed: {e}")
            # Fallback: return original documents
            return documents[:top_k] if top_k else documents
    
    def batch_rerank(self, query_doc_pairs: List[Tuple[str, List[Dict[str, Any]]]], 
                     top_k: Optional[int] = None) -> List[List[Dict[str, Any]]]:
        """
        Rerank multiple query-document sets efficiently
        
        Args:
            query_doc_pairs: List of (query, documents) tuples
            top_k: Number of top results per query
            
        Returns:
            List of reranked document lists
        """
        results = []
        for query, documents in query_doc_pairs:
            reranked = self.rerank(query, documents, top_k)
            results.append(reranked)
        return results

# Singleton instance
_reranker_instance = None

def get_reranker() -> BGEReranker:
    """Get singleton BGE reranker instance"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = BGEReranker()
    return _reranker_instance

def rerank_documents(query: str, documents: List[Dict[str, Any]], 
                    top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Convenience function to rerank documents"""
    reranker = get_reranker()
    return reranker.rerank(query, documents, top_k)