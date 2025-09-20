import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
import os
import uuid
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Import reranker (with graceful fallback if not available)
try:
    from .reranker import rerank_documents
except ImportError:
    rerank_documents = None

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class VectorSearch:
    """Enhanced vector search with Qdrant and BM25 integration"""
    
    def __init__(self, collection_name: str = "ai_mentor_docs", qdrant_host: str = "localhost", qdrant_port: int = 6333):
        # Qdrant client (local in-memory by default)
        self.client = QdrantClient(":memory:")  # In-memory for simplicity, can change to "localhost" for persistent
        self.collection_name = collection_name
        
        # Document metadata and BM25 index
        self.documents = []  # Store document metadata
        self.embeddings = []  # Backward compatibility for legacy search
        self.bm25_corpus = []  # Preprocessed text corpus for BM25
        self.bm25_index = None  # BM25Okapi index
        self.stopwords = set(stopwords.words('english'))
        
        # Backward compatibility files
        self.index_file = "data/embeddings/search_index.json"
        self.embeddings_file = "data/embeddings/embeddings.npy"
        
        # Collection configuration for dual vectors
        self.vector_config = {
            "content": VectorParams(size=384, distance=Distance.COSINE),  # Main content embedding
            "summary": VectorParams(size=384, distance=Distance.COSINE)   # Title/summary embedding
        }
        
        # Initialize collections and load existing data
        self._setup_qdrant_collection()
        self.load_index()
    
    def _setup_qdrant_collection(self):
        """Setup Qdrant collection with dual vector support"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            
            if not collection_exists:
                # Create collection with dual vectors
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=self.vector_config
                )
                print(f"Created Qdrant collection: {self.collection_name}")
            else:
                print(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            print(f"Warning: Qdrant setup failed: {e}. Using fallback mode.")
            
    def _preprocess_text_for_bm25(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing"""
        if not text:
            return []
        
        # Tokenize and lowercase
        try:
            tokens = word_tokenize(text.lower())
        except:
            # Fallback if NLTK tokenization fails
            tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens 
                 if token not in self.stopwords and len(token) > 2]
        
        return tokens
    
    def _rebuild_bm25_index(self):
        """Rebuild BM25 index from current documents"""
        if not self.documents:
            self.bm25_index = None
            return
            
        try:
            # Rebuild corpus
            self.bm25_corpus = []
            for doc in self.documents:
                content = doc.get('content', '')
                title = doc.get('title', '')
                section = doc.get('section', '')
                keywords = ' '.join(doc.get('keywords', []))
                
                # Combine different text fields with weighting
                combined_text = f"{title} {title} {section} {content} {keywords}"
                tokens = self._preprocess_text_for_bm25(combined_text)
                self.bm25_corpus.append(tokens)
            
            # Create BM25 index
            if self.bm25_corpus:
                self.bm25_index = BM25Okapi(self.bm25_corpus)
                print(f"Built BM25 index with {len(self.bm25_corpus)} documents")
        except Exception as e:
            print(f"Warning: BM25 index build failed: {e}")
            self.bm25_index = None
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[np.ndarray], 
                     summary_embeddings: Optional[List[np.ndarray]] = None):
        """Add documents and their embeddings to the search index with dual vector support"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        if summary_embeddings and len(summary_embeddings) != len(documents):
            raise ValueError("Number of summary embeddings must match number of documents")
        
        # Prepare points for Qdrant
        points = []
        start_idx = len(self.documents)
        
        for i, (doc, content_embedding) in enumerate(zip(documents, embeddings)):
            point_id = str(uuid.uuid4())
            doc['point_id'] = point_id
            doc['vector_index'] = start_idx + i
            
            # Prepare dual vectors
            vectors = {"content": content_embedding.tolist()}
            
            # Use summary embedding if provided, otherwise use content embedding
            if summary_embeddings:
                vectors["summary"] = summary_embeddings[i].tolist()
            else:
                # Generate simple summary vector from title + section
                title = doc.get('title', '')
                section = doc.get('section', '')
                summary_text = f"{title} {section}"
                
                # For now, use content embedding as fallback
                vectors["summary"] = content_embedding.tolist()
            
            # Create Qdrant point
            point = PointStruct(
                id=point_id,
                vector=vectors,
                payload={
                    'title': doc.get('title', ''),
                    'content': doc.get('content', ''),
                    'doc_type': doc.get('doc_type', ''),
                    'language': doc.get('language', ''),
                    'course': doc.get('course', ''),
                    'semester': doc.get('semester', ''),
                    'section': doc.get('section', ''),
                    'page': doc.get('page'),
                    'keywords': doc.get('keywords', []),
                    'vector_index': start_idx + i
                }
            )
            points.append(point)
        
        # Add to Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        except Exception as e:
            print(f"Warning: Qdrant upsert failed: {e}. Using fallback storage.")
        
        # Add to local storage for backward compatibility and BM25
        self.documents.extend(documents)
        
        # Rebuild BM25 index
        self._rebuild_bm25_index()
        
        # Save to disk for backward compatibility
        self.save_index()
        
        print(f"Added {len(documents)} documents to enhanced search index (Qdrant + BM25)")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10, 
               similarity_threshold: float = 0.1, vector_name: str = "content") -> List[Dict[str, Any]]:
        """Search for similar documents using Qdrant vector similarity"""
        try:
            # Search using Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=(vector_name, query_embedding.tolist()),
                limit=top_k,
                score_threshold=similarity_threshold
            )
            
            results = []
            for result in search_results:
                # Merge payload with similarity score
                doc_data = result.payload.copy() if result.payload else {}
                doc_data['similarity_score'] = float(result.score)
                doc_data['point_id'] = result.id
                
                # Add content from local storage if available
                vector_idx = doc_data.get('vector_index')
                if vector_idx is not None and vector_idx < len(self.documents):
                    local_doc = self.documents[vector_idx]
                    doc_data.update({
                        'content': local_doc.get('content', ''),
                        'chunk_id': local_doc.get('chunk_id', ''),
                        'source_id': local_doc.get('source_id', ''),
                        'start_char': local_doc.get('start_char'),
                        'end_char': local_doc.get('end_char')
                    })
                
                results.append(doc_data)
            
            return results
            
        except Exception as e:
            print(f"Warning: Qdrant search failed: {e}. Using fallback search.")
            return self._fallback_search(query_embedding, top_k, similarity_threshold)
    
    def _fallback_search(self, query_embedding: np.ndarray, top_k: int = 10, 
                        similarity_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Fallback to original in-memory search if Qdrant fails"""
        if not self.documents:
            return []
        
        # Use simple cosine similarity with stored embeddings if available
        # For now, return empty since we're transitioning to Qdrant
        return []
    
    def hybrid_search(self, query_embedding: np.ndarray, query_text: str, 
                     top_k: int = 10, alpha: float = 0.7, vector_name: str = "content") -> List[Dict[str, Any]]:
        """Enhanced hybrid search combining Qdrant vector similarity and BM25"""
        if not self.documents:
            return []
        
        # First try dense search with very low threshold
        dense_results = self.search(query_embedding, top_k=top_k * 3, similarity_threshold=0.001, vector_name=vector_name)
        
        # BM25 sparse search  
        sparse_results = self._bm25_search(query_text, top_k=top_k * 3)
        
        # If both searches return nothing, use keyword fallback
        if not dense_results and not sparse_results:
            print(f"No dense/sparse results found, using keyword fallback for: {query_text}")
            return self._keyword_fallback_search(query_text, top_k)
        
        # Combine scores using hybrid approach
        combined_scores = {}
        
        # Add dense scores
        for result in dense_results:
            doc_id = result.get('vector_index', result.get('point_id'))
            if doc_id is not None:
                combined_scores[doc_id] = {
                    'dense_score': result['similarity_score'],
                    'sparse_score': 0.0,
                    'doc_data': result
                }
        
        # Add sparse scores
        for result in sparse_results:
            doc_id = result.get('vector_index')
            if doc_id is not None:
                sparse_score = result.get('bm25_score', 0)
                if doc_id in combined_scores:
                    combined_scores[doc_id]['sparse_score'] = sparse_score
                else:
                    combined_scores[doc_id] = {
                        'dense_score': 0.0,
                        'sparse_score': sparse_score,
                        'doc_data': result
                    }
        
        # Calculate final hybrid scores
        final_results = []
        for doc_id, scores in combined_scores.items():
            hybrid_score = alpha * scores['dense_score'] + (1 - alpha) * scores['sparse_score']
            
            result_doc = scores['doc_data'].copy()
            result_doc['similarity_score'] = hybrid_score
            result_doc['dense_score'] = scores['dense_score']
            result_doc['sparse_score'] = scores['sparse_score']
            
            final_results.append(result_doc)
        
        # Sort by hybrid score and return top-k
        final_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return final_results[:top_k]
    
    def _keyword_fallback_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Fallback keyword search when all else fails"""
        if not self.documents:
            return []
        
        query_words = set(query.lower().split())
        results = []
        
        for i, doc in enumerate(self.documents):
            content = doc.get('content', '').lower()
            title = doc.get('title', '').lower()
            
            # Simple keyword matching score
            content_matches = sum(1 for word in query_words if word in content)
            title_matches = sum(1 for word in query_words if word in title) * 2  # Title gets more weight
            
            total_matches = content_matches + title_matches
            
            if total_matches > 0:
                # Calculate a simple score
                score = total_matches / len(query_words)
                
                result = doc.copy()
                result['similarity_score'] = score
                result['vector_index'] = i
                results.append(result)
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
    
    def _bm25_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """BM25-based sparse search"""
        if not self.bm25_index or not query.strip():
            return []
        
        try:
            # Preprocess query
            query_tokens = self._preprocess_text_for_bm25(query)
            if not query_tokens:
                return []
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.documents) and scores[idx] > 0:
                    result = self.documents[idx].copy()
                    result['bm25_score'] = float(scores[idx])
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Warning: BM25 search failed: {e}")
            return []
    
    def _keyword_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Legacy keyword search - now redirects to BM25"""
        return self._bm25_search(query, top_k)
    
    def dual_vector_search(self, content_embedding: np.ndarray, summary_embedding: Optional[np.ndarray] = None,
                          top_k: int = 10, content_weight: float = 0.7) -> List[Dict[str, Any]]:
        """Search using dual vectors (content + summary) with weighted combination"""
        if summary_embedding is None:
            # Use only content vector
            return self.search(content_embedding, top_k=top_k, vector_name="content")
        
        try:
            # Search with content vector
            content_results = self.search(content_embedding, top_k=top_k * 2, vector_name="content")
            
            # Search with summary vector  
            summary_results = self.search(summary_embedding, top_k=top_k * 2, vector_name="summary")
            
            # Combine results with weighting
            combined_scores = {}
            
            # Add content scores
            for result in content_results:
                doc_id = result.get('point_id', result.get('vector_index'))
                if doc_id:
                    combined_scores[doc_id] = {
                        'content_score': result['similarity_score'],
                        'summary_score': 0.0,
                        'doc_data': result
                    }
            
            # Add summary scores
            for result in summary_results:
                doc_id = result.get('point_id', result.get('vector_index'))
                if doc_id:
                    if doc_id in combined_scores:
                        combined_scores[doc_id]['summary_score'] = result['similarity_score']
                    else:
                        combined_scores[doc_id] = {
                            'content_score': 0.0,
                            'summary_score': result['similarity_score'],
                            'doc_data': result
                        }
            
            # Calculate weighted scores
            final_results = []
            for doc_id, scores in combined_scores.items():
                weighted_score = (content_weight * scores['content_score'] + 
                                (1 - content_weight) * scores['summary_score'])
                
                result_doc = scores['doc_data'].copy()
                result_doc['similarity_score'] = weighted_score
                result_doc['content_score'] = scores['content_score']
                result_doc['summary_score'] = scores['summary_score']
                
                final_results.append(result_doc)
            
            # Sort and return top-k
            final_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return final_results[:top_k]
            
        except Exception as e:
            print(f"Warning: Dual vector search failed: {e}")
            return self.search(content_embedding, top_k=top_k, vector_name="content")
    
    def filter_by_metadata(self, results: List[Dict[str, Any]], 
                          filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter search results by metadata"""
        filtered_results = []
        
        for result in results:
            match = True
            
            for key, value in filters.items():
                if key in result:
                    if isinstance(value, list):
                        if result[key] not in value:
                            match = False
                            break
                    else:
                        if result[key] != value:
                            match = False
                            break
            
            if match:
                filtered_results.append(result)
        
        return filtered_results
    
    def get_diverse_results(self, results: List[Dict[str, Any]], 
                           lambda_param: float = 0.2) -> List[Dict[str, Any]]:
        """Apply Maximal Marginal Relevance (MMR) for diverse results"""
        if len(results) <= 1:
            return results
        
        selected = [results[0]]  # Start with the most relevant
        remaining = results[1:]
        
        while remaining and len(selected) < len(results):
            best_score = -1
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                # Relevance score
                relevance = candidate.get('similarity_score', 0)
                
                # Diversity score (negative similarity to already selected)
                max_similarity = 0
                for selected_doc in selected:
                    # Simple text-based similarity for diversity
                    content_sim = self._text_similarity(
                        candidate.get('content', ''),
                        selected_doc.get('content', '')
                    )
                    max_similarity = max(max_similarity, content_sim)
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
            else:
                break
        
        return selected
    
    def rerank_and_mmr(self, query: str, query_embedding: np.ndarray, 
                       search_method: str = "hybrid", top_k_initial: int = 30,
                       top_k_rerank: int = 10, top_k_final: int = 6,
                       lambda_param: float = 0.2, **search_kwargs) -> List[Dict[str, Any]]:
        """
        Complete rerank + MMR pipeline for Task 6:
        1. Search top_k=30 results using specified method
        2. BGE rerank to top 6-10 results  
        3. Apply MMR (λ≈0.2) for diversity
        
        Args:
            query: Search query text
            query_embedding: Query embedding vector
            search_method: "hybrid", "dense", "dual_vector" 
            top_k_initial: Initial search results (30)
            top_k_rerank: Results after reranking (6-10)
            top_k_final: Final MMR results (≤ top_k_rerank)
            lambda_param: MMR lambda parameter (≈0.2)
            **search_kwargs: Additional search parameters
            
        Returns:
            Final diverse, reranked results
        """
        # Step 1: Initial search with large top_k
        if search_method == "hybrid":
            query_text = search_kwargs.pop('query_text', query)  # Remove from kwargs to avoid conflict
            initial_results = self.hybrid_search(
                query_embedding, query_text, top_k=top_k_initial, **search_kwargs
            )
        elif search_method == "dual_vector":
            initial_results = self.dual_vector_search(
                query_embedding, top_k=top_k_initial, **search_kwargs
            )
        else:  # dense search
            initial_results = self.search(
                query_embedding, top_k=top_k_initial, **search_kwargs
            )
        
        if not initial_results:
            return []
        
        print(f"📊 Initial search ({search_method}): {len(initial_results)} results")
        
        # Step 2: BGE Reranker (30 → 6-10)
        if rerank_documents:
            try:
                reranked_results = rerank_documents(query, initial_results, top_k_rerank)
                print(f"🔄 BGE Reranked: {len(reranked_results)} results")
            except Exception as e:
                print(f"⚠️  Reranking failed, using original: {e}")
                reranked_results = initial_results[:top_k_rerank]
        else:
            print("⚠️  BGE reranker not available, using top results")
            reranked_results = initial_results[:top_k_rerank]
        
        # Step 3: MMR for diversity (6-10 → final)
        final_results = self.get_diverse_results(reranked_results, lambda_param)
        final_results = final_results[:top_k_final]
        
        print(f"🎯 Final MMR (λ={lambda_param}): {len(final_results)} diverse results")
        
        # Add pipeline metadata
        for i, result in enumerate(final_results):
            result['pipeline_rank'] = i + 1
            result['pipeline_method'] = f"{search_method}_rerank_mmr"
        
        return final_results
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on common words"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def save_index(self):
        """Save the search index to disk"""
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        
        # Save document metadata
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        # Save embeddings (if available for backward compatibility)
        if hasattr(self, 'embeddings') and self.embeddings:
            np.save(self.embeddings_file, np.array(self.embeddings))
    
    def load_index(self):
        """Load the search index from disk"""
        try:
            # Load document metadata
            if os.path.exists(self.index_file):
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            
            # Load embeddings
            if os.path.exists(self.embeddings_file):
                embeddings_array = np.load(self.embeddings_file)
                self.embeddings = [emb for emb in embeddings_array]
            
            print(f"Loaded {len(self.documents)} documents from search index")
            
        except Exception as e:
            print(f"Error loading search index: {e}")
            self.documents = []
            self.embeddings = []
    
    def clear(self):
        """Clear all data from the enhanced search index"""
        # Clear Qdrant collection
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._setup_qdrant_collection()  # Recreate empty collection
        except Exception as e:
            print(f"Warning: Qdrant clear failed: {e}")
        
        # Clear local data
        self.documents = []
        self.bm25_corpus = []
        self.bm25_index = None
        
        # Remove files
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        if os.path.exists(self.embeddings_file):
            os.remove(self.embeddings_file)
        
        print("Enhanced search index cleared (Qdrant + BM25)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the enhanced search index"""
        if not self.documents:
            return {"total_documents": 0}
        
        doc_types = {}
        languages = {}
        sources = {}
        
        for doc in self.documents:
            # Count by document type
            doc_type = doc.get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            # Count by language
            language = doc.get('language', 'unknown')
            languages[language] = languages.get(language, 0) + 1
            
            # Count by source
            source = doc.get('source_id', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        # Get Qdrant stats
        qdrant_count = 0
        try:
            collection_info = self.client.get_collection(self.collection_name)
            qdrant_count = collection_info.points_count
        except:
            qdrant_count = 0
        
        return {
            "total_documents": len(self.documents),
            "qdrant_points": qdrant_count,
            "bm25_index_active": self.bm25_index is not None,
            "bm25_corpus_size": len(self.bm25_corpus),
            "doc_types": doc_types,
            "languages": languages,
            "sources": sources,
            "search_capabilities": ["vector_similarity", "bm25_sparse", "hybrid_search", "dual_vector"]
        }
