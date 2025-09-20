import numpy as np
from typing import List, Dict, Any
import os

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class EmbeddingGenerator:
    """Generate multilingual embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        self.use_simple_embeddings = not SentenceTransformer
        
        if not SentenceTransformer:
            print("sentence-transformers not available, using simple embeddings fallback")
        
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            if SentenceTransformer:
                self.model = SentenceTransformer(self.model_name)
                print(f"Loaded embedding model: {self.model_name}")
            else:
                print("sentence-transformers not available, using fallback")
                self.model = None
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            # Fallback to a smaller model if the main one fails
            try:
                if SentenceTransformer:
                    self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                    print("Loaded fallback embedding model")
                else:
                    self.model = None
            except Exception as e2:
                print(f"Failed to load any embedding model, using fallback: {e2}")
                self.model = None
    
    def generate_embeddings(self, documents: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Generate embeddings for a list of documents"""
        if not self.model:
            # Fallback to simple text-based embeddings using basic word counts
            return self._generate_simple_embeddings(documents)
        
        texts = []
        for doc in documents:
            # Combine title and content for better embeddings
            title = doc.get('title', '')
            section = doc.get('section', '')
            content = doc.get('content', '')
            
            # Create a combined text with context
            combined_text = f"{title}"
            if section:
                combined_text += f" - {section}"
            combined_text += f": {content}"
            
            texts.append(combined_text)
        
        # Generate embeddings in batches for efficiency
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a search query"""
        if not self.model:
            return self._generate_simple_query_embedding(query)
        
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embedding
    
    def generate_content_summary_embeddings(self, documents: List[Dict[str, Any]]) -> tuple:
        """Generate both content and summary embeddings for hybrid search"""
        content_embeddings = []
        summary_embeddings = []
        
        for doc in documents:
            if self.model:
                # Content embedding (full text)
                content = doc.get('content', '')
                content_emb = self.model.encode(content, normalize_embeddings=True)
                content_embeddings.append(content_emb)
                
                # Summary embedding (title + section + keywords)
                title = doc.get('title', '')
                section = doc.get('section', '')
                keywords = ' '.join(doc.get('keywords', []))
                summary_text = f"{title} {section} {keywords}".strip()
                
                summary_emb = self.model.encode(summary_text, normalize_embeddings=True)
                summary_embeddings.append(summary_emb)
            else:
                # Fallback embeddings
                content_embeddings.append(self._generate_simple_query_embedding(doc.get('content', '')))
                summary_text = f"{doc.get('title', '')} {doc.get('section', '')} {' '.join(doc.get('keywords', []))}".strip()
                summary_embeddings.append(self._generate_simple_query_embedding(summary_text))
        
        return content_embeddings, summary_embeddings
    
    def compute_similarity(self, query_embedding: np.ndarray, doc_embeddings: List[np.ndarray]) -> List[float]:
        """Compute cosine similarity between query and document embeddings"""
        similarities = []
        
        for doc_embedding in doc_embeddings:
            # Cosine similarity (since embeddings are normalized)
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append(float(similarity))
        
        return similarities
    
    def save_embeddings(self, embeddings: List[np.ndarray], filepath: str):
        """Save embeddings to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, np.array(embeddings))
    
    def load_embeddings(self, filepath: str) -> List[np.ndarray]:
        """Load embeddings from disk"""
        if os.path.exists(filepath):
            embeddings_array = np.load(filepath)
            return [embedding for embedding in embeddings_array]
        return []
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model"""
        if not self.model:
            return 384  # Default dimension for MiniLM models
        
        # Test with a sample text to get dimension
        test_embedding = self.model.encode("test", convert_to_numpy=True)
        return test_embedding.shape[0]
    
    def _generate_simple_embeddings(self, documents: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Fallback simple embedding generation using TF-IDF approach"""
        from collections import Counter
        import math
        
        # Build vocabulary from all documents
        all_words = set()
        doc_texts = []
        
        for doc in documents:
            content = doc.get('content', '')
            # Combine title and content for better context
            title = doc.get('title', '')
            section = doc.get('section', '')
            combined_text = f"{title} {section} {content}".lower()
            
            words = combined_text.split()
            doc_texts.append(words)
            all_words.update(words)
        
        # Create vocabulary mapping
        vocab = sorted(all_words)
        vocab_size = min(len(vocab), 384)  # Limit vocab size for efficiency
        word_to_idx = {word: i for i, word in enumerate(vocab[:vocab_size])}
        
        # Generate TF-IDF embeddings
        embeddings = []
        total_docs = len(doc_texts)
        
        for words in doc_texts:
            # Count word frequencies
            word_counts = Counter(words)
            embedding = np.zeros(vocab_size)
            
            for word, count in word_counts.items():
                if word in word_to_idx:
                    idx = word_to_idx[word]
                    
                    # TF (Term Frequency)
                    tf = count / len(words) if len(words) > 0 else 0
                    
                    # IDF (Inverse Document Frequency) - simplified
                    doc_freq = sum(1 for doc_words in doc_texts if word in doc_words)
                    idf = math.log(total_docs / max(1, doc_freq))
                    
                    # TF-IDF score
                    embedding[idx] = tf * idf
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            embeddings.append(embedding)
        
        # Save vocabulary for consistent query embeddings
        self._simple_vocab = word_to_idx
        self._doc_texts = doc_texts  # For IDF calculation
        
        return embeddings
    
    def _generate_simple_query_embedding(self, query: str) -> np.ndarray:
        """Fallback simple query embedding using same TF-IDF approach"""
        import math
        from collections import Counter
        
        if not hasattr(self, '_simple_vocab') or not hasattr(self, '_doc_texts'):
            # Fallback to basic approach if vocab not built
            words = query.lower().split()
            embedding = np.zeros(384)
            for i, word in enumerate(words[:384]):
                # Use a more semantic approach - character-based features
                word_hash = abs(hash(word))
                for j in range(min(len(word), 20)):  # Use character features
                    embedding[(word_hash + j) % 384] += 1.0 / len(word)
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        
        # Use TF-IDF approach consistent with documents
        words = query.lower().split()
        word_counts = Counter(words)
        embedding = np.zeros(len(self._simple_vocab))
        
        total_docs = len(self._doc_texts)
        
        for word, count in word_counts.items():
            if word in self._simple_vocab:
                idx = self._simple_vocab[word]
                
                # TF for query
                tf = count / len(words) if len(words) > 0 else 0
                
                # IDF from document collection
                doc_freq = sum(1 for doc_words in self._doc_texts if word in doc_words)
                idf = math.log(total_docs / max(1, doc_freq))
                
                embedding[idx] = tf * idf
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
