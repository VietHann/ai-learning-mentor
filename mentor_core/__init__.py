"""
AI Virtual Mentor Core Package

This package contains the core logic for the AI Virtual Mentor system,
including document processing, embeddings, vector search, and AI integration.
"""

from .document_processor import DocumentProcessor
from .embeddings import EmbeddingGenerator
from .vector_search import VectorSearch
from .gemini_client import GeminiClient
from .question_classifier import QuestionClassifier
from .citation_formatter import CitationFormatter
from .reranker import BGEReranker

__version__ = "1.0.0"
__author__ = "AI Virtual Mentor Team"

__all__ = [
    "DocumentProcessor",
    "EmbeddingGenerator", 
    "VectorSearch",
    "GeminiClient",
    "QuestionClassifier",
    "CitationFormatter",
    "BGEReranker"
]