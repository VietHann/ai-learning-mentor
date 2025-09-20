#!/usr/bin/env python3
"""
RAGAS-style Evaluation Framework for AI Virtual Mentor
Implements RAG evaluation metrics without external dependencies
"""

import json
import math
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Compatibility imports  
try:
    import numpy as np
except ImportError:
    np = None
    
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class EvaluationResult:
    """Results container for RAG evaluation"""
    question: str
    generated_answer: str
    expected_answer: str
    retrieved_contexts: List[str]
    metrics: Dict[str, float]
    timestamp: str
    metadata: Dict[str, Any]


class RAGASEvaluator:
    """RAGAS-style evaluation metrics for RAG systems"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2)
            )
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts"""
        if not SKLEARN_AVAILABLE or not text1.strip() or not text2.strip():
            # Fallback: simple token overlap
            tokens1 = set(text1.lower().split())
            tokens2 = set(text2.lower().split())
            if not tokens1 or not tokens2:
                return 0.0
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            return intersection / union if union > 0 else 0.0
        
        try:
            # Use TF-IDF similarity
            texts = [text1, text2]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception:
            # Fallback to token overlap
            tokens1 = set(text1.lower().split())
            tokens2 = set(text2.lower().split())
            if not tokens1 or not tokens2:
                return 0.0
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            return intersection / union if union > 0 else 0.0
    
    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Faithfulness: Measures how well the answer is grounded in the retrieved context
        Score: 0.0 (not faithful) to 1.0 (fully faithful)
        """
        if not contexts or not answer.strip():
            return 0.0
        
        # Combine all contexts
        combined_context = " ".join(contexts)
        
        # Split answer into statements (simple sentence splitting)
        statements = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
        if not statements:
            return 0.0
        
        faithful_count = 0
        for statement in statements:
            # Check if statement is supported by context
            similarity = self.compute_semantic_similarity(statement, combined_context)
            if similarity > 0.3:  # Threshold for "supported by context"
                faithful_count += 1
        
        return faithful_count / len(statements)
    
    def evaluate_answer_relevance(self, question: str, answer: str) -> float:
        """
        Answer Relevance: Measures how well the answer addresses the question
        Score: 0.0 (irrelevant) to 1.0 (highly relevant)
        """
        if not question.strip() or not answer.strip():
            return 0.0
        
        return self.compute_semantic_similarity(question, answer)
    
    def evaluate_context_precision(self, question: str, contexts: List[str], 
                                 relevant_threshold: float = 0.3) -> float:
        """
        Context Precision: Measures relevance of retrieved contexts to the question
        Score: 0.0 (no relevant context) to 1.0 (all contexts relevant)
        """
        if not contexts or not question.strip():
            return 0.0
        
        relevant_count = 0
        for context in contexts:
            if context.strip():
                similarity = self.compute_semantic_similarity(question, context)
                if similarity > relevant_threshold:
                    relevant_count += 1
        
        return relevant_count / len(contexts)
    
    def evaluate_context_recall(self, expected_answer: str, contexts: List[str],
                              coverage_threshold: float = 0.4) -> float:
        """
        Context Recall: Measures how well contexts cover the expected answer
        Score: 0.0 (no coverage) to 1.0 (full coverage)
        """
        if not contexts or not expected_answer.strip():
            return 0.0
        
        combined_context = " ".join(contexts)
        similarity = self.compute_semantic_similarity(expected_answer, combined_context)
        
        # Convert similarity to recall score
        return min(similarity / coverage_threshold, 1.0)
    
    def evaluate_answer_correctness(self, expected_answer: str, generated_answer: str) -> float:
        """
        Answer Correctness: Semantic similarity between expected and generated answers
        Score: 0.0 (completely wrong) to 1.0 (perfect match)
        """
        if not expected_answer.strip() or not generated_answer.strip():
            return 0.0
        
        return self.compute_semantic_similarity(expected_answer, generated_answer)
    
    def evaluate_citation_quality(self, answer: str, citations: List[str]) -> float:
        """
        Citation Quality: Measures presence and quality of citations
        Score: 0.0 (no citations) to 1.0 (excellent citations)
        """
        if not citations:
            return 0.0
        
        # Check for citation markers in answer
        citation_patterns = [
            r'\[.*?\]',  # [Source 1]
            r'\(.*?\)',  # (Page 5)
            r'trang \d+', # trang 5
            r'page \d+',  # page 5
            r'section \d+', # section 2
            r'mục \d+'    # mục 3
        ]
        
        citation_score = 0.0
        for pattern in citation_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                citation_score += 0.2
        
        # Quality bonus for detailed citations
        detailed_citations = 0
        for citation in citations:
            if any(keyword in citation.lower() for keyword in ['page', 'trang', 'section', 'mục']):
                detailed_citations += 1
        
        if detailed_citations > 0:
            citation_score += 0.2 * (detailed_citations / len(citations))
        
        return min(citation_score, 1.0)
    
    def evaluate_integrity_compliance(self, answer: str, integrity_mode: str) -> float:
        """
        Integrity Compliance: Checks if answer respects academic integrity mode
        Score: 0.0 (violates mode) to 1.0 (fully compliant)
        """
        if integrity_mode == "normal":
            return 1.0  # No restrictions
        
        # Code detection patterns
        code_indicators = [
            r'def \w+\(',  # Python functions
            r'public class',  # Java classes
            r'for\s*\(',  # For loops
            r'while\s*\(',  # While loops
            r'if\s*\(',   # If statements
            r'import\s+\w+',  # Import statements
            r'#include',  # C++ includes
            r'{\s*\w+.*}',  # Code blocks
        ]
        
        code_violations = 0
        for pattern in code_indicators:
            if re.search(pattern, answer, re.IGNORECASE):
                code_violations += 1
        
        if integrity_mode == "exam":
            # Strictest mode - no code at all
            if code_violations > 0:
                return 0.0
            # Also check for algorithmic details
            algorithm_patterns = [
                r'algorithm:',
                r'step \d+:',
                r'thuật toán:',
                r'bước \d+:',
                r'implementation',
                r'cài đặt'
            ]
            for pattern in algorithm_patterns:
                if re.search(pattern, answer, re.IGNORECASE):
                    return 0.5  # Partial violation
            return 1.0
        
        elif integrity_mode == "academic":
            # Moderate restrictions - pseudocode OK, full code not OK
            if code_violations > 2:  # Allow some pseudocode
                return 0.5
            return 1.0
        
        return 1.0
    
    def evaluate_response_completeness(self, question: str, answer: str, 
                                     expected_length_ratio: float = 0.5) -> float:
        """
        Response Completeness: Measures if answer is sufficiently detailed
        Score: 0.0 (too brief) to 1.0 (appropriately detailed)
        """
        if not question.strip() or not answer.strip():
            return 0.0
        
        # Basic length check
        answer_words = len(answer.split())
        question_words = len(question.split())
        
        if answer_words < question_words * expected_length_ratio:
            return 0.3  # Too brief
        
        # Check for key information indicators
        info_indicators = [
            'because', 'since', 'due to',  # Explanations
            'for example', 'such as', 'like',  # Examples
            'however', 'but', 'although',  # Contrasts
            'vì', 'do', 'tại vì',  # Vietnamese explanations
            'ví dụ', 'chẳng hạn như',  # Vietnamese examples
            'tuy nhiên', 'nhưng'  # Vietnamese contrasts
        ]
        
        info_score = 0
        for indicator in info_indicators:
            if indicator in answer.lower():
                info_score += 0.1
        
        completeness_score = min(0.7 + info_score, 1.0)
        return completeness_score
    
    def evaluate_language_consistency(self, question: str, answer: str) -> float:
        """
        Language Consistency: Checks if answer language matches question language
        Score: 0.0 (inconsistent) to 1.0 (consistent)
        """
        def detect_language(text: str) -> str:
            # Simple language detection
            vietnamese_words = ['là', 'của', 'trong', 'với', 'được', 'có', 'không', 'này', 'đó']
            english_words = ['the', 'is', 'of', 'in', 'to', 'and', 'a', 'that', 'it', 'with']
            
            text_lower = text.lower()
            vietnamese_count = sum(1 for word in vietnamese_words if word in text_lower)
            english_count = sum(1 for word in english_words if word in text_lower)
            
            if vietnamese_count > english_count:
                return 'vietnamese'
            elif english_count > vietnamese_count:
                return 'english'
            else:
                return 'mixed'
        
        question_lang = detect_language(question)
        answer_lang = detect_language(answer)
        
        if question_lang == answer_lang:
            return 1.0
        elif question_lang == 'mixed' or answer_lang == 'mixed':
            return 0.8  # Mixed is acceptable
        else:
            return 0.3  # Language mismatch
    
    def comprehensive_evaluate(self, question: str, generated_answer: str, 
                             expected_answer: str, retrieved_contexts: List[str],
                             citations: List[str], integrity_mode: str = "normal",
                             metadata: Dict[str, Any] = None) -> EvaluationResult:
        """
        Comprehensive evaluation using all RAGAS metrics
        """
        start_time = time.time()
        
        metrics = {
            'faithfulness': self.evaluate_faithfulness(generated_answer, retrieved_contexts),
            'answer_relevance': self.evaluate_answer_relevance(question, generated_answer),
            'context_precision': self.evaluate_context_precision(question, retrieved_contexts),
            'context_recall': self.evaluate_context_recall(expected_answer, retrieved_contexts),
            'answer_correctness': self.evaluate_answer_correctness(expected_answer, generated_answer),
            'citation_quality': self.evaluate_citation_quality(generated_answer, citations),
            'integrity_compliance': self.evaluate_integrity_compliance(generated_answer, integrity_mode),
            'response_completeness': self.evaluate_response_completeness(question, generated_answer),
            'language_consistency': self.evaluate_language_consistency(question, generated_answer)
        }
        
        # Compute overall score
        weights = {
            'faithfulness': 0.15,
            'answer_relevance': 0.20,
            'context_precision': 0.10,
            'context_recall': 0.10,
            'answer_correctness': 0.20,
            'citation_quality': 0.10,
            'integrity_compliance': 0.10,
            'response_completeness': 0.05,
            'language_consistency': 0.05
        }
        
        overall_score = sum(metrics[key] * weights[key] for key in metrics.keys())
        metrics['overall_score'] = overall_score
        metrics['evaluation_time'] = time.time() - start_time
        
        return EvaluationResult(
            question=question,
            generated_answer=generated_answer,
            expected_answer=expected_answer,
            retrieved_contexts=retrieved_contexts,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )


def run_golden_qa_evaluation(golden_qa_path: str, mentor_system=None) -> Dict[str, Any]:
    """
    Run evaluation against Golden Q&A dataset
    """
    evaluator = RAGASEvaluator()
    
    try:
        with open(golden_qa_path, 'r', encoding='utf-8') as f:
            golden_data = json.load(f)
    except FileNotFoundError:
        return {"error": f"Golden Q&A file not found: {golden_qa_path}"}
    
    results = []
    category_scores = {}
    
    for category_name, category_data in golden_data['categories'].items():
        category_results = []
        
        for qa_item in category_data['questions']:
            question = qa_item['question']
            expected_answer = qa_item['expected_answer']
            
            # Mock evaluation if no mentor system provided
            if mentor_system is None:
                # Generate mock response for testing
                generated_answer = f"Mock response to: {question[:50]}..."
                retrieved_contexts = ["Mock context 1", "Mock context 2"]
                citations = ["Mock citation 1"]
            else:
                # Use actual mentor system
                generated_answer, citations = mentor_system.generate_response(
                    question, 
                    academic_mode=True,
                    language="auto"
                )
                retrieved_contexts = mentor_system.get_last_retrieved_contexts()
            
            # Evaluate
            result = evaluator.comprehensive_evaluate(
                question=question,
                generated_answer=generated_answer,
                expected_answer=expected_answer,
                retrieved_contexts=retrieved_contexts,
                citations=citations,
                integrity_mode="academic",
                metadata={
                    'category': category_name,
                    'difficulty': qa_item['difficulty'],
                    'topics': qa_item['topics'],
                    'qa_id': qa_item['id']
                }
            )
            
            category_results.append(result)
            results.append(result)
        
        # Calculate category average
        if category_results:
            avg_scores = {}
            for metric in category_results[0].metrics.keys():
                avg_scores[metric] = sum(r.metrics[metric] for r in category_results) / len(category_results)
            category_scores[category_name] = avg_scores
    
    # Overall statistics
    if results:
        overall_metrics = {}
        for metric in results[0].metrics.keys():
            overall_metrics[metric] = sum(r.metrics[metric] for r in results) / len(results)
    else:
        overall_metrics = {}
    
    return {
        'evaluation_summary': {
            'total_questions': len(results),
            'overall_metrics': overall_metrics,
            'category_scores': category_scores,
            'timestamp': datetime.now().isoformat()
        },
        'detailed_results': [
            {
                'qa_id': r.metadata.get('qa_id', 'unknown'),
                'question': r.question,
                'category': r.metadata.get('category', 'unknown'),
                'difficulty': r.metadata.get('difficulty', 'unknown'),
                'metrics': r.metrics,
                'generated_answer_preview': r.generated_answer[:200] + "..." if len(r.generated_answer) > 200 else r.generated_answer
            }
            for r in results
        ]
    }


if __name__ == "__main__":
    print("🧪 RAGAS Evaluation Framework Test")
    
    # Test basic functionality
    evaluator = RAGASEvaluator()
    
    # Sample evaluation
    test_result = evaluator.comprehensive_evaluate(
        question="What is machine learning?",
        generated_answer="Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
        expected_answer="Machine learning is a field of AI that allows systems to learn from data.",
        retrieved_contexts=["Machine learning is part of AI", "ML uses data to make predictions"],
        citations=["Source: ML Textbook, Page 15"],
        integrity_mode="academic"
    )
    
    print(f"✅ Test evaluation completed")
    print(f"Overall Score: {test_result.metrics['overall_score']:.3f}")
    print(f"Key Metrics:")
    print(f"  - Answer Relevance: {test_result.metrics['answer_relevance']:.3f}")
    print(f"  - Faithfulness: {test_result.metrics['faithfulness']:.3f}")
    print(f"  - Citation Quality: {test_result.metrics['citation_quality']:.3f}")
    
    # Test Golden Q&A evaluation
    print("\n🧪 Testing Golden Q&A Evaluation")
    qa_results = run_golden_qa_evaluation('evaluation/golden_qa.json')
    
    if 'error' not in qa_results:
        print(f"✅ Golden Q&A evaluation completed")
        print(f"Total Questions: {qa_results['evaluation_summary']['total_questions']}")
        print(f"Overall Score: {qa_results['evaluation_summary']['overall_metrics'].get('overall_score', 0):.3f}")
    else:
        print(f"⚠️ Golden Q&A evaluation: {qa_results['error']}")
    
    print("🎯 RAGAS Evaluation Framework Ready!")