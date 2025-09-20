
import json
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class TestVariant(Enum):
    """Test variant identifiers"""
    CONTROL = "control"
    VARIANT_A = "variant_a"
    VARIANT_B = "variant_b"
    VARIANT_C = "variant_c"


@dataclass
class ABTestConfig:
    """Configuration for A/B test"""
    test_name: str
    description: str
    variants: Dict[str, Dict[str, Any]]
    success_metrics: List[str]
    sample_size_per_variant: int
    confidence_level: float = 0.95
    minimum_effect_size: float = 0.1
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class TestResult:
    """Result of individual test case"""
    test_id: str
    variant: str
    question: str
    response: str
    metrics: Dict[str, float]
    execution_time: float
    timestamp: str
    metadata: Dict[str, Any]


@dataclass
class ABTestResults:
    """Complete A/B test results with statistical analysis"""
    test_config: ABTestConfig
    results_by_variant: Dict[str, List[TestResult]]
    statistical_analysis: Dict[str, Any]
    recommendations: List[str]
    completion_date: str


class ABTester:
    """A/B Testing framework for mentor system evaluation"""
    
    def __init__(self):
        self.current_tests: Dict[str, ABTestConfig] = {}
        self.completed_tests: Dict[str, ABTestResults] = {}
    
    def create_test_configuration(self, test_name: str, description: str,
                                control_config: Dict[str, Any],
                                variant_configs: Dict[str, Dict[str, Any]],
                                success_metrics: List[str],
                                sample_size: int = 50) -> ABTestConfig:
        """Create A/B test configuration"""
        
        variants = {"control": control_config}
        variants.update(variant_configs)
        
        config = ABTestConfig(
            test_name=test_name,
            description=description,
            variants=variants,
            success_metrics=success_metrics,
            sample_size_per_variant=sample_size
        )
        
        self.current_tests[test_name] = config
        return config
    
    def run_test_scenario(self, test_config: ABTestConfig, 
                         test_questions: List[Dict[str, Any]],
                         mentor_system_factory: Callable[[Dict[str, Any]], Any]) -> ABTestResults:
        """
        Run complete A/B test scenario
        """
        print(f"🧪 Running A/B Test: {test_config.test_name}")
        print(f"📝 Description: {test_config.description}")
        print(f"🎯 Variants: {list(test_config.variants.keys())}")
        print(f"📊 Sample size per variant: {test_config.sample_size_per_variant}")
        
        results_by_variant = {}
        
        # Run tests for each variant
        for variant_name, variant_config in test_config.variants.items():
            print(f"\n🔄 Testing variant: {variant_name}")
            
            # Initialize mentor system with variant configuration
            mentor_system = mentor_system_factory(variant_config)
            
            variant_results = []
            questions_to_test = random.sample(
                test_questions, 
                min(test_config.sample_size_per_variant, len(test_questions))
            )
            
            for i, question_data in enumerate(questions_to_test):
                print(f"  Question {i+1}/{len(questions_to_test)}: {question_data['question'][:50]}...")
                
                start_time = time.time()
                
                try:
                    # Generate response using mentor system
                    if mentor_system is None:
                        # Mock response for testing
                        response = f"Mock {variant_name} response to: {question_data['question'][:30]}..."
                        metrics = self._generate_mock_metrics(variant_name)
                    else:
                        response, citations = mentor_system.generate_response(
                            question=question_data['question'],
                            language=question_data.get('language', 'Vietnamese'),
                            academic_mode=variant_config.get('academic_mode', True)
                        )
                        
                        # Evaluate response using RAGAS metrics
                        from evaluation.ragas_evaluator import RAGASEvaluator
                        evaluator = RAGASEvaluator()
                        
                        eval_result = evaluator.comprehensive_evaluate(
                            question=question_data['question'],
                            generated_answer=response,
                            expected_answer=question_data.get('expected_answer', ''),
                            retrieved_contexts=getattr(mentor_system, 'last_contexts', []),
                            citations=citations,
                            integrity_mode=variant_config.get('integrity_mode', 'academic')
                        )
                        metrics = eval_result.metrics
                
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    response = f"Error occurred: {str(e)}"
                    metrics = {metric: 0.0 for metric in test_config.success_metrics}
                
                execution_time = time.time() - start_time
                
                result = TestResult(
                    test_id=f"{test_config.test_name}_{variant_name}_{i}",
                    variant=variant_name,
                    question=question_data['question'],
                    response=response,
                    metrics=metrics,
                    execution_time=execution_time,
                    timestamp=datetime.now().isoformat(),
                    metadata={
                        'question_category': question_data.get('category', 'unknown'),
                        'difficulty': question_data.get('difficulty', 'unknown'),
                        'variant_config': variant_config
                    }
                )
                
                variant_results.append(result)
            
            results_by_variant[variant_name] = variant_results
            print(f"  ✅ Completed {len(variant_results)} test cases for {variant_name}")
        
        # Perform statistical analysis
        statistical_analysis = self._analyze_results(results_by_variant, test_config.success_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(statistical_analysis, test_config)
        
        ab_results = ABTestResults(
            test_config=test_config,
            results_by_variant=results_by_variant,
            statistical_analysis=statistical_analysis,
            recommendations=recommendations,
            completion_date=datetime.now().isoformat()
        )
        
        self.completed_tests[test_config.test_name] = ab_results
        return ab_results
    
    def _generate_mock_metrics(self, variant_name: str) -> Dict[str, float]:
        """Generate mock metrics for testing purposes"""
        base_scores = {
            'overall_score': 0.75,
            'answer_relevance': 0.80,
            'faithfulness': 0.70,
            'citation_quality': 0.65,
            'integrity_compliance': 0.90,
            'language_consistency': 0.85
        }
        
        # Add some variance based on variant
        variant_multipliers = {
            'control': 1.0,
            'variant_a': 1.1,
            'variant_b': 0.95,
            'variant_c': 1.05
        }
        
        multiplier = variant_multipliers.get(variant_name, 1.0)
        noise = random.uniform(0.9, 1.1)  # Add random noise
        
        return {
            metric: min(score * multiplier * noise, 1.0)
            for metric, score in base_scores.items()
        }
    
    def _analyze_results(self, results_by_variant: Dict[str, List[TestResult]],
                        success_metrics: List[str]) -> Dict[str, Any]:
        """Perform statistical analysis on A/B test results"""
        
        analysis = {
            'sample_sizes': {},
            'metric_comparisons': {},
            'statistical_significance': {},
            'effect_sizes': {},
            'confidence_intervals': {}
        }
        
        # Calculate sample sizes and basic statistics
        for variant, results in results_by_variant.items():
            analysis['sample_sizes'][variant] = len(results)
        
        # Analyze each success metric
        for metric in success_metrics:
            metric_analysis = {}
            metric_data = {}
            
            # Extract metric values for each variant
            for variant, results in results_by_variant.items():
                values = [r.metrics.get(metric, 0.0) for r in results]
                metric_data[variant] = values
                
                metric_analysis[variant] = {
                    'mean': sum(values) / len(values) if values else 0,
                    'count': len(values),
                    'min': min(values) if values else 0,
                    'max': max(values) if values else 0,
                    'std': self._calculate_std(values) if len(values) > 1 else 0
                }
            
            analysis['metric_comparisons'][metric] = metric_analysis
            
            # Perform pairwise statistical tests
            if len(metric_data) >= 2:
                variants = list(metric_data.keys())
                control_variant = 'control' if 'control' in variants else variants[0]
                
                significance_tests = {}
                effect_sizes = {}
                
                for variant in variants:
                    if variant != control_variant:
                        p_value, effect_size = self._perform_statistical_test(
                            metric_data[control_variant],
                            metric_data[variant]
                        )
                        
                        significance_tests[f"{control_variant}_vs_{variant}"] = {
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'confidence_level': 0.95
                        }
                        
                        effect_sizes[f"{control_variant}_vs_{variant}"] = effect_size
                
                analysis['statistical_significance'][metric] = significance_tests
                analysis['effect_sizes'][metric] = effect_sizes
        
        return analysis
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _perform_statistical_test(self, control_values: List[float], 
                                variant_values: List[float]) -> Tuple[float, float]:
        """Perform statistical test between control and variant"""
        
        if SCIPY_AVAILABLE and len(control_values) > 5 and len(variant_values) > 5:
            # Use scipy for proper statistical tests
            try:
                statistic, p_value = stats.ttest_ind(control_values, variant_values)
                
                # Calculate Cohen's d (effect size)
                control_mean = sum(control_values) / len(control_values)
                variant_mean = sum(variant_values) / len(variant_values)
                control_std = self._calculate_std(control_values)
                variant_std = self._calculate_std(variant_values)
                
                pooled_std = ((len(control_values) - 1) * control_std**2 + 
                             (len(variant_values) - 1) * variant_std**2) / \
                             (len(control_values) + len(variant_values) - 2)
                pooled_std = pooled_std ** 0.5
                
                effect_size = (variant_mean - control_mean) / pooled_std if pooled_std > 0 else 0
                
                return float(p_value), float(effect_size)
                
            except Exception:
                pass  # Fall back to manual calculation
        
        # Manual statistical test (simplified)
        control_mean = sum(control_values) / len(control_values) if control_values else 0
        variant_mean = sum(variant_values) / len(variant_values) if variant_values else 0
        
        # Simple difference test (not proper statistical test)
        difference = abs(variant_mean - control_mean)
        
        # Mock p-value based on difference magnitude
        if difference > 0.1:
            p_value = 0.01  # "Significant"
        elif difference > 0.05:
            p_value = 0.08  # "Marginally significant"
        else:
            p_value = 0.5   # "Not significant"
        
        # Simple effect size calculation
        control_std = self._calculate_std(control_values)
        effect_size = difference / control_std if control_std > 0 else 0
        
        return p_value, effect_size
    
    def _generate_recommendations(self, statistical_analysis: Dict[str, Any],
                                test_config: ABTestConfig) -> List[str]:
        """Generate actionable recommendations based on test results"""
        
        recommendations = []
        
        # Find best performing variant for each metric
        for metric, comparisons in statistical_analysis['metric_comparisons'].items():
            best_variant = max(comparisons.keys(), key=lambda v: comparisons[v]['mean'])
            best_score = comparisons[best_variant]['mean']
            
            recommendations.append(
                f"🎯 For {metric}: '{best_variant}' performs best with score {best_score:.3f}"
            )
            
            # Check for statistical significance
            if metric in statistical_analysis['statistical_significance']:
                sig_tests = statistical_analysis['statistical_significance'][metric]
                for comparison, result in sig_tests.items():
                    if result['significant']:
                        effect = statistical_analysis['effect_sizes'][metric][comparison]
                        recommendations.append(
                            f"📊 {comparison} shows significant difference (p={result['p_value']:.4f}, "
                            f"effect size={effect:.3f})"
                        )
        
        # Overall recommendations
        overall_best = self._find_overall_best_variant(statistical_analysis, test_config.success_metrics)
        if overall_best:
            recommendations.append(f"🏆 Overall recommendation: Use '{overall_best}' configuration")
        
        # Sample size recommendations
        min_sample_size = min(statistical_analysis['sample_sizes'].values())
        if min_sample_size < 30:
            recommendations.append(
                f"⚠️ Consider increasing sample size (current min: {min_sample_size}, recommended: 30+)"
            )
        
        return recommendations
    
    def _find_overall_best_variant(self, statistical_analysis: Dict[str, Any],
                                 success_metrics: List[str]) -> Optional[str]:
        """Find overall best performing variant across all metrics"""
        
        variant_scores = {}
        
        for metric in success_metrics:
            if metric in statistical_analysis['metric_comparisons']:
                comparisons = statistical_analysis['metric_comparisons'][metric]
                
                for variant, stats in comparisons.items():
                    if variant not in variant_scores:
                        variant_scores[variant] = []
                    variant_scores[variant].append(stats['mean'])
        
        # Calculate average score for each variant
        variant_averages = {
            variant: sum(scores) / len(scores)
            for variant, scores in variant_scores.items()
        }
        
        if variant_averages:
            return max(variant_averages.keys(), key=lambda v: variant_averages[v])
        
        return None
    
    def export_results(self, test_name: str, output_path: str) -> bool:
        """Export A/B test results to JSON file"""
        
        if test_name not in self.completed_tests:
            return False
        
        results = self.completed_tests[test_name]
        
        # Convert to serializable format
        export_data = {
            'test_config': asdict(results.test_config),
            'statistical_analysis': results.statistical_analysis,
            'recommendations': results.recommendations,
            'completion_date': results.completion_date,
            'summary_metrics': self._generate_summary_metrics(results)
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Export error: {e}")
            return False
    
    def _generate_summary_metrics(self, results: ABTestResults) -> Dict[str, Any]:
        """Generate summary metrics for export"""
        
        summary = {
            'total_test_cases': sum(len(tests) for tests in results.results_by_variant.values()),
            'variants_tested': len(results.results_by_variant),
            'success_metrics_evaluated': len(results.test_config.success_metrics),
            'average_execution_time': {}
        }
        
        # Calculate average execution times
        for variant, tests in results.results_by_variant.items():
            if tests:
                avg_time = sum(t.execution_time for t in tests) / len(tests)
                summary['average_execution_time'][variant] = avg_time
        
        return summary


# Pre-defined test configurations
def create_standard_ab_tests():
    """Create standard A/B test configurations for mentor system"""
    
    tester = ABTester()
    
    # Test 1: Embedding Models Comparison
    embedding_test = tester.create_test_configuration(
        test_name="embedding_models_comparison",
        description="Compare different embedding approaches for document retrieval",
        control_config={
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "embedding_fallback": True,
            "search_threshold": 0.3
        },
        variant_configs={
            "variant_a": {
                "embedding_model": "tfidf_only",
                "embedding_fallback": False,
                "search_threshold": 0.3
            },
            "variant_b": {
                "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
                "embedding_fallback": True,
                "search_threshold": 0.5  # Higher threshold
            }
        },
        success_metrics=["answer_relevance", "context_precision", "faithfulness", "overall_score"],
        sample_size=30
    )
    
    # Test 2: Academic Integrity Modes
    integrity_test = tester.create_test_configuration(
        test_name="integrity_modes_effectiveness",
        description="Compare effectiveness of different academic integrity modes",
        control_config={
            "integrity_mode": "academic",
            "academic_mode": True,
            "response_style": "guidance"
        },
        variant_configs={
            "variant_a": {
                "integrity_mode": "normal",
                "academic_mode": False,
                "response_style": "complete"
            },
            "variant_b": {
                "integrity_mode": "exam",
                "academic_mode": True,
                "response_style": "minimal"
            }
        },
        success_metrics=["integrity_compliance", "answer_completeness", "answer_relevance"],
        sample_size=25
    )
    
    # Test 3: Search and Reranking
    search_test = tester.create_test_configuration(
        test_name="search_reranking_optimization",
        description="Optimize search and reranking parameters for better results",
        control_config={
            "search_top_k": 30,
            "rerank_top_k": 6,
            "mmr_lambda": 0.2,
            "use_reranking": True
        },
        variant_configs={
            "variant_a": {
                "search_top_k": 50,
                "rerank_top_k": 10,
                "mmr_lambda": 0.2,
                "use_reranking": True
            },
            "variant_b": {
                "search_top_k": 30,
                "rerank_top_k": 6,
                "mmr_lambda": 0.5,  # More diversity
                "use_reranking": True
            },
            "variant_c": {
                "search_top_k": 30,
                "rerank_top_k": 6,
                "mmr_lambda": 0.2,
                "use_reranking": False  # No reranking
            }
        },
        success_metrics=["context_precision", "context_recall", "answer_relevance", "faithfulness"],
        sample_size=40
    )
    
    return tester, [embedding_test, integrity_test, search_test]


if __name__ == "__main__":
    print("🧪 A/B Testing Framework for AI Virtual Mentor")
    
    # Create standard test configurations
    tester, test_configs = create_standard_ab_tests()
    
    print(f"✅ Created {len(test_configs)} test configurations:")
    for config in test_configs:
        print(f"  - {config.test_name}: {config.description}")
    
    # Mock test questions for demonstration
    mock_questions = [
        {
            "question": "Machine Learning là gì?",
            "category": "theory_explanation",
            "difficulty": "intermediate",
            "expected_answer": "Machine Learning là một phần của AI..."
        },
        {
            "question": "Implement binary search algorithm",
            "category": "exercise",
            "difficulty": "intermediate",
            "expected_answer": "Binary search requires sorted array..."
        },
        {
            "question": "TCP vs UDP protocols",
            "category": "theory_explanation", 
            "difficulty": "intermediate",
            "expected_answer": "TCP is connection-oriented..."
        }
    ]
    
    # Run demonstration test (with mock mentor system)
    print(f"\n🚀 Running demonstration A/B test: {test_configs[1].test_name}")
    
    def mock_mentor_factory(config):
        """Mock mentor system factory for testing"""
        return None  # Will use mock responses
    
    try:
        results = tester.run_test_scenario(
            test_config=test_configs[1],  # Integrity modes test
            test_questions=mock_questions,
            mentor_system_factory=mock_mentor_factory
        )
        
        print(f"\n📊 A/B Test Results Summary:")
        print(f"Total test cases: {sum(len(tests) for tests in results.results_by_variant.values())}")
        print(f"Variants tested: {list(results.results_by_variant.keys())}")
        print(f"\n💡 Key Recommendations:")
        for rec in results.recommendations[:3]:
            print(f"  {rec}")
        
        # Export results
        export_success = tester.export_results(
            test_configs[1].test_name, 
            f"evaluation/ab_test_results_{test_configs[1].test_name}.json"
        )
        
        if export_success:
            print(f"✅ Results exported successfully")
        else:
            print(f"⚠️ Export failed")
            
    except Exception as e:
        print(f"❌ Test execution error: {e}")
    
    print("🎯 A/B Testing Framework Ready!")