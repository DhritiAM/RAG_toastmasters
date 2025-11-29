"""
Evaluation Framework for Toastmasters RAG System
Measures retrieval quality, generation quality, and system performance
"""

import json
import time
import numpy as np

import sys
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict

# Add parent directory to path to import RAG class
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rag_pipeline.main import RAG
from rag_pipeline.retriever import Retriever

@dataclass
class EvaluationMetric:
    """Container for evaluation metrics"""
    name: str
    value: float
    description: str

class RAGEvaluator:
    """
    Comprehensive evaluation framework for RAG systems
    Implements multiple metrics for retrieval and generation quality
    """
    
    def __init__(self, rag_system: RAG):
        """
        Args:
            rag_system: Instance of RAG class from rag_pipeline.main
        """
        self.rag: RAG = rag_system
        self.retriever: Retriever = rag_system.retriever
        self.results = []
        self.retrieval_top_k = getattr(self.rag, "top_k_retrieve", None)
        self.reranker_enabled = getattr(self.rag, "reranker_enabled", False)
        self.reranker_top_k = getattr(self.rag, "top_k_rerank", None)
        
        # Load metadata for coverage analysis
        with open(self.rag.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
    
    def evaluate_retrieval(
        self, 
        test_cases_path: str,
        k: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality using test cases
        
        Args:
            test_cases_path: Path to JSON file with test cases containing 'query' and 'relevant_global_ids'
            k: Number of results to retrieve (defaults to config top_k_retrieve)
            
        Returns:
            Dictionary of metrics
        """
        k = k or self.retrieval_top_k or 5

        precisions = []
        recalls = []
        mrrs = []  # Mean Reciprocal Rank
        retrieval_times = []

        # Load test cases
        with open(test_cases_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
        
        for test_case in test_cases:
            query = test_case['query']
            relevant_global_ids = set(test_case['relevant_global_ids'])
            
            # Time the retrieval
            start = time.time()
            pipeline_result = self.rag.query(
                query,
                top_k_retrieve=k,
                top_k_rerank=k,
                generate_response=False
            )
            retrieval_time = time.time() - start
            retrieval_times.append(retrieval_time)
            
            selected_results = pipeline_result.get('selected_results') or []
            retrieved_global_ids = [
                r['global_id'] for r in selected_results if 'global_id' in r
            ]
            
            # Calculate metrics
            relevant_retrieved = set(retrieved_global_ids) & relevant_global_ids
            
            # Precision@k
            precision = len(relevant_retrieved) / len(retrieved_global_ids) if retrieved_global_ids else 0
            precisions.append(precision)
            
            # Recall@k
            recall = len(relevant_retrieved) / len(relevant_global_ids) if relevant_global_ids else 0
            recalls.append(recall)
            
            # MRR (Mean Reciprocal Rank)
            reciprocal_rank = 0
            for i, global_id in enumerate(retrieved_global_ids, 1):
                if global_id in relevant_global_ids:
                    reciprocal_rank = 1 / i
                    break
            mrrs.append(reciprocal_rank)
        
        return {
            'precision@k': np.mean(precisions),
            'recall@k': np.mean(recalls),
            'f1@k': 2 * np.mean(precisions) * np.mean(recalls) / (np.mean(precisions) + np.mean(recalls)) if (np.mean(precisions) + np.mean(recalls)) > 0 else 0,
            'mrr': np.mean(mrrs),
            'avg_retrieval_time': np.mean(retrieval_times),
            'std_retrieval_time': np.std(retrieval_times)
        }
    
    def evaluate_relevance_distribution(
        self, 
        queries: List[str], 
        k: Optional[int] = None
    ) -> Dict:
        """
        Analyze the distribution of retrieval scores
        
        Args:
            queries: List of test queries
            k: Number of results per query (defaults to config top_k_retrieve)
            
        Returns:
            Statistics about score distribution
        """
        k = k or self.retrieval_top_k or 5

        all_scores = []
        score_ranges = defaultdict(int)
        
        for query in queries:
            results = self.retriever.retrieve(query, top_k=k)
            scores = [r['score'] for r in results]
            all_scores.extend(scores)
            
            # Categorize scores
            for score in scores:
                if score >= 0.8:
                    score_ranges['high (≥0.8)'] += 1
                elif score >= 0.6:
                    score_ranges['medium (0.6-0.8)'] += 1
                else:
                    score_ranges['low (<0.6)'] += 1
        
        return {
            'mean_score': np.mean(all_scores),
            'median_score': np.median(all_scores),
            'std_score': np.std(all_scores),
            'min_score': np.min(all_scores),
            'max_score': np.max(all_scores),
            'score_distribution': dict(score_ranges)
        }
    
    def evaluate_coverage(self) -> Dict:
        """
        Evaluate how well the knowledge base covers different topics
        
        Returns:
            Coverage statistics
        """
        # Analyze source distribution from metadata
        chunks_per_source = defaultdict(int)
        for meta in self.metadata:
            chunks_per_source[meta['source_file']] += 1
        
        values = list(chunks_per_source.values())
        
        return {
            'total_sources': len(chunks_per_source),
            'total_chunks': len(self.metadata),
            'avg_chunks_per_source': np.mean(values) if values else 0,
            'std_chunks_per_source': np.std(values) if values else 0,
            'min_chunks_per_source': min(values) if values else 0,
            'max_chunks_per_source': max(values) if values else 0,
            'source_balance': np.std(values) / np.mean(values) if values and np.mean(values) > 0 else 0,  # Lower is more balanced
        }
    
    def evaluate_latency(
        self, 
        queries: List[str], 
        k: Optional[int] = None,
        num_runs: int = 10
    ) -> Dict:
        """
        Benchmark system latency under different conditions
        
        Args:
            queries: Test queries
            k: Number of results to retrieve (defaults to config top_k_retrieve)
            num_runs: Number of times to run each query
            
        Returns:
            Latency statistics
        """
        k = k or self.retrieval_top_k or 5

        latencies = []
        
        for query in queries:
            query_latencies = []
            for _ in range(num_runs):
                start = time.time()
                _ = self.retriever.retrieve(query, top_k=k)
                latency = (time.time() - start) * 1000  # Convert to ms
                query_latencies.append(latency)
            latencies.extend(query_latencies)
        
        return {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'std_latency_ms': np.std(latencies)
        }
    
        
    def generate_report(
        self,
        test_cases_path: str,
        output_path: str = "evaluation_report.txt"
    ) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            test_cases_path: Path to test cases JSON file with ground truth for retrieval eval
            output_path: Path to save report
            
        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("TOASTMASTERS RAG SYSTEM - EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # System info
        coverage = self.evaluate_coverage()
        report_lines.append("SYSTEM STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Chunks: {coverage['total_chunks']}")
        report_lines.append(f"Total Sources: {coverage['total_sources']}")
        report_lines.append(f"Retrieval top_k (config): {self.retrieval_top_k or 'default (5)'}")
        report_lines.append(f"Reranker enabled: {self.reranker_enabled}")
        if self.reranker_enabled:
            report_lines.append(f"Reranker top_k (config): {self.reranker_top_k or 'default'}")
        report_lines.append("")
        
        # Coverage evaluation
        report_lines.append("KNOWLEDGE BASE COVERAGE")
        report_lines.append("-" * 80)
        report_lines.append(f"Sources: {coverage['total_sources']}")
        report_lines.append(f"Chunks per source (avg): {coverage['avg_chunks_per_source']:.1f}")
        report_lines.append(f"Balance score: {coverage['source_balance']:.3f} (lower is better)")
        report_lines.append("")
        
        # Retrieval evaluation
        report_lines.append("RETRIEVAL QUALITY METRICS")
        report_lines.append("-" * 80)
        eval_k = self.retrieval_top_k or 5
        retrieval_metrics = self.evaluate_retrieval(test_cases_path, k=eval_k)
        report_lines.append(f"Precision@{eval_k}: {retrieval_metrics['precision@k']:.3f}")
        report_lines.append(f"Recall@{eval_k}: {retrieval_metrics['recall@k']:.3f}")
        report_lines.append(f"F1@{eval_k}: {retrieval_metrics['f1@k']:.3f}")
        report_lines.append(f"Mean Reciprocal Rank: {retrieval_metrics['mrr']:.3f}")
        report_lines.append(f"Avg Retrieval Time: {retrieval_metrics['avg_retrieval_time']*1000:.2f}ms")
        report_lines.append("")
        
        
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"\nEvaluation report saved to {output_path}")
        
        return report


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    print("Loading RAG system...")
    rag = RAG(
        config_path="../config.yaml",
        verbose=False
    )
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = RAGEvaluator(rag)
    
    # Generate report
    print("\nGenerating evaluation report...")
    test_cases_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    report = evaluator.generate_report(
        test_cases_path=test_cases_path,
        output_path="evaluation_report.txt"
    )
    
    print("\n" + report)
