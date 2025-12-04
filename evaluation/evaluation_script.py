"""
Evaluation Framework for Toastmasters RAG System
Measures retrieval quality, generation quality, and system performance
"""

import json
import time
import numpy as np
import yaml

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
    
    def __init__(self, rag_system: RAG, config_path: Optional[str] = None):
        """
        Args:
            rag_system: Instance of RAG class from rag_pipeline.main
            config_path: Path to config.yaml file (defaults to "../config.yaml")
        """
        self.rag: RAG = rag_system
        self.retriever: Retriever = rag_system.retriever
        self.results = []
        
        # Load config file to get query arguments
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                config = config_data.get('rag_pipeline', {})
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            print("Falling back to RAG object attributes")
            config = {}
        
        # Get values from config, with fallback to RAG object attributes
        self.top_k_retrieve = config.get('top_k_retrieve') or getattr(self.rag, "top_k_retrieve", 5)
        reranker_config = config.get('reranker', {})
        self.reranker_enabled = reranker_config.get('enable', False) or getattr(self.rag, "reranker_enabled", False)
        self.reranker_top_k = reranker_config.get('top_k') or getattr(self.rag, "top_k_rerank", None)
        
        # Determine which k to use for evaluation metrics
        # If reranking is enabled, use reranker top_k, else use top_k_retrieve
        if self.reranker_enabled and self.reranker_top_k is not None:
            self.eval_k = self.reranker_top_k
        else:
            self.eval_k = self.top_k_retrieve
        
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
            k: Number of results to evaluate (defaults to eval_k based on config: top_k_retrieve if reranker disabled, else reranker.top_k)
            
        Returns:
            Dictionary of metrics
        """
        # Use the k value determined from config (top_k_retrieve if reranker disabled, else reranker.top_k)
        k = k or self.eval_k

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
            # Use config values for retrieval and reranking
            # If reranker is disabled, use top_k_retrieve for selection; otherwise use reranker_top_k
            top_k_rerank_value = self.reranker_top_k if self.reranker_enabled else self.top_k_retrieve
            pipeline_result = self.rag.query(
                query,
                top_k_retrieve=self.top_k_retrieve,
                top_k_rerank=top_k_rerank_value,
                generate_response=False
            )
            retrieval_time = time.time() - start
            retrieval_times.append(retrieval_time)
            
            selected_results = pipeline_result.get('selected_results') or []
            retrieved_global_ids = [
                r['global_id'] for r in selected_results if 'global_id' in r
            ]
            
            # Calculate metrics using the selected k value
            # Limit to k results for metric calculation
            retrieved_global_ids = retrieved_global_ids[:k]
            
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

        
        # Retrieval evaluation
        report_lines.append("RETRIEVAL QUALITY METRICS")
        report_lines.append("-" * 80)
        # Use eval_k which is determined from config (top_k_retrieve if reranker disabled, else reranker.top_k)
        retrieval_metrics = self.evaluate_retrieval(test_cases_path, k=self.eval_k)
        report_lines.append(f"Evaluation using k={self.eval_k} ({'reranker.top_k' if self.reranker_enabled else 'top_k_retrieve'})")
        report_lines.append(f"Reranker enabled: {self.reranker_enabled}")
        report_lines.append(f"Precision@{self.eval_k}: {retrieval_metrics['precision@k']:.3f}")
        report_lines.append(f"Recall@{self.eval_k}: {retrieval_metrics['recall@k']:.3f}")
        report_lines.append(f"F1@{self.eval_k}: {retrieval_metrics['f1@k']:.3f}")
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
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    evaluator = RAGEvaluator(rag, config_path=config_path)
    
    # Generate report
    print("\nGenerating evaluation report...")
    test_cases_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    report = evaluator.generate_report(
        test_cases_path=test_cases_path,
        output_path="evaluation_report.txt"
    )
    
    print("\n" + report)
