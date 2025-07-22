"""
Integration of semantic entropy computation with SingleResponse class.
Updated for 2D logprobs_list structure: [num_generations, num_logprobs_per_generation]
"""

import gc
import torch
import numpy as np
from model.se import (
    EntailmentDeberta,
    get_semantic_ids, logsumexp_by_id, predictive_entropy, predictive_entropy_rao,
    cluster_assignment_entropy, context_entails_response
)
import sys
sys.path.append("/home/hanwen/workspace/2025/MASLab")
from recorder import SingleResponse

class SemanticEntropyCalculator:
    """Helper class to compute semantic entropy for SingleResponse objects."""
    
    def __init__(self):
        """
        Initialize the semantic entropy calculator.
        """
        self.entailment_model = EntailmentDeberta()
    
    
    def compute_semantic_entropy(self, single_response: SingleResponse, strict_entailment=False, agg='sum_normalized', 
                               length_normalize=True):
        """
        Compute semantic entropy for a SingleResponse object following the reference implementation.
        
        Args:
            single_response: SingleResponse object containing generations and logprobs
            strict_entailment: Whether to use strict bidirectional entailment
            agg: Aggregation method for logsumexp ('sum_normalized')
            length_normalize: Whether to apply length normalization (mean of token logprobs)
            
        Returns:
            dict: Contains semantic entropy, cluster entropy, and other metrics
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Extract data from SingleResponse
        responses = single_response.generation_list
        log_liks = single_response.logprobs_list  # [num_generations, num_tokens]
        question = single_response.prompt
        
        
        # Apply length normalization of generation probabilities (following reference)
        if length_normalize:
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]
        else:
            # Sum across tokens for each generation
            log_liks_agg = [np.sum(log_lik) for log_lik in log_liks]
        
        responses_for_entailment = responses
        
        # Get semantic cluster IDs (following reference implementation)
        semantic_ids = get_semantic_ids(
            responses_for_entailment, 
            model=self.entailment_model, 
            strict_entailment=strict_entailment
        )
        
        # Compute entropy from frequencies of cluster assignments
        cluster_entropy = cluster_assignment_entropy(semantic_ids)
        
        # Compute regular (naive) entropy
        regular_entropy = predictive_entropy(log_liks_agg)
        
        # Compute semantic entropy using log-sum-exp and Rao's entropy formula
        log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg=agg)
        semantic_entropy = predictive_entropy_rao(log_likelihood_per_semantic_id)
        
        # Count unique semantic clusters
        n_clusters = len(set(semantic_ids))
        
        return {
            'semantic_entropy': semantic_entropy,
            'cluster_assignment_entropy': cluster_entropy,
            'regular_entropy': regular_entropy,
            'n_semantic_clusters': n_clusters,
            'semantic_ids': semantic_ids,
            'log_liks_agg': log_liks_agg,
            'log_likelihood_per_semantic_id': log_likelihood_per_semantic_id
        }

# Usage examples:

def example_usage():
    """Example of how to use SemanticEntropyCalculator with SingleResponse."""
    
    # Initialize calculator with DeBERTa model (fastest, no API calls needed)
    calculator = SemanticEntropyCalculator()

    
    # Example SingleResponse with correct 2D structure
    example_response = SingleResponse(
        prompt="What is the capital of France?",
        generation_list=[
            "Paris is the capital of France.",
            "The capital of France is Paris.", 
            "France's capital city is Paris.",
            "Paris",
            "The answer is Paris.",
            "It's Paris.",
            "Lyon is the capital of France.",  # Different/wrong answer
            "Paris, France",
            "The French capital is Paris.",
            "Paris is France's capital."
        ],
        # 2D array: [num_generations, num_tokens_per_generation]
        logprobs_list=[
            [-0.1, -0.05, -0.02, -0.03, -0.01],  # Generation 1 token logprobs
            [-0.08, -0.04, -0.02, -0.03],        # Generation 2 token logprobs  
            [-0.12, -0.06, -0.03, -0.04, -0.02], # Generation 3 token logprobs
            [-0.8],                               # Generation 4 token logprobs
            [-0.06, -0.04, -0.02, -0.03],        # Generation 5 token logprobs
            [-0.15, -0.25],                       # Generation 6 token logprobs
            [-1.2, -0.8, -0.3, -0.2],           # Generation 7 token logprobs (wrong answer)
            [-0.12, -0.13],                       # Generation 8 token logprobs
            [-0.07, -0.05, -0.03, -0.03],        # Generation 9 token logprobs
            [-0.05, -0.04, -0.02, -0.01]         # Generation 10 token logprobs
        ],
        # 2D array: [num_generations, num_tokens_per_generation]
        token_list=[
            ["Paris", "is", "the", "capital", "of"],
            ["The", "capital", "is", "Paris"],
            ["France's", "capital", "city", "is", "Paris"],
            ["Paris"],
            ["The", "answer", "is", "Paris"],
            ["It's", "Paris"],
            ["Lyon", "is", "the", "capital"],
            ["Paris", "France"],
            ["The", "French", "capital", "is"],
            ["Paris", "is", "France's", "capital"]
        ]
    )
    
    # Compute semantic entropy (matching reference implementation)
    results = calculator.compute_semantic_entropy(
        example_response,
        length_normalize=True,  # Apply length normalization like reference
    )
    
    print(f"Semantic Entropy: {results['semantic_entropy']:.4f}")
    print(f"Cluster Assignment Entropy: {results['cluster_assignment_entropy']:.4f}")
    print(f"Regular Entropy: {results['regular_entropy']:.4f}")
    print(f"Number of semantic clusters: {results['n_semantic_clusters']}")
    print(f"Semantic cluster assignments: {results['semantic_ids']}")

if __name__ == "__main__":
    example_usage()