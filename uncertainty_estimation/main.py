from manager import SemanticEntropyCalculator
from recorder import SingleResponse, MASDataset, SingleQuestion
from tqdm import tqdm
import torch
import gc

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
    # import os
    # from tqdm import tqdm
    
    # def iterate_files(folder_path):
    #     file_list = []
    #     for root, dirs, files in os.walk(folder_path):
    #         for file in files:
    #             file_path = os.path.join(root, file)
    #             file_list.append(file_path)
    #     return file_list

    # dataset_all = MASDataset("MMLU")
    # files = iterate_files('/home/hanwen/workspace/2025/MASLab/results/Meta-Llama-3-8B-Instruct')
    # for file in tqdm(files):
    #     dataset = MASDataset.load_pickle(file)
    #     dataset_all.concatenate_questions(dataset.questions)
    # dataset_all.save_pickle('/home/hanwen/workspace/2025/MASLab/results/Meta-Llama-3-8B-Instruct/MMLU_500/llm_debate_infer_semantic_entropy.pkl')
    dataset_path = '/home/hanwen/workspace/2025/MASLab/results/Meta-Llama-3-8B-Instruct/MMLU_500/llm_debate_infer_semantic_entropy.pkl'
    dataset = MASDataset.load_pickle(dataset_path)
    calculator = SemanticEntropyCalculator()
    for question in tqdm(dataset.questions):
        for round_num in question.responses:
            for agent_id in question.responses[round_num]:
                single_response = question.responses[round_num][agent_id]
                try:
                    results = calculator.compute_semantic_entropy(single_response)
                    single_response.uncertainty = results
                except Exception as e:
                    print(e)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    single_response.uncertainty = None
        final_response = question.final_response
        try:
            results = calculator.compute_semantic_entropy(final_response)
            final_response.uncertainty = results
        except Exception as e:
            print(e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    dataset.save_pickle(dataset_path)
            