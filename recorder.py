from collections import defaultdict
import pickle
import os
from termcolor import colored

class SingleResponse:
    def __init__(self, prompt, generation_list, logprobs_list, token_list, uncertainty=None):
        self.prompt = prompt # the query of the response
        self.generation_list = generation_list # the generation list of the response, we sample 10 generations for each response
        self.logprobs_list = logprobs_list # the logprobs list of the response, we sample 10 logprobs for each response [num_generations, num_logprobs]
        self.token_list = token_list # the token list of the response, we sample 10 tokens for each response [num_generations, num_tokens]
        self.uncertainty = uncertainty # the uncertainty of the response
        
    def to_dict(self):
        return {
            "prompt": self.prompt,
            "generation_list": self.generation_list,
            "logprobs_list": self.logprobs_list,
            "token_list": self.token_list,
            "uncertainty": self.uncertainty
        }

class SingleQuestion:
    def __init__(self, question, gt_answer=None, final_response=None, correct=None, uncertainty_estimation=None):
        self.question = question
        self.responses = defaultdict(dict)
        self.gt_answer = gt_answer
        self.final_response = None
        self.correct = None
        self.uncertainty_estimation = None
        
    def add_response(self, round_num, agent_id, single_response: SingleResponse):
        self.responses[round_num][agent_id] = single_response
        
    def set_final_response(self, final_response: SingleResponse):
        self.final_response = final_response
        
    def set_gt_answer(self, gt_answer):
        self.gt_answer = gt_answer
        
    def set_correct(self, correct):
        self.correct = correct
        
    def set_uncertainty_estimation(self, uncertainty_estimation):
        self.uncertainty_estimation = uncertainty_estimation
        
    def to_dict(self):
        return {
            "question": self.question,
            "responses": self.responses,
            "gt_answer": self.gt_answer,
            "correct": self.correct,
            "uncertainty_estimation": self.uncertainty_estimation,
            "final_response": self.final_response.to_dict()
        }
    
class MASDataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.questions = []
    
    def add_question(self, question: SingleQuestion):
        if question.gt_answer is None:
            print(colored(f"Ground truth answer is not set", "red"))
        if question.final_response is None:
            print(colored(f"Final response is not set", "red"))
        self.questions.append(question)
        
    def concatenate_questions(self, questions: list[SingleQuestion]):
        for question in questions:
            self.add_question(question)
            
    def __len__(self):
        return len(self.questions)
    
    def to_dict(self):
        return {
            "dataset_name": self.dataset_name,
            "questions": [question.to_dict() for question in self.questions]
        }

    def save_pickle(self, filepath):
        """Save dataset as pickle file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        # print(f"Dataset saved to {filepath} with {len(self.questions)} questions")
    
    @classmethod
    def load_pickle(cls, filepath):
        """Load dataset from pickle file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
        
if __name__ == "__main__":
    dataset = MASDataset.load_pickle("/home/hanwen/workspace/2025/MASLab/results/Meta-Llama-3-8B-Instruct/MMLU/llm_debate_infer.pkl")
    print(len(dataset.questions))