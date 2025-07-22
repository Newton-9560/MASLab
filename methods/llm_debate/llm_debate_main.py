# The official implementation of LLM Debate https://github.com/composable-models/llm_multiagent_debate offen encounters errors.
# This is a modified version of the original implementation.

import os
from ..mas_base import MAS
from recorder import SingleResponse, SingleQuestion, MASDataset
from termcolor import colored

class LLM_Debate_Main(MAS):
    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)

        self.agents_num = self.method_config["agents_num"]
        self.rounds_num = self.method_config["rounds_num"]
    
    def inference(self, sample, dataset=None):
        if dataset is not None:
            single_question = SingleQuestion(sample["query"])
            single_question.set_gt_answer(sample)
        else:
            single_question = None
            

        query = sample["query"]

        agent_contexts = [[{"role": "user", "content": f"""{query} Make sure to state your answer at the end of the response."""}] for agent in range(self.agents_num)]

        for round in range(self.rounds_num):
            # print(colored(f">> Round {round + 1} of {self.rounds_num}", "white", 'on_red'))
            for i, agent_context in enumerate(agent_contexts):
                # print(colored(f"Agent number: {i}", "black", "on_blue"))
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = self.construct_message(agent_contexts_other, query, 2*round - 1)
                    agent_context.append(message)

                generation_list, logprob_list, token_list = self.call_llm(messages=agent_context, logprobs=True, number_generated=10)
                response = generation_list[0]
                agent_context.append({"role": "assistant", "content": response})
                if dataset is not None:
                    single_response = SingleResponse(agent_context, generation_list, logprob_list, token_list)
                    single_question.add_response(round, i, single_response)
        
        answers = [agent_context[-1]['content'] for agent_context in agent_contexts]
        
        final_answer = self.aggregate(query, answers, single_question)
        if dataset is not None:
            dataset.add_question(single_question)
        return {"response": final_answer}
    
    def construct_message(self, agents, question, idx):

        # Use introspection in the case in which there are no other agents.
        if len(agents) == 0:
            return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

        prefix_string = "These are the recent/updated opinions from other agents: "

        for agent in agents:
            agent_response = agent[idx]["content"]
            response = "\n\n One agent response: ```{}```".format(agent_response)

            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response. \n The original problem is {}.".format(question)
        return {"role": "user", "content": prefix_string}

    def aggregate(self, query, answers, single_question=None):
        aggregate_instruction = f"Task:\n{query}\n\n"
        for i, answer in enumerate(answers):
            aggregate_instruction += f"Solution {i+1}:\n{answer}\n\n"
        aggregate_instruction += "Given all the above solutions, reason over them carefully and provide a final answer to the task."
        # print(colored(f"Aggregate instruction: {aggregate_instruction}", "red"))
        generation_list, logprob_list, token_list = self.call_llm(prompt=aggregate_instruction, logprobs=True, number_generated=10)
        response = generation_list[0]
        if single_question is not None:
            single_question.set_final_response(SingleResponse(aggregate_instruction, generation_list, logprob_list, token_list))
        return response