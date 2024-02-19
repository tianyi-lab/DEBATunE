import os
import sys
import torch
import transformers
import argparse
from transformers import (
    LlamaForCausalLM, LlamaTokenizer, GenerationConfig)
from tqdm import tqdm
import json
import re

import time
import openai

import tiktoken
gpt_encoder = tiktoken.get_encoding("cl100k_base")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', default="Debate_topic/debate_train_v1.json", type=str)
    parser.add_argument('--save_path', default="try_debate_v1.jsonl", type=str)
    parser.add_argument("--arg_num", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7, help="")
    parser.add_argument("--top_p", type=float, default=1.0, help="")
    parser.add_argument("--do_sample", type=bool, default=True, help="")
    parser.add_argument('--model_engine', default="", type=str)
    parser.add_argument("--api_base",type=str,default='')
    parser.add_argument("--api_key", type=str, default='')
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    args = parser.parse_args()
    return args

Agent1_sys = 'You are attending a debate on the topic of "{topic}". You are {side1} to this topic and your opponent is {side2} to this topic.'
Agent2_sys = 'You are attending a debate on the topic of "{topic}". You are {side2} to this topic and your opponent is {side1} to this topic.'


Agent1_00 = \
"""You are attending a debate on the topic of "{topic}". You are {side1} to this topic.
Please provide your arguments on your {side1} side, you don't need to make detailed explanations right now, just list your arguments each in one line with a serial digit.
"""


Agent1_0 = \
"""You are currently debating for one specific argument proposed by you: "{argument}".
Please provide your detailed explanations, and supporting evidence for this argument. If possible, please include real-world examples and statistics. 
AI:
"""

Agent2_0 = \
"""You are currently debating for one specific argument proposed by your opponent: "{argument}".
Opponent: {argument_1_0} 
Please contradict your opponent's argument by raising critical questions, explanations, and supporting evidence. If possible, please include real-world examples and statistics. 
AI:
"""

Agent1_1 = \
"""You are currently debating for one specific argument proposed by you: "{argument}".
AI: {argument_1_0}
Opponent: {argument_2_0}
Please first answer your opponent's questions if any. Then contradict your opponent's argument by raising critical questions, explanations, and supporting evidence. If possible, please include real-world examples and statistics. 
AI:
"""

Agent2_1 = \
"""You are currently debating for one specific argument proposed by your opponent: "{argument}".
Opponent: {argument_1_0}
AI: {argument_2_0}
Opponent: {argument_1_1}
Please first answer your opponent's questions if any. Then contradict your opponent's argument by raising critical questions, explanations, and supporting evidence. If possible, please include real-world examples and statistics.
AI:
"""

Agent1_sys_new = 'You are watching a debate on the topic of "{topic}". You are {side1} to this topic.'
Agent1_2_new = \
"""You are watching a debating for one specific argument "{argument}" for the topic of "{topic}":
{side1}: {argument_1_0}
{side2}: {argument_2_0}
{side1}: {argument_1_1}
{side2}: {argument_2_1}
You are {side1} to this topic, based on the above debate, propose your argument, explanations, and supporting evidence, as detailed as possible. 
You do not need to mention or respond to the debate, just make your arguments as good as possible. 
"""


def call_chat(args, sys_prompt, prompt, max_tokens):
    response = ''
    message =[
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": prompt,
            },
    ]
    max_tokens = min(max_tokens,4040-len(gpt_encoder.encode(prompt)))
    retry = 0
    wait_base = 10
    while response== '':
        try:
            completion = openai.ChatCompletion.create(
                        model=args.model_engine,
                        messages=message,
                        temperature=args.temperature,
                        max_tokens=max_tokens,
                        top_p=args.top_p,
            )
            response = completion.choices[0].message.content
        except:
            retry += 1
            time.sleep(wait_base)
            wait_base = wait_base*2
    return response

def init_jsonl_file(args):

    with open(args.json_path,'r') as f:
        json_data = json.load(f)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(json_data)
    sampled_data = json_data[start_idx:end_idx]

    if not os.path.exists(args.save_path):
        with open(args.save_path, "w") as file:
            pass  # Creates an empty file

    with open(args.save_path, "r") as file:
        exsisting_num =  sum(1 for _ in file)
    sampled_data = sampled_data[exsisting_num:]

    return sampled_data

def main():
    args = parse_args()
    if args.api_base != '':
        openai.api_base = args.api_base
    openai.api_key = args.api_key

    sampled_data = init_jsonl_file(args)

    for topic in tqdm(sampled_data):
        temp_dict = {}
        temp_dict['topic'] = topic

        for i in range(2):
            if i == 0:
                side1 = "Affirmative"
                side2 = 'Negative'
            else:
                side1 = 'Negative'
                side2 = "Affirmative"

            Agent1_sys_prompt = Agent1_sys.format_map({"topic":topic,"side1":side1,"side2":side2})
            Agent2_sys_prompt = Agent2_sys.format_map({"topic":topic,"side1":side1,"side2":side2})

            # Define the arguments
            Agent1_00_prompt = Agent1_00.format_map({"topic":topic,"side1":side1,"side2":side2})
            Agent1_00_argument = call_chat(args, Agent1_sys_prompt, Agent1_00_prompt, 512)
            # print(Agent1_00_argument)
            # arguments = re.findall(r'\d+\.\s(.*?\.)', Agent1_00_argument)[:5]
            arguments = re.findall(r'\d+\.\s(.*?\.?)\s*(?=\d+\.|$)', Agent1_00_argument)[:args.arg_num]
            # print(arguments)
            temp_dict[side1] = {}
            temp_dict[side1]['arguments'] = arguments

            for argument in arguments:
                temp_dict[side1][argument] = []
                # print("===========================Argument===========================")
                # print(argument)
                # print("===========================Argument===========================")

                # Debate for each argument
                Agent1_0_prompt = Agent1_0.format_map({"topic":topic,"argument":argument,"side1":side1,"side2":side2})
                Agent1_0_argument = call_chat(args, Agent1_sys_prompt, Agent1_0_prompt, 800)
                temp_dict[side1][argument].append(Agent1_0_argument)
                # print('===========================Agent1_0_prompt===========================')
                # print(Agent1_0_argument)
                
                Agent2_0_prompt = Agent2_0.format_map({"topic":topic,"argument":argument,"side1":side1,"side2":side2,"argument_1_0":Agent1_0_argument})
                Agent2_0_argument = call_chat(args, Agent2_sys_prompt, Agent2_0_prompt, 800)
                temp_dict[side1][argument].append(Agent2_0_argument)
                # print('===========================Agent2_0_prompt===========================')
                # print(Agent2_0_argument)


                Agent1_1_prompt = Agent1_1.format_map({"topic":topic,"argument":argument,"side1":side1,"side2":side2,"argument_1_0":Agent1_0_argument,"argument_2_0":Agent2_0_argument})
                Agent1_1_argument = call_chat(args, Agent1_sys_prompt, Agent1_1_prompt, 800)
                temp_dict[side1][argument].append(Agent1_1_argument)
                # print('===========================Agent1_1_prompt===========================')
                # print(Agent1_1_argument)

                Agent2_1_prompt = Agent2_1.format_map({"topic":topic,"argument":argument,"side1":side1,"side2":side2,"argument_1_0":Agent1_0_argument,"argument_2_0":Agent2_0_argument,"argument_1_1":Agent1_1_argument})
                Agent2_1_argument = call_chat(args, Agent2_sys_prompt, Agent2_1_prompt, 800)
                temp_dict[side1][argument].append(Agent2_1_argument)
                # print('===========================Agent2_1_prompt===========================')
                # print(Agent2_1_argument)

                Agent1_2_prompt = Agent1_2_new.format_map({"topic":topic,"argument":argument,"side1":side1,"side2":side2,"argument_1_0":Agent1_0_argument,"argument_2_0":Agent2_0_argument, "argument_1_1":Agent1_1_argument, "argument_2_1":Agent2_1_argument})
                Agent1_2_argument = call_chat(args, Agent1_sys_new, Agent1_2_prompt, 2048)
                temp_dict[side1][argument].append(Agent1_2_argument)
                # print('===========================Agent1_2_prompt===========================')
                # print(Agent1_2_argument)

        with open(args.save_path, "a") as file:
            file.write(json.dumps(temp_dict) + '\n')

if __name__ == "__main__":
    main()