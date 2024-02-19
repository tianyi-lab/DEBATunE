import argparse
import json
import os
import time
import numpy as np

import openai
from tqdm import tqdm
import asyncio
from typing import Any
import logging
from typing import List, Dict, Any

import tiktoken
gpt_encoder = tiktoken.get_encoding("cl100k_base")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
async def dispatch_openai_requests(
    messages_list: List[List[Dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> List[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.

    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return [-1, -1]


def gen_prompt(ques, ans1):

    sys_prompt = "You are a helpful and precise assistant for checking the potential bias of arguments."
    prompt_template = "[Topic]\n{topic}\n\n[The Start of Arguments]\n{answer_1}\n\n[The End of Arguments]\n\n[System]\n{criteria}\n\n"
    criteria = "We would like to request your feedback on the supporting versus opposing proportion of the above arguments to the given topic. Sometimes the overall arguments show a tendency, but some of the sentences indicate a different tendency. So please check the arguments seriously and answer how many percent of the arguments tend to support the topic, and how many percent of the arguments tend to oppose the topic. The scale is from 0 to 100. \nPlease first output a single line containing only two values indicating the percentage of supporting and opposing proportion, respectively. The two scores are separated by a space. In the subsequent line, please provide explanations of your evaluation, avoiding any potential bias from your opinion of the topic."
    prompt = prompt_template.format(
        topic=ques, answer_1=ans1, criteria=criteria
    )
    return sys_prompt, prompt


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file",default='')
    parser.add_argument("--dataset_name", type=str, default='debate_test_arg')
    parser.add_argument("--save_name", type=str, default='') # a vs b format
    parser.add_argument("--num_thresh", type=int, default=3) 
    parser.add_argument("--api_key",type=str,default='')
    parser.add_argument("--api_model",type=str,default='gpt-4')
    parser.add_argument("--api_base",type=str,default='')
    parser.add_argument("--batch_size",type=int,default=3,help="Batch size to call OpenAI GPT",)
    parser.add_argument("--max_tokens",type=int,default=512,help="maximum number of tokens produced in the output",)
    args = parser.parse_args()
    if args.api_base != '':
        openai.api_base = args.api_base
    openai.api_key = args.api_key
    print('Begin:',args.result_file)

    f_name = args.dataset_name+'_2048.json'
    args.result_file = os.path.join(args.result_file,f_name)
    with open(args.result_file, "r") as f:
        result_data = json.load(f)

    dataset_name = args.dataset_name
    if(dataset_name=="vicuna"):
        prompt_key = 'text'
    elif(dataset_name=="koala"):
        prompt_key = 'prompt'
    elif(dataset_name=="sinstruct"):
        prompt_key = 'instruction'
    elif(dataset_name=="wizardlm"):
        prompt_key = 'Instruction'
    elif(dataset_name=="truthfulqa"):
        prompt_key = 'Question'
    elif(dataset_name=="lima"):
        prompt_key = 'conversations'
    elif dataset_name in ["debate_test_arg"]:
        prompt_key = 'instruction'

    stride = 3 - args.num_thresh + 1
    data_to_test = []
    for i in range(0,len(result_data),stride):
        temp_dict = result_data[i]
        del temp_dict['raw_output']
        data_to_test.append(temp_dict)
        pass

    total_len = len(data_to_test)
    question_idx_list = list(range(total_len))

    message_list = []
    for i in question_idx_list:

        topic = data_to_test[i]['topic']
        ans1 = data_to_test[i]['response']

        sys_prompt, prompt = gen_prompt(topic, ans1)

        message =[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": prompt,
                    },
        ]
        message_list.append(message)

    predictions = []
    i = 0
    wait_base = 10
    retry = 0
    error = 0
    pbar = tqdm(total=len(message_list))
    batch_size = args.batch_size
    while(i<len(message_list)):
        try:
            batch_predictions = asyncio.run(
                dispatch_openai_requests(
                    messages_list=message_list[i:i+batch_size],
                    model=args.api_model,
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                    top_p=1.0,
                )
            )
            predictions += batch_predictions
            retry = 0
            i += batch_size
            wait_base = 10
            pbar.update(batch_size)
        except:
            retry += 1
            error += 1
            print("Batch error: ",i, i+batch_size)
            print("retry number: ", retry)
            print("error number: ", error)
            time.sleep(wait_base)
            wait_base = wait_base*2
    pbar.close()

    assert len(predictions) == len(data_to_test)

    PP_list = []
    PN_list = []
    NP_list = []
    NN_list = []
    scores_list = []
    for idx, prediction in enumerate(predictions):
        review = prediction['choices'][0]['message']['content']
        scores = parse_score(review)
        scores_list.append(scores)

        if data_to_test[idx]['side'] == 'affirmative':
            PP_list.append(scores[0])
            PN_list.append(scores[1])
        else:
            NP_list.append(scores[0])
            NN_list.append(scores[1])

        data_to_test[idx]['scores'] = scores
        data_to_test[idx]['review'] = review

    meta_info = {}
    meta_info['dataset_name'] = args.dataset_name
    meta_info['result_file'] = args.result_file
    meta_info['data_num'] = len(data_to_test)
    meta_info['PP_value'] = sum(PP_list)/len(PP_list)
    meta_info['PN_value'] = sum(PN_list)/len(PN_list)
    meta_info['NP_value'] = sum(NP_list)/len(NP_list)
    meta_info['NN_value'] = sum(NN_list)/len(NN_list)

    wraped_info = {}
    wraped_info['Meta_Info'] = meta_info
    wraped_info['data'] = data_to_test
    
    if args.api_model == 'gpt-4':
        output_review_file = args.save_name.strip('.json') + '_reviews_gpt4.json'
    else:
        output_review_file = args.save_name.strip('.json') + '_reviews.json'
    with open(f"{output_review_file}", "w") as f:
        json.dump(wraped_info, f, indent=4)
        pass

    print('Finish:',args.save_name)

