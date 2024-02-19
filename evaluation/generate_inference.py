from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
import argparse
import json
import os
from tqdm import tqdm

PROMPT_DICT_ALPACA = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
PROMPT_DICT_WIZARDLM = {
    "prompt_input": (
        "{instruction}\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "{instruction}\n\n### Response:"
    ),
}
PROMPT_DICT_VICUNA = {
    "prompt_input": (
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction}\nInput:\n{input} ASSISTANT:"
    ),
    "prompt_no_input": (
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} ASSISTANT:"
    ),
}
PROMPT_DICT_LLAMA2 = { 
    "prompt_input": (
"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction}\nInput:\n{input} [/INST]
"""),
    "prompt_no_input": (
"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST]
""")
}

PROMPT_DICT_ZEPHYR = { 
    "prompt_input": (
"""<|system|>
</s>
<|user|>
{instruction}\nInput:\n{input}</s>
<|assistant|>
"""),
    "prompt_no_input": (
"""<|system|>
</s>
<|user|>
{instruction}</s>
<|assistant|>
""")
}


DEBATE_INSTRUCTION = """Suppose you are {side} to the topic "{topic}", please provide explanations and supporting evidence for this argument "{argument}" """

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument('--json_path', default="Debate_data/debate_test_2.jsonl", type=str)
    parser.add_argument('--save_file', default="debate_inference", type=str)
    parser.add_argument('--file_name', default="debate_test_arg", type=str)
    parser.add_argument('--output_dir', default="", type=str)
    parser.add_argument('--prompt', default='vicuna', type=str)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--arg_num", type=int, default=3)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", cache_dir="../cache/")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, device_map="auto", cache_dir="../cache/")
    model.to(device)
    model.eval()

    if args.prompt == 'alpaca':
        prompt_input = PROMPT_DICT_ALPACA["prompt_input"]
        prompt_no_input = PROMPT_DICT_ALPACA["prompt_no_input"]
    elif args.prompt == 'wiz':
        prompt_input = PROMPT_DICT_WIZARDLM["prompt_input"]
        prompt_no_input = PROMPT_DICT_WIZARDLM["prompt_no_input"]
    elif args.prompt == 'vicuna':
        prompt_input = PROMPT_DICT_VICUNA["prompt_input"]
        prompt_no_input = PROMPT_DICT_VICUNA["prompt_no_input"]
    elif args.prompt == 'llama2':
        prompt_input = PROMPT_DICT_LLAMA2["prompt_input"]
        prompt_no_input = PROMPT_DICT_LLAMA2["prompt_no_input"]
    elif args.prompt == 'zephyr':
        prompt_input = PROMPT_DICT_ZEPHYR["prompt_input"]
        prompt_no_input = PROMPT_DICT_ZEPHYR["prompt_no_input"]

    json_data = []
    with open(args.json_path, 'r') as file:
        for line in file:
            json_data.append(json.loads(line.strip()))

    results = []
    for data in tqdm(json_data):

        topic = data['topic']
        side_affirmative_arguments = data['Affirmative']['arguments']
        side_negative_arguments = data['Negative']['arguments']

        for i in range(2):
            if i == 0:
                side1 = "affirmative"
                arguments = side_affirmative_arguments
            else:
                side1 = 'negative'
                arguments = side_negative_arguments

            for jj, argument_i in enumerate(arguments):
                if jj >= args.arg_num :
                    break
                
                debate_instruction = DEBATE_INSTRUCTION.format_map({'side':side1, 'topic':topic, 'argument':argument_i})
                prompt = prompt_no_input.format_map({"instruction":debate_instruction})

                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs.input_ids.to(device)
                generate_ids = model.generate(input_ids, max_length=args.max_length)
                outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

                point = {}
                point['topic'] = topic
                point['side'] = side1
                point['argument'] = argument_i
                point['instruction'] = debate_instruction
                point['raw_output'] = outputs
                if args.prompt in ['alpaca','wiz']:
                    point['response'] = outputs.split("Response:")[1]
                elif args.prompt in ['vicuna']:
                    point['response'] = outputs.split("ASSISTANT:")[1]
                elif args.prompt in ['llama2']:
                    point['response'] = outputs.split("[/INST]")[1]
                elif args.prompt in ['zephyr']:
                    point['response'] = outputs.split("<|assistant|>")[1]

                results.append(point)

    if args.output_dir == '':
        output_dir =  os.path.join(args.model_name_or_path, args.save_file)
    else:
        output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    saved_name = args.file_name + "_" + str(args.max_length) + ".json"
    with open(os.path.join(output_dir, saved_name), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()