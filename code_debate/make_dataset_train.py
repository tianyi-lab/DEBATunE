import os
import json
import random
import argparse

INSTRUCTION_1_LIST = [
    """Suppose you are {side} to the topic "{topic}", please provide explanations and supporting evidence for this argument "{argument}" """,
    """Assuming your position is {side} on the subject of "{topic}", please offer justifications and substantiate your stance with this reasoning "{argument}" """,
    """If you hold the {side} perspective regarding "{topic}", please elucidate your viewpoint and back it up with this line of argumentation "{argument}" """,
    """Given that your stance is {side} in the debate over "{topic}", kindly provide detailed explanations and evidence to support this claim "{argument}" """,
    """In the event that you are {side} on the issue of "{topic}", please furnish detailed explanations and supporting evidence for this assertion "{argument}" """,
    """Should you be {side} in relation to the topic "{topic}", please present your explanations and corroborate them with this argument "{argument}" """,
    """When considering your position as {side} on "{topic}", please articulate your reasons and validate them with this evidence "{argument}" """,
    """Being {side} in the discussion about "{topic}", please expound on your viewpoint and bolster it with this rationale "{argument}" """,
    """If your stance aligns with {side} concerning "{topic}", kindly offer a detailed rationale and support it with this line of reasoning "{argument}" """,
    """Assuming you advocate the {side} position on "{topic}", please provide thorough explanations and fortify them with this argument "{argument}" """,
    """In case your perspective is {side} regarding "{topic}", please deliver comprehensive justifications and evidence supporting this proposition "{argument}" """
]

INSTRUCTION_2_LIST = [
    """Suppose you are {side} to the topic "{topic}", please provide some arguments with detailed explanations and supporting evidence.""",
    """If you take the {side} stance on the issue "{topic}", kindly present some arguments, complete with thorough explanations and supporting evidence.""",
    """Assuming your position is {side} regarding "{topic}", please offer various arguments accompanied by detailed explanations and substantiated evidence.""",
    """Should you align with the {side} viewpoint on "{topic}", please provide a number of arguments with in-depth explanations and relevant supporting evidence.""",
    """In the case that you are {side} on the matter of "{topic}", please furnish several arguments along with comprehensive explanations and corroborative evidence.""",
    """When you represent the {side} perspective on "{topic}", please deliver some arguments, including detailed explanations and appropriate evidence to support them.""",
    """Being {side} in relation to the topic "{topic}", kindly articulate some arguments with extensive explanations and accompanying evidence.""",
    """If your stance is {side} with respect to "{topic}", please enumerate several arguments, providing detailed explanations and evidence to back them up.""",
    """Given that you are {side} in the discussion of "{topic}", please present a series of arguments, each with thorough explanations and supporting evidence.""",
    """If you're positioned {side} concerning the subject "{topic}", please propose some arguments, each with well-elaborated explanations and suitable evidence.""",
    """Assuming a {side} position on the topic "{topic}", kindly put forward various arguments, ensuring each is accompanied by detailed explanations and evidence for support."""
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', default="try_debate_v1.jsonl", type=str)
    parser.add_argument('--save_path', default="try_debate_v1_train_sft.json", type=str)
    parser.add_argument('--arg_num', default=5, type=int)
    args = parser.parse_args()
    return args

def make_pair(args, data):

    data_all = []

    topic = data['topic']
    side_affirmative_arguments = data['Affirmative']['arguments']
    side_affirmative_explainations = data['Affirmative']
    side_negative_arguments = data['Negative']['arguments']
    side_negative_explainations = data['Negative']

    for i in range(2):
        if i == 0:
            side1 = "affirmative"
            arguments = side_affirmative_arguments
            explanations = side_affirmative_explainations
        else:
            side1 = 'negative'
            arguments = side_negative_arguments
            explanations = side_negative_explainations

        # Instruction: Topic + side + argument
        # Output: one argument, final round 
        for i in range(len(arguments)):
            if i == args.arg_num:
                break
            argument_i = arguments[i]
            INSTRUCTION_1 = random.choice(INSTRUCTION_1_LIST)
            instruction_i = INSTRUCTION_1.format_map({'side':side1, 'topic':topic, 'argument':argument_i})
            response_i = explanations[argument_i][-1]
            data_pair = {"instruction":instruction_i,"output":response_i}
            data_all.append(data_pair)

    return data_all


def main():
    args = parse_args()
    formated_data_all = []

    if args.json_path[-6:] == '.jsonl':
        data_all = []
        with open(args.json_path, 'r') as file:
            for line in file:
                data_all.append(json.loads(line.strip()))
    else:
        with open(args.json_path, "r") as f:
            data_all = json.load(f)

    for data_i in data_all:
        data_i_temp = make_pair(args, data_i)
        formated_data_all += data_i_temp
        pass

    # Write merged data to json file
    with open(args.save_path, 'w') as file:
        json.dump(formated_data_all, file, indent=4)
        pass

if __name__ == '__main__':
    main()