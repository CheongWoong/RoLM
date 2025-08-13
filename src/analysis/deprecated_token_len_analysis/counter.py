import jsonlines
import json
import os
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import defaultdict
from utils.prompt_formatting import format_example
from utils.model_map import MODEL_MAP
import tiktoken

def main(args):
    model_name = args.model_name
    prompting_strategy = args.prompting_strategy
    task_name = args.task_name

    # Load tokenizer and model
    if "gpt-4o" in model_name:
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[model_name])


    data_token_len = []
    format_0 = []
    format_1 = []
    format_2 = []
    format_3 = []
    format_4 = []
    format_5 = []
    format_6 = []
    format_7 = []
    min_token_format = defaultdict(int)

    output_path = f"results/{task_name}/{model_name}/{prompting_strategy}_tokenlen.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    input_path = f"preprocessed_datasets/{task_name}_test_4_shot.jsonl"
    with open(output_path, "w") as fout:
        with jsonlines.open(input_path) as fin:
            # print the number of examples in the input file
            lines = sum(1 for line in fin)
            print(f"Number of examples: {lines}")
            print(f"Total counts of samples: {lines} * 8 = {lines * 8}")
        with jsonlines.open(input_path) as fin:
            for example in tqdm(fin.iter()):
                row = {
                    "id": example["id"],
                    "token_len": {},
                    "diff_token_len": {},
                    "min_token_len": 1000000,
                    "min_format": []
                }

                for format_type in [str(num) for num in range(8)]:
                    instruction, user_input = format_example(example, task_name, prompting_strategy, format_type=format_type)

                    if "Llama" in model_name:
                        messages = [{"role": "system", "content": instruction}, {"role": "user", "content": user_input}]
                    else:
                        messages = [{"role": "user", "content": instruction + "\n\n" + user_input}]

                    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")

                    if "gpt-4o" in model_name:
                        tokens_per_message = 3
                        tokens_per_name = 1

                        for message in messages:
                            input_length += tokens_per_message
                            for key, value in message.items():
                                input_length += len(tokenizer.encode(value))
                                if key == "name":
                                    input_length += tokens_per_name
                        input_length += 3 # every reply is primed with <|start|>assistant<|message|>
                    else:
                        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")

                        input_length = len(input_ids[0])
                    row["token_len"][format_type] = input_length
                    # print(input_length)
                    if input_length < row["min_token_len"]:
                        row["min_token_len"] = input_length
                        row["min_format"] = [format_type]
                    elif input_length == row["min_token_len"]:
                        row["min_format"].append(format_type)
                        

                    for format, token_len in row["token_len"].items():
                        row["diff_token_len"][format] = token_len - row["min_token_len"]
                        assert row["diff_token_len"][format] >= 0
                data_token_len.append(sum(row["token_len"].values())/8)
                format_0.append(row["diff_token_len"]["0"])
                format_1.append(row["diff_token_len"]["1"])
                format_2.append(row["diff_token_len"]["2"])
                format_3.append(row["diff_token_len"]["3"])
                format_4.append(row["diff_token_len"]["4"])
                format_5.append(row["diff_token_len"]["5"])
                format_6.append(row["diff_token_len"]["6"])
                format_7.append(row["diff_token_len"]["7"])
                
                
                for format in row["min_format"]:
                    min_token_format[format] += 1

                json.dump(row, fout)
                fout.write("\n")
    print(f"{task_name} Average token length: ", sum(data_token_len)/len(data_token_len))
    mean_0, mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7 = sum(format_0)/len(format_0), sum(format_1)/len(format_1), sum(format_2)/len(format_2), sum(format_3)/len(format_3), sum(format_4)/len(format_4), sum(format_5)/len(format_5), sum(format_6)/len(format_6), sum(format_7)/len(format_7)
    print(f"{task_name} format 0 mean: {mean_0}, std: {sum([(x - mean_0)**2 for x in format_0])/(len(format_0)-1)}")
    print(f"{task_name} format 1 mean: {mean_1}, std: {sum([(x - mean_1)**2 for x in format_1])/(len(format_1)-1)}")
    print(f"{task_name} format 2 mean: {mean_2}, std: {sum([(x - mean_2)**2 for x in format_2])/(len(format_2)-1)}")
    print(f"{task_name} format 3 mean: {mean_3}, std: {sum([(x - mean_3)**2 for x in format_3])/(len(format_3)-1)}")
    print(f"{task_name} format 4 mean: {mean_4}, std: {sum([(x - mean_4)**2 for x in format_4])/(len(format_4)-1)}")
    print(f"{task_name} format 5 mean: {mean_5}, std: {sum([(x - mean_5)**2 for x in format_5])/(len(format_5)-1)}")
    print(f"{task_name} format 6 mean: {mean_6}, std: {sum([(x - mean_6)**2 for x in format_6])/(len(format_6)-1)}")
    print(f"{task_name} format 7 mean: {mean_7}, std: {sum([(x - mean_7)**2 for x in format_7])/(len(format_7)-1)}")
    print(f"{task_name} min token format: {min_token_format}")

def analyze_token_length(model_name, task_name, prompting_strategy):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[model_name])

    max_new_tokens = 20
    if "cot" in prompting_strategy:
        max_new_tokens += 100

    data_token_len = []
    format_0 = []
    format_1 = []
    format_2 = []
    format_3 = []
    format_4 = []
    format_5 = []
    format_6 = []
    format_7 = []
    min_token_format = defaultdict(int)

    output_path = f"results/{task_name}/{model_name}/{prompting_strategy}_tokenlen.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    input_path = f"preprocessed_datasets/{task_name}_test.jsonl"
    with open(output_path, "w") as fout:
        with jsonlines.open(input_path) as fin:
            for example in tqdm(fin.iter()):
                row = {
                    "id": example["id"],
                    "token_len": {},
                    "diff_token_len": {},
                    "min_token_len": 1000000,
                    "min_format": []
                }

                for format_type in [str(num) for num in range(8)]:
                    formatted_example = format_example(example, task_name, prompting_strategy, format_type=format_type)

                    if "Instruct" in model_name or "instruct" in model_name:
                        messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": formatted_example}]
                        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
                    else:
                        input_ids = tokenizer.encode(formatted_example, return_tensors="pt").to("cuda")

                    input_length = len(input_ids[0])
                    row["token_len"][format_type] = input_length
                    # print(input_length)
                    if input_length < row["min_token_len"]:
                        row["min_token_len"] = input_length
                        row["min_format"] = [format_type]
                    elif input_length == row["min_token_len"]:
                        row["min_format"].append(format_type)
                        

                    for format, token_len in row["token_len"].items():
                        row["diff_token_len"][format] = token_len - row["min_token_len"]
                        assert row["diff_token_len"][format] >= 0
                data_token_len.append(sum(row["token_len"].values())/8)
                format_0.append(row["diff_token_len"]["0"])
                format_1.append(row["diff_token_len"]["1"])
                format_2.append(row["diff_token_len"]["2"])
                format_3.append(row["diff_token_len"]["3"])
                format_4.append(row["diff_token_len"]["4"])
                format_5.append(row["diff_token_len"]["5"])
                format_6.append(row["diff_token_len"]["6"])
                format_7.append(row["diff_token_len"]["7"])
                
                
                for format in row["min_format"]:
                    min_token_format[format] += 1

                json.dump(row, fout)
                fout.write("\n")
    print(f"{task_name} Average token length: ", sum(data_token_len)/len(data_token_len))
    mean_0, mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7 = sum(format_0)/len(format_0), sum(format_1)/len(format_1), sum(format_2)/len(format_2), sum(format_3)/len(format_3), sum(format_4)/len(format_4), sum(format_5)/len(format_5), sum(format_6)/len(format_6), sum(format_7)/len(format_7)
    formats_mean_std = {
        "format0": (mean_0, sum([(x - mean_0)**2 for x in format_0])/(len(format_0)-1)),
        "format1": (mean_1, sum([(x - mean_1)**2 for x in format_1])/(len(format_1)-1)),
        "format2": (mean_2, sum([(x - mean_2)**2 for x in format_2])/(len(format_2)-1)),
        "format3": (mean_3, sum([(x - mean_3)**2 for x in format_3])/(len(format_3)-1)),
        "format4": (mean_4, sum([(x - mean_4)**2 for x in format_4])/(len(format_4)-1)),
        "format5": (mean_5, sum([(x - mean_5)**2 for x in format_5])/(len(format_5)-1)),
        "format6": (mean_6, sum([(x - mean_6)**2 for x in format_6])/(len(format_6)-1)),
        "format7": (mean_7, sum([(x - mean_7)**2 for x in format_7])/(len(format_7)-1))
    }
    min_token_format = dict(min_token_format)
    return formats_mean_std, min_token_format


if __name__=="__main__":
    from argparse import ArgumentParser
    args = ArgumentParser()
    args.add_argument("--task_name", type=str, default="QASC")
    args.add_argument("--prompting_strategy", type=str, default="zero-shot")
    args.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct")
    args = args.parse_args()

    main(args)
    print("Done!")
    print(f"Task name: {args.task_name}")
    print(f"Prompting strategy: {args.prompting_strategy}")
    print(f"Model name: {args.model_name}")