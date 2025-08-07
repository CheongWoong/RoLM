import argparse
import os

from tqdm.auto import tqdm
import jsonlines
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from src.utils.prompt_formatting import format_example
from src.utils.model_map import MODEL_MAP


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="CommonsenseQA")
    parser.add_argument("--prompting_strategy", type=str, default="zero-shot")
    args = parser.parse_args()
    model_name, dataset_name, prompting_strategy = args.model_name, args.dataset_name, args.prompting_strategy
    NUM_FORMATS = 8

    input_dir = "preprocessed_datasets"
    input_path = os.path.join(input_dir, f"{dataset_name}_valid.jsonl")
    output_path = input_path.replace("valid.jsonl", "train_single_token.jsonl")
    with jsonlines.open(input_path) as fin, open(output_path, "w") as fout:
        input_examples = ""
        for idx, example in tqdm(enumerate(fin.iter())):
            for format_type in [str(num) for num in range(NUM_FORMATS)]:
                # Format example
                formatted_example = format_example(example, dataset_name, prompting_strategy, format_type=format_type, disable_prefilling=False)
                first_idx = formatted_example.find("```json")
                sep_idx = formatted_example.find("```json", first_idx+1)
                inp, out = formatted_example, example["answer"]

                if inp[-1] == " ":
                    inp = inp[:-1]
                    out = " " + out
                    
                # out += "```"

                json.dump({"input": inp, "output": out}, fout)
                fout.write("\n")
