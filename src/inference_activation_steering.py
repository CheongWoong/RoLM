import argparse
import os
import json
import jsonlines

from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pyvene as pv
from einops import rearrange

import numpy as np
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.utils.prompt_formatting import format_example
from src.utils.model_map import MODEL_MAP

########################################################################
# Reference: https://github.com/likenneth/honest_llama/tree/master
def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped

class Collector():
    collect_state = True
    collect_action = False  
    def __init__(self, multiplier, head):
        self.head = head
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s): 
        if self.head == -1:
            self.states.append(b[0, -1].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
        else:
            self.states.append(b[0, -1].reshape(32, -1)[self.head].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
        return b
    
class ITI_Intervener():
    collect_state = True
    collect_action = True
    attr_idx = -1
    def __init__(self, direction, multiplier):
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction)
        self.direction = direction.cuda().half()
        self.multiplier = multiplier
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s): 
        self.states.append(b[0, -1].detach().clone())  # original b is (batch_size=1, seq_len, #head x D_head), now it's (#head x D_head)
        action = self.direction.to(b.device)
        self.actions.append(action.detach().clone())
        b[0, -1] = b[0, -1] + action * self.multiplier
        return b
########################################################################

def read_memmap(filepath):
    with open(filepath.replace(".dat", ".conf"), "r") as fin_config:
        memmap_configs = json.load(fin_config)
        return np.memmap(filepath, mode="r", shape=tuple(memmap_configs["shape"]), dtype=memmap_configs["dtype"])


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="CommonsenseQA")
    parser.add_argument("--prompting_strategy", type=str, default="zero-shot")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=5.0)
    args = parser.parse_args()
    model_name, dataset_name, prompting_strategy, max_new_tokens, alpha = args.model_name, args.dataset_name, args.prompting_strategy, args.max_new_tokens, args.alpha
    NUM_FORMATS = 8

    # Create output directory
    output_dir = f"results/{dataset_name}/{model_name}"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[model_name])
    if "Phi-3.5-vision" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_MAP[model_name],
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            _attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_MAP[model_name],
            device_map="auto",
            torch_dtype=torch.float16,
        )
    model.eval()
    model.generation_config.temperature=None
    model.generation_config.top_p=None

    top_k_heads_path = os.path.join(output_dir, f"{prompting_strategy}_activation_steering_top_k_heads.json")
    with open(top_k_heads_path) as fin:
        top_k_heads = json.load(fin)
    top_k_head_indices = []
    for idx, probing_accuracy in top_k_heads:
        top_k_head_indices.append(idx)
    print("Top-K heads to be steered:")
    print(top_k_head_indices)
    print()

    target_steering_direction = np.zeros((model.config.num_hidden_layers, model.config.hidden_size))
    target_steering_direction = rearrange(target_steering_direction, 'l (h d) -> l h d', h=model.config.num_attention_heads)
    steering_direction_path = os.path.join(output_dir, f"{prompting_strategy}_activation_steering_direction.npy")
    steering_direction = np.load(steering_direction_path)

    pv_config = []
    for layer_idx, head_idx in top_k_head_indices:
        target_steering_direction[layer_idx][head_idx] += steering_direction[layer_idx][head_idx]
    target_steering_direction = rearrange(target_steering_direction, 'l h d -> l (h d)')
    for layer_idx in range(model.config.num_hidden_layers):
        intervener = ITI_Intervener(target_steering_direction[layer_idx], alpha)
        pv_config.append({
            "component": f"model.layers[{layer_idx}].self_attn.o_proj.input",
            "intervention": wrapper(intervener),
        })

    intervened_model = pv.IntervenableModel(pv_config, model)

    # Create output directory
    output_dir = f"results/{dataset_name}/{model_name}"
    output_path = os.path.join(output_dir, f"{prompting_strategy}_activation_steering_{alpha}_raw_predictions.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    input_dir = "preprocessed_datasets"
    input_path = os.path.join(input_dir, f"{dataset_name}_test_4_shot.jsonl")
    with jsonlines.open(input_path) as fin, open(output_path, "w") as fout:
        input_examples = ""
        for idx, example in tqdm(enumerate(fin.iter())):
            # if idx >= 100:
            #     break

            # Prepare result
            result = {
                "id": example["id"],
                "raw_predictions": {},
                "top_tokens": {},
                "top_probs": {}
            }

            for format_type in [str(num) for num in range(NUM_FORMATS)]:
                # Format example
                formatted_example = format_example(example, dataset_name, prompting_strategy, format_type=format_type)

                # Save a formatted input example
                if idx == 0:
                    input_examples += formatted_example + "\n\n"

                # Chat templates for models
                if "Instruct" in model_name or "instruct" in model_name or "DeepSeek" in model_name:
                    messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": formatted_example}]
                    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
                else:
                    input_ids = tokenizer.encode(formatted_example, return_tensors="pt").to("cuda")

                # Generate the output
                with torch.no_grad():
                    # output = model.generate(
                    _, output = intervened_model.generate(
                        {"input_ids": input_ids},
                        max_new_tokens=max_new_tokens,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                        do_sample=False,
                        temperature=0,
                        tokenizer=tokenizer,
                        stop_strings=["}```", "}\n```", "}\n\n```"],
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                # Decode the output
                input_length = len(input_ids[0])
                generated_texts = [tokenizer.decode(output_ids[input_length:], skip_special_tokens=True) for output_ids in output.sequences]

                # Write to result
                result["raw_predictions"][format_type] = generated_texts

            # Write to file
            json.dump(result, fout)
            fout.write("\n")