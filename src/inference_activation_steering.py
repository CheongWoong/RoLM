import argparse
import os
import json
import jsonlines

from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pyvene as pv

import numpy as np
from collections import defaultdict

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

    predictions_path = os.path.join(output_dir, f"{prompting_strategy}_predictions.jsonl")
    raw_predictions_path = os.path.join(output_dir, f"{prompting_strategy}_raw_predictions.jsonl")
    try:
        with jsonlines.open(predictions_path) as fin:
            id_predictions_map, id_consistency_map = {}, {}
            for example in fin.iter():
                id_predictions_map[example["id"]] = example["predictions"]
                id_consistency_map[example["id"]] = example["consistency"]["mean"]
    except:
        pass

    if dataset_name == "CommonsenseQA":
        NUM_HALF = int(977/2)
    elif dataset_name == "QASC":
        NUM_HALF = int(741/2)
    elif dataset_name == "100TFQA":
        NUM_HALF = int(80/2)
    elif dataset_name == "GSM8K":
        NUM_HALF = int(1056/2)

    X, Y, Z = [], [], []
    id_majority_level_map = defaultdict(list)
    with jsonlines.open(raw_predictions_path) as fin:
        for idx, example in enumerate(fin.iter()):
            confidences = []
            for format_id, top_tokens in example["top_tokens"].items():
                confidence, flag = -0.01, False
                for i, top_tokenss in enumerate(top_tokens[::-1]):
                    if top_tokenss[0] == id_predictions_map[example["id"]][format_id]:
                        if "top_probs" in example:
                            confidence = example["top_probs"][format_id][-(i+1)][0]
                        else:
                            confidence = np.exp(example["top_logprobs"][format_id][-(i+1)][0])
                        confidences.append(confidence)
                        flag = True
                        break
                if not flag:
                    confidences.append(confidence)
                    print("could not find match")
            if len(confidences) != 8:
                # print(example["top_tokens"])
                print(len(confidences))
                # raise Exception
            else:
                id_predictions_map[example["id"]].pop('majority_voting')
                for ii in range(8):
                    zz = list(id_predictions_map[example["id"]].values()).count(id_predictions_map[example["id"]][str(ii)])
                    # zz = ((zz>7)*1.0) # binarize
                    # zz = max(8-zz, zz) # pairing
                    # zz = np.digitize(confidences[ii], np.array([0.0, 0.5, 0.9, 1.01])) # confidence
                    id_majority_level_map[example["id"]].append(zz)

    id_majority_level_map = list(id_majority_level_map.values())
    id_majority_level_map = np.array(id_majority_level_map)
    id_majority_level_map = id_majority_level_map.reshape(-1)

    layer_wise_path = os.path.join(output_dir, f"{prompting_strategy}_layer_wise_hidden_states.dat")
    head_wise_path = os.path.join(output_dir, f"{prompting_strategy}_head_wise_hidden_states.dat")

    input_dir = "preprocessed_datasets"
    input_path = os.path.join(input_dir, f"{dataset_name}_test_4_shot.jsonl")

    layer_wise_hidden_states = read_memmap(layer_wise_path)
    num_samples, num_formats, num_layers, hidden_size = layer_wise_hidden_states.shape

    X1 = layer_wise_hidden_states[NUM_HALF:,:,:,:].reshape(-1, num_layers, hidden_size)
    id_majority_level_map1 = id_majority_level_map[NUM_HALF*NUM_FORMATS:]
    X1_major = X1[id_majority_level_map1 > 7].mean(0)
    X1_minor = X1[id_majority_level_map1 < 5].mean(0)

    X2 = layer_wise_hidden_states[:NUM_HALF,:,:,:].reshape(-1, num_layers, hidden_size)
    id_majority_level_map2 = id_majority_level_map[:NUM_HALF*NUM_FORMATS]
    X2_major = X2[id_majority_level_map2 > 7].mean(0)
    X2_minor = X2[id_majority_level_map2 < 5].mean(0)

    #########################################################################################
    #########################################################################################
    #########################################################################################

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

    majority_direction1 = X1_major - X1_minor
    majority_direction2 = X2_major - X2_minor

    TARGET_LAYERS = [31]

    pv_config1, pv_config2 = [], []
    for layer_idx in range(num_layers):
        if layer_idx not in TARGET_LAYERS:
            continue
        intervener1 = ITI_Intervener(majority_direction1[layer_idx], alpha)
        pv_config1.append({
            "component": f"model.layers[{layer_idx}].output",
            "intervention": wrapper(intervener1),
        })
        intervener2 = ITI_Intervener(majority_direction2[layer_idx], alpha)
        pv_config2.append({
            "component": f"model.layers[{layer_idx}].output",
            "intervention": wrapper(intervener2),
        })
    intervened_model1 = pv.IntervenableModel(pv_config1, model)
    intervened_model2 = pv.IntervenableModel(pv_config2, model)

    #########################################################################################
    #########################################################################################
    #########################################################################################

    # Create output directory
    output_dir = f"results/{dataset_name}/{model_name}"
    output_path = os.path.join(output_dir, f"{prompting_strategy}_activation_steering_{alpha}_raw_predictions.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    input_dir = "preprocessed_datasets"
    input_path = os.path.join(input_dir, f"{dataset_name}_test_4_shot.jsonl")
    with jsonlines.open(input_path) as fin, open(output_path, "w") as fout:
        input_examples = ""
        for idx, example in tqdm(enumerate(fin.iter())):
            if idx == 0:
                intervened_model = intervened_model1
            elif idx == NUM_HALF:
                intervened_model = intervened_model2

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