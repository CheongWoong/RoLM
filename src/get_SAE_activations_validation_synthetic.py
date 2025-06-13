import argparse
import os

from tqdm.auto import tqdm
import jsonlines
import json
from copy import deepcopy
import itertools
from collections import Counter, defaultdict

from transformers import AutoTokenizer, AutoModelForCausalLM
from einops import rearrange
import torch
import numpy as np

from sae_lens import SAE, HookedSAETransformer

from src.utils.prompt_formatting import format_example
from src.utils.model_map import MODEL_MAP


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
    args = parser.parse_args()
    model_name, dataset_name, prompting_strategy = args.model_name, args.dataset_name, args.prompting_strategy
    NUM_FORMATS = 8

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[model_name])
    if "Phi-3.5-vision" in model_name:
        # model = AutoModelForCausalLM.from_pretrained(
        #     MODEL_MAP[model_name],
        #     device_map="auto",
        #     torch_dtype=torch.float16,
        #     trust_remote_code=True,
        #     _attn_implementation="eager",
        # )
        model = None
    else:
        # model = AutoModelForCausalLM.from_pretrained(
        #     MODEL_MAP[model_name],
        #     device_map="auto",
        #     torch_dtype=torch.float16,
        # )
        model = HookedSAETransformer.from_pretrained_no_processing(MODEL_MAP[model_name], dtype=torch.float16, device="cuda")
    model.eval()

    #####
    if dataset_name == "CommonsenseQA":
        options = ["A", "B", "C", "D", "E"]
    elif dataset_name == "QASC":
        options = ["A", "B", "C", "D", "E", "F", "G", "H"]
    elif dataset_name == "100TFQA":
        options = ["True", "False"]
    elif dataset_name == "GSM8K":
        options = []
        # need to implement loading options from the dataset (collect answers in the validation set and use that as options)
        raise Exception
    else:
        raise Exception
    #####

    # Create output directory
    output_dir = f"results/{dataset_name}/{model_name}"
    layer_wise_path = os.path.join(output_dir, f"{prompting_strategy}_layer_wise_sae_hidden_states_validation_synthetic.dat")
    # head_wise_path = os.path.join(output_dir, f"{prompting_strategy}_head_wise_hidden_states_validation.dat")
    os.makedirs(output_dir, exist_ok=True)

    input_dir = "preprocessed_datasets"
    input_path = os.path.join(input_dir, f"{dataset_name}_valid.jsonl")
    output_predictions_path = os.path.join(output_dir, f"{prompting_strategy}_predictions_validation.jsonl")
    # output_predictions_path = os.path.join(output_dir, f"{prompting_strategy}_predictions.jsonl")

    np.random.seed(0)

    majority_num = 4
    minority_num = 8 - majority_num
    new_predictions = []
    counter1, counter2 = defaultdict(float), defaultdict(float)
    majority_preds = []
    all_pairs, all_idx_pairs = [], []
    with jsonlines.open(output_predictions_path) as fin, open(output_predictions_path.replace(".jsonl", "_synthetic.jsonl"), "w") as fout:
        for example in fin.iter():
            majority_pred = example["predictions"]["majority_voting"]
            majority_preds.append(majority_pred)

            minority_preds = deepcopy(options)
            minority_preds.remove(majority_pred)

            majority_list = [majority_pred for _ in range(majority_num)]
            minority_list = np.random.choice(minority_preds, minority_num, replace=True).tolist()
            # pairs = list(itertools.product(majority_list, minority_list))
            # all_pairs += pairs
            # for pair in pairs:
            #     counter1[pair] += 1

            majority_idx_list = np.random.choice(8, majority_num, replace=False)
            minority_idx_list = [i for i in range(8) if i not in majority_idx_list]
            # idx_pairs = list(itertools.product(majority_idx_list, minority_idx_list))
            # all_idx_pairs += idx_pairs
            # for idx_pair in idx_pairs:
            #     counter2[idx_pair] += 1

            new_example = {}
            new_example["id"] = example["id"]
            new_example["predictions"] = {}
            for format_idx in range(NUM_FORMATS):
                if format_idx in majority_idx_list:
                    new_example["predictions"][str(format_idx)] = majority_pred
                else:
                    new_example["predictions"][str(format_idx)] = minority_list.pop(0)
            new_predictions.append(list(new_example["predictions"].values()))
            new_example["predictions"]["majority_voting"] = majority_pred
            json.dump(new_example, fout)
            fout.write("\n")

    # print(counter1)
    # print(counter2)
    # print()

    # assert len(all_pairs) == len(all_idx_pairs)
    # N = len(all_pairs)

    # # Unique values
    # unique_answer_pairs = list(set(all_pairs))
    # unique_format_pairs = list(set(all_idx_pairs))

    # # Mapping from pair to indices
    # answer_to_indices = defaultdict(list)
    # format_to_indices = defaultdict(list)

    # for i, (a_pair, f_pair) in enumerate(zip(all_pairs, all_idx_pairs)):
    #     answer_to_indices[a_pair].append(i)
    #     format_to_indices[f_pair].append(i)

    # # Initialize weights
    # weights = np.ones(N)

    # # Target weight per group (uniform marginals)
    # target_answer_weight = len(all_pairs) / len(unique_answer_pairs)
    # target_format_weight = len(all_pairs) / len(unique_format_pairs)

    # # Iterative proportional fitting
    # for iteration in range(20):  # usually converges quickly
    #     # Normalize by answer_pair groups
    #     for a_pair in unique_answer_pairs:
    #         indices = answer_to_indices[a_pair]
    #         total = weights[indices].sum()
    #         if total > 0:
    #             weights[indices] *= target_answer_weight / total

    #     # Normalize by format_pair groups
    #     for f_pair in unique_format_pairs:
    #         indices = format_to_indices[f_pair]
    #         total = weights[indices].sum()
    #         if total > 0:
    #             weights[indices] *= target_format_weight / total

    # counter1, counter2 = defaultdict(float), defaultdict(float)
    # for idx, weight in enumerate(weights):
    #     counter1[all_pairs[idx]] += weight
    #     counter2[all_idx_pairs[idx]] += weight
    #     print(weight)
    # print(counter1)
    # print(counter2)
    # print()

    num_inputs = len(open(input_path, "r").readlines())

    layer_wise_shape = (num_inputs, NUM_FORMATS, 32, 4096*8)
    # head_wise_shape = (num_inputs, NUM_FORMATS, model.config.num_hidden_layers, model.config.num_attention_heads, model.config.hidden_size // model.config.num_attention_heads)
    layer_wise_memmap = np.memmap(layer_wise_path, dtype="float16", mode="w+", shape=layer_wise_shape)
    # head_wise_memmap = np.memmap(head_wise_path, dtype="float16", mode="w+", shape=head_wise_shape)

    with jsonlines.open(input_path) as fin:
        sae_idxs = range(0, 16)
        saes = []
        for sae_idx in sae_idxs:
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release="llama_scope_lxr_8x",  # <- Release name
                sae_id=f"l{sae_idx}r_8x",  # <- SAE id (not always a hook point!)
                device="cuda",
            )
            sae.use_error_term = True
            saes.append(sae)
        for idx, example in tqdm(enumerate(fin.iter())):
            for format_type in [str(num) for num in range(NUM_FORMATS)]:
                # Format example
                formatted_example = format_example(example, dataset_name, prompting_strategy, format_type=format_type, disable_prefilling=False)
                formatted_example += '"'
                formatted_example += new_predictions[idx][int(format_type)]

                # Chat templates for models
                if "Instruct" in model_name or "instruct" in model_name or "DeepSeek" in model_name:
                    messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": formatted_example}]
                    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
                else:
                    input_ids = tokenizer.encode(formatted_example, return_tensors="pt").to("cuda")

                # Generate the output
                with torch.no_grad():
                    _, cache = model.run_with_cache_with_saes(input_ids, saes=saes)

                for sae_idx in sae_idxs:
                    layer_wise_hidden_states = [cache[f"blocks.{sae_idx}.hook_resid_post.hook_sae_acts_post"][0, -1, :].float().cpu()]
                    layer_wise_hidden_states = torch.stack(layer_wise_hidden_states, dim=0).squeeze().numpy()
                    layer_wise_memmap[idx, int(format_type), sae_idx] = layer_wise_hidden_states
                    # head_wise_memmap[idx, int(format_type)] = head_wise_hidden_states

        for sae in saes:
            del sae
        torch.cuda.empty_cache()

    with jsonlines.open(input_path) as fin:
        sae_idxs = range(16, 32)
        saes = []
        for sae_idx in sae_idxs:
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release="llama_scope_lxr_8x",  # <- Release name
                sae_id=f"l{sae_idx}r_8x",  # <- SAE id (not always a hook point!)
                device="cuda",
            )
            sae.use_error_term = True
            saes.append(sae)
        for idx, example in tqdm(enumerate(fin.iter())):
            for format_type in [str(num) for num in range(NUM_FORMATS)]:
                # Format example
                formatted_example = format_example(example, dataset_name, prompting_strategy, format_type=format_type, disable_prefilling=False)

                # Chat templates for models
                if "Instruct" in model_name or "instruct" in model_name or "DeepSeek" in model_name:
                    messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": formatted_example}]
                    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
                else:
                    input_ids = tokenizer.encode(formatted_example, return_tensors="pt").to("cuda")

                # Generate the output
                with torch.no_grad():
                    _, cache = model.run_with_cache_with_saes(input_ids, saes=saes)

                for sae_idx in sae_idxs:
                    layer_wise_hidden_states = [cache[f"blocks.{sae_idx}.hook_resid_post.hook_sae_acts_post"][0, -1, :].float().cpu()]
                    layer_wise_hidden_states = torch.stack(layer_wise_hidden_states, dim=0).squeeze().numpy()
                    layer_wise_memmap[idx, int(format_type), sae_idx] = layer_wise_hidden_states
                    # head_wise_memmap[idx, int(format_type)] = head_wise_hidden_states

    # Ensure data is written to disk
    layer_wise_memmap.flush()
    # head_wise_memmap.flush()
    layer_wise_memmap_configs = {"shape": layer_wise_shape, "dtype": "float16"}
    # head_wise_memmap_configs = {"shape": head_wise_shape, "dtype": "float16"}
    json.dump(layer_wise_memmap_configs, open(layer_wise_path.replace(".dat", ".conf"), "w"))
    # json.dump(head_wise_memmap_configs, open(head_wise_path.replace(".dat", ".conf"), "w"))
    print("Processing complete. Hidden states saved using memmap.")