import argparse
import os
import json
import jsonlines

from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pyvene as pv
from einops import rearrange

from sae_lens import SAE, HookedSAETransformer

import numpy as np
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.utils.prompt_formatting import format_example
from src.utils.model_map import MODEL_MAP

########################################################################
from functools import partial

def steering(
    activations, hook, steering_strength=1.0, sae_vector=None, steering_vector=None
):
    direction = torch.matmul(steering_vector, sae_vector)
    modified = activations.clone()
    modified[:, -1, :] += steering_strength*direction
    return modified
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
        model = HookedSAETransformer.from_pretrained_no_processing(MODEL_MAP[model_name], dtype=torch.bfloat16, device="cuda")
    model.eval()


    sae_idx = 15

    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="llama_scope_lxr_8x",  # <- Release name
        sae_id=f"l{sae_idx}r_8x",  # <- SAE id (not always a hook point!)
        device="cuda",
    )
    sae.use_error_term = True

    # model.generation_config.temperature=None
    # model.generation_config.top_p=None

    steering_direction_path = os.path.join(output_dir, f"{prompting_strategy}_SAE_steering_direction_{sae_idx}.npy")
    steering_direction = np.load(steering_direction_path)
    steering_vector = torch.from_numpy(steering_direction).bfloat16().to("cuda")

    # Create output directory
    output_dir = f"results/{dataset_name}/{model_name}"
    output_path = os.path.join(output_dir, f"{prompting_strategy}_SAE_steering_{sae_idx}_{alpha}_raw_predictions.jsonl")
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

                # input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)

                steering_hook = partial(
                    steering,
                    sae_vector=sae.W_dec,
                    steering_vector=steering_vector,
                    steering_strength=alpha,
                )

                # Generate the output
                # with torch.no_grad():
                with model.hooks(fwd_hooks=[(sae.cfg.hook_name, steering_hook)]):                
                    output = model.generate(
                    # _, output = intervened_model.generate(
                        # {"input_ids": input_ids},
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                        temperature=0,
                        prepend_bos=False,
                    )

                # Decode the output
                input_length = len(input_ids[0])
                output_ids = output[0]
                generated_texts = [tokenizer.decode(output_ids[input_length:], skip_special_tokens=True)]

                # Write to result
                result["raw_predictions"][format_type] = generated_texts

            # Write to file
            json.dump(result, fout)
            fout.write("\n")