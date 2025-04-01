import argparse
import os

from tqdm.auto import tqdm
import jsonlines
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import NNsight
from einops import rearrange
import torch
import numpy as np

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
    model = NNsight(model)

    # Create output directory
    output_dir = f"results/{dataset_name}/{model_name}"
    layer_wise_path = os.path.join(output_dir, f"{prompting_strategy}_layer_wise_hidden_states.dat")
    head_wise_path = os.path.join(output_dir, f"{prompting_strategy}_head_wise_hidden_states.dat")
    os.makedirs(output_dir, exist_ok=True)

    input_dir = "preprocessed_datasets"
    input_path = os.path.join(input_dir, f"{dataset_name}_valid.jsonl")

    num_inputs = len(open(input_path, "r").readlines())
    with jsonlines.open(input_path) as fin:
        all_results = {
            "layer_wise_hidden_states_list": [],
            "head_wise_hidden_states_list": [],
        }
        layer_wise_shape = (num_inputs, NUM_FORMATS, model.config.num_hidden_layers, model.config.hidden_size)
        head_wise_shape = (num_inputs, NUM_FORMATS, model.config.num_hidden_layers, model.config.num_attention_heads, model.config.hidden_size // model.config.num_attention_heads)
        layer_wise_memmap = np.memmap(layer_wise_path, dtype="float16", mode="w+", shape=layer_wise_shape)
        head_wise_memmap = np.memmap(head_wise_path, dtype="float16", mode="w+", shape=head_wise_shape)
        for idx, example in tqdm(enumerate(fin.iter())):
            for format_type in [str(num) for num in range(NUM_FORMATS)]:
                # Format example
                formatted_example = format_example(example, dataset_name, prompting_strategy, format_type=format_type)

                # Chat templates for models
                if "Instruct" in model_name or "instruct" in model_name or "DeepSeek" in model_name:
                    messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": formatted_example}]
                    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
                else:
                    input_ids = tokenizer.encode(formatted_example, return_tensors="pt").to("cuda")

                # Generate the output
                with torch.no_grad():
                    with model.trace(input_ids) as tracer:
                        layer_wise_hidden_states = [model.model.layers[i].output[0].squeeze().detach().cpu().save() for i in range(model.config.num_hidden_layers)]
                        head_wise_hidden_states = [model.model.layers[i].self_attn.o_proj.input.squeeze().detach().cpu().save() for i in range(model.config.num_hidden_layers)]
                    layer_wise_hidden_states = torch.stack(layer_wise_hidden_states, dim=0).squeeze().numpy()[:,-1,:]
                    head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()[:,-1,:]
                    head_wise_hidden_states = rearrange(head_wise_hidden_states, 'l (h d) -> l h d', h=model.config.num_attention_heads)
                layer_wise_memmap[idx, int(format_type)] = layer_wise_hidden_states
                head_wise_memmap[idx, int(format_type)] = head_wise_hidden_states

        # Ensure data is written to disk
        layer_wise_memmap.flush()
        head_wise_memmap.flush()
        layer_wise_memmap_configs = {"shape": layer_wise_shape, "dtype": "float16"}
        head_wise_memmap_configs = {"shape": head_wise_shape, "dtype": "float16"}
        json.dump(layer_wise_memmap_configs, open(layer_wise_path.replace(".dat", ".conf"), "w"))
        json.dump(head_wise_memmap_configs, open(head_wise_path.replace(".dat", ".conf"), "w"))
        print("Processing complete. Hidden states saved using memmap.")