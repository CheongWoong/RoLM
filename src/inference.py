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
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument('--save_logprobs', default=False, action='store_true')
    parser.add_argument('--majority_voting', default=False, action='store_true')
    args = parser.parse_args()
    model_name, dataset_name, prompting_strategy, max_new_tokens = args.model_name, args.dataset_name, args.prompting_strategy, args.max_new_tokens
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
    model.generation_config.temperature=None
    model.generation_config.top_p=None

    # Create output directory
    output_dir = f"results/{dataset_name}/{model_name}"
    if args.majority_voting:
        output_path = os.path.join(output_dir, f"{prompting_strategy}_majority_voting_raw_predictions.jsonl")
    else:
        output_path = os.path.join(output_dir, f"{prompting_strategy}_raw_predictions.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    input_dir = "preprocessed_datasets"
    input_path = os.path.join(input_dir, f"{dataset_name}_test_4_shot.jsonl")
    with jsonlines.open(input_path) as fin, open(output_path, "w") as fout:
        input_examples = ""
        for idx, example in tqdm(enumerate(fin.iter())):
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
                    if args.majority_voting:
                        output = model.generate(
                            input_ids,
                            max_new_tokens=max_new_tokens,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                            do_sample=True,
                            top_p=0.9,
                            num_return_sequences=10,
                            tokenizer=tokenizer,
                            stop_strings=["}```", "}\n```", "}\n\n```"],
                            return_dict_in_generate=True,
                        )
                    else:
                        output = model.generate(
                            input_ids,
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

                if args.save_logprobs and not args.majority_voting:
                    logits = torch.stack(output.scores, dim=1)[0]
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    top_probs, top_token_ids = torch.topk(probs, 5, dim=-1)
                    response_top_tokens, response_top_probs = [], []
                    for i in range(len(top_probs)):
                        top_tokens_list, top_probs_list = [], []
                        for j in range(len(top_probs[0])):
                            top_token = tokenizer.decode(top_token_ids[i][j])
                            top_tokens_list.append(top_token)
                            top_probs_list.append(top_probs[i][j].item())
                        response_top_tokens.append(top_tokens_list)
                        response_top_probs.append(top_probs_list)
                    result["top_tokens"][format_type] = response_top_tokens
                    result["top_probs"][format_type] = response_top_probs

                # Write to result
                result["raw_predictions"][format_type] = generated_texts

            # Write to file
            json.dump(result, fout)
            fout.write("\n")

            # Save a formatted input example
            if idx == 0:
                with open(os.path.join(output_dir, f"{prompting_strategy}_input_examples.jsonl"), "w") as fout_input_examples:
                    fout_input_examples.write(input_examples)