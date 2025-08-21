import argparse
import os

from tqdm.auto import tqdm
import jsonlines
import json

from openai import OpenAI
import time

from src.utils.prompt_formatting import format_example


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-11-20")
    parser.add_argument("--dataset_name", type=str, default="CommonsenseQA")
    parser.add_argument("--prompting_strategy", type=str, default="zero-shot")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument('--save_logprobs', default=False, action='store_true')
    parser.add_argument('--majority_voting', default=False, action='store_true')
    args = parser.parse_args()
    model_name, dataset_name, prompting_strategy, max_new_tokens = args.model_name, args.dataset_name, args.prompting_strategy, args.max_new_tokens
    NUM_FORMATS = 8

    # Create output directory
    output_dir = f"results/{dataset_name}/{model_name}"
    if args.majority_voting:
        output_path = os.path.join(output_dir, f"{prompting_strategy}_majority_voting_raw_predictions.jsonl")
    else:
        output_path = os.path.join(output_dir, f"{prompting_strategy}_raw_predictions.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize OpenAI Client API
    with open("src/LIESLDKWEIHL", "r") as fin:
        OPENAI_API_KEY = "sk-" + fin.readline()

    client = OpenAI(
        api_key=OPENAI_API_KEY,
    )

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
                "top_logprobs": {}
            }

            for format_type in [str(num) for num in range(NUM_FORMATS)]:
                # Format example
                formatted_example = format_example(example, dataset_name, prompting_strategy, format_type=format_type)

                # Save a formatted input example
                if idx == 0:
                    input_examples += formatted_example + "\n\n"

                # Chat templates for OpenAI Chat API
                messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": formatted_example}]

                # Generate the output
                while True:
                    try:
                        if args.majority_voting:
                            responses = client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                max_tokens=max_new_tokens,
                                seed=0, # This does not work as system_fingerprint changes
                                top_p=0.9,
                                n=10,
                            )
                            break
                        else:
                            responses = client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                max_tokens=max_new_tokens,
                                temperature=0,
                                logprobs=True,
                                top_logprobs=5,
                            )
                            break
                    except Exception as e:
                        print('Error!', e)
                        time.sleep(3)

                # Fetch the output
                ### Fetch the output (top-k tokens at i-th position)
                if args.save_logprobs and not args.majority_voting:
                    response_top_tokens, response_top_logprobs = [], []
                    for response_logprob in responses.choices[0].logprobs.content:
                        top_tokens, top_logprobs = [], []
                        for top_logprob in response_logprob.top_logprobs:
                            top_tokens.append(top_logprob.token)
                            top_logprobs.append(top_logprob.logprob)
                        response_top_tokens.append(top_tokens)
                        response_top_logprobs.append(top_logprobs)
                    result["top_tokens"][format_type] = response_top_tokens
                    result["top_logprobs"][format_type] = response_top_logprobs

                # Write to result
                result["raw_predictions"][format_type] = [responses.choices[i].message.content for i in range(len(responses.choices))]

            # Write to file
            json.dump(result, fout)
            fout.write("\n")

            # Save a formatted input example
            if idx == 0:
                with open(os.path.join(output_dir, f"{prompting_strategy}_input_examples.jsonl"), "w") as fout_input_examples:
                    fout_input_examples.write(input_examples)