import argparse
import os

import jsonlines
import json
import random
from sklearn.model_selection import train_test_split

from src.utils.data_preprocessing import preprocess_data


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="CommonsenseQA")
    parser.add_argument("--num_demonstrations", type=int, default=4)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    k = args.num_demonstrations

    # Create output directory
    output_dir = "preprocessed_datasets"
    os.makedirs(output_dir, exist_ok=True)

    # Preprocess data
    examples = preprocess_data(dataset_name)

    # Validation-test split (2:8)
    valid, test = train_test_split(examples, train_size=0.2, random_state=0)
    print(len(valid), len(test), len(examples))
    path_valid = os.path.join(output_dir, f"{dataset_name}_valid.jsonl")
    path_test = os.path.join(output_dir, f"{dataset_name}_test.jsonl")
    with open(path_valid, "w") as fout_valid, open(path_test, "w") as fout_test:
        for example in valid:
            json.dump(example, fout_valid)
            fout_valid.write("\n")
        for example in test:
            json.dump(example, fout_test)
            fout_test.write("\n")

    # Generate few-shot dataset
    demonstrations = []
    with jsonlines.open(path_valid) as fin:
        for example in fin.iter():
            demonstrations.append(example)
    path_test_k_shot = os.path.join(output_dir, f"{dataset_name}_test_{k}_shot.jsonl")
    with open(path_test_k_shot, "w") as fout, jsonlines.open(path_test) as fin:
        random.seed(0)
        for example in fin.iter():
            demonstrations = random.sample(demonstrations, k=k)
            example["demonstrations"] = demonstrations
            json.dump(example, fout)
            fout.write("\n")