import argparse
import os

from tqdm.auto import tqdm
import jsonlines
import json
from collections import defaultdict, Counter
import numpy as np

from src.utils.output_postprocessing import parse_output


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="CommonsenseQA")
    parser.add_argument("--prompting_strategy", type=str, default="zero-shot")
    args = parser.parse_args()
    model_name, dataset_name, prompting_strategy = args.model_name, args.dataset_name, args.prompting_strategy
    NUM_FORMAT_ELEMENTS = 5
    NUM_FORMATS = pow(2, NUM_FORMAT_ELEMENTS)

    # Prepare id2answer map
    input_path = f"preprocessed_datasets/{dataset_name}_valid.jsonl"
    with jsonlines.open(input_path) as fin:
        id_answer_map = {}
        for example in fin.iter():
            if dataset_name == "GSM8K":
                id_answer_map[example["id"]] = example["answer"].replace(",", "")
            else:
                id_answer_map[example["id"]] = example["answer"]

    output_dir = f"results/{dataset_name}/{model_name}"
    output_predictions_path = os.path.join(output_dir, f"{prompting_strategy}_ext_predictions_validation.jsonl")
    output_score_path = os.path.join(output_dir, f"{prompting_strategy}_ext_score_validation.json")
    raw_predictions_path = os.path.join(output_dir, f"{prompting_strategy}_ext_raw_predictions_validation.jsonl")
    total_count, parse_error_count = 0, defaultdict(int)
    try:
        with jsonlines.open(raw_predictions_path) as fin:
            with open(output_predictions_path, "w") as fout_predictions, open(output_score_path, "w") as fout_score:
                accuracy_list, consistency_list = defaultdict(list), defaultdict(list)
                # Read each example of raw predictions
                for example in tqdm(fin.iter()):
                    # Fetch answer for evaluation
                    answer = id_answer_map[example["id"]]

                    # Prepare result
                    result = {
                        "id": example["id"],
                        "predictions": {},
                        "accuracy": {},
                        "consistency": {},
                    }

                    # Output parsing for each format
                    for format_type in [str(num) for num in range(NUM_FORMATS)]:
                        raw_predictions = example["raw_predictions"][format_type]
                        if type(raw_predictions) != list:
                            raw_predictions = [raw_predictions]
                        predictions = []
                        for raw_prediction in raw_predictions:
                            ### Parse output and compute accuracy
                            prediction, is_parsed_ok = parse_output(raw_prediction, dataset_name)
                            predictions.append(prediction)
                            if not is_parsed_ok:
                                parse_error_count[example["id"]] = 1
                        if len(predictions) > 1:
                            prediction_counts = Counter(predictions)
                            prediction, count = prediction_counts.most_common(1)[0]
                        accuracy = (answer == prediction)*1.0
                        ### Write to result
                        result["predictions"][format_type] = prediction
                        result["accuracy"][format_type] = accuracy
                        ### Write to global statistics
                        accuracy_list[format_type].append(accuracy)
                    total_count += 1

                    ### Make lists of predictions and accuracy scores for an example
                    example_predictions = list(result["predictions"].values())
                    example_accuracy = list(result["accuracy"].values())
                    ### Compute mean/min/max accuracy and write to result
                    result["accuracy"]["mean"] = np.mean(example_accuracy)
                    result["accuracy"]["min"] = np.min(example_accuracy)
                    result["accuracy"]["max"] = np.max(example_accuracy)
                    ### Write to global statistics
                    accuracy_list["mean"].append(result["accuracy"]["mean"])
                    accuracy_list["min"].append(result["accuracy"]["min"])
                    accuracy_list["max"].append(result["accuracy"]["max"])

                    # Majority voting
                    ### Compute majority voting prediction and accuracy
                    prediction_counts = Counter(example_predictions)
                    majority_prediction, count = prediction_counts.most_common(1)[0]
                    majority_voting_accuracy = (answer == majority_prediction)*1.0
                    ### Write to result
                    result["predictions"]["majority_voting"] = majority_prediction
                    result["accuracy"]["majority_voting"] = majority_voting_accuracy
                    ### Write to global statistics
                    accuracy_list["majority_voting"].append(majority_voting_accuracy)

                    # Measure consistency score
                    example_consistency = []
                    for i in range(len(example_predictions)):
                        for j in range(i+1, len(example_predictions)):
                            ### Compute consistency
                            consistency = (example_predictions[i] == example_predictions[j])*1.0
                            ### Save consistency scores for an example
                            example_consistency.append(consistency)
                            ### Write to global statistics
                            consistency_list[f"{i}_{j}"].append(consistency)
                    ### Compute mean consistency and write to result
                    result["consistency"]["mean"] = np.mean(example_consistency)
                    ### Write to global statistics
                    consistency_list["mean"].append(result["consistency"]["mean"])

                    # Write result to file
                    json.dump(result, fout_predictions)
                    fout_predictions.write("\n")

                # Aggregate global statistics and write to file
                aggregated_result = {"accuracy": {}, "consistency": {}}
                for key, value in accuracy_list.items():
                    aggregated_result["accuracy"][key] = round(np.mean(value), 5)
                for key, value in consistency_list.items():
                    aggregated_result["consistency"][key] = round(np.mean(value), 5)
                json.dump(aggregated_result, fout_score, indent=4)
                print(model_name, dataset_name, prompting_strategy)
                print("Error / Total:", int(np.sum(list(parse_error_count.values()))), "/", total_count)
    except Exception as e:
        print(e)