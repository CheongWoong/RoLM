from tqdm.auto import tqdm
import jsonlines
import os

input_dir = "original_datasets"


def preprocess_data(dataset_name):
    if dataset_name == "CommonsenseQA":
        return preprocess_CommonsenseQA()
    elif dataset_name == "QASC":
        return preprocess_QASC()
    elif dataset_name == "100TFQA":
        return preprocess_100TFQA()
    elif dataset_name == "GSM8K":
        return preprocess_GSM8K()
    else:
        raise NotImplementedError

def preprocess_CommonsenseQA():
    input_path = os.path.join(input_dir, "CommonsenseQA_dev.jsonl")

    with jsonlines.open(input_path) as fin:
        examples = []
        for row in tqdm(fin.iter()):
            questions = {"original": row["question"]["stem"].strip()}

            example = {}
            example["id"] = row["id"]
            example["answer"] = row["answerKey"].strip()
            example["questions"] = questions
            example["options"] = row["question"]["choices"]
            example["question_concept"] = row["question"]["question_concept"].strip()

            examples.append(example)
    return examples

def preprocess_QASC():
    input_path = os.path.join(input_dir, "QASC_dev.jsonl")

    with jsonlines.open(input_path) as fin:
        examples = []
        for row in tqdm(fin.iter()):
            questions = {"original": row["question"]["stem"].strip()}

            example = {}
            example["id"] = row["id"]
            example["answer"] = row["answerKey"].strip()
            example["questions"] = questions
            example["options"] = row["question"]["choices"]
            example["fact1"] = fact1 = row["fact1"].strip()
            example["fact2"] = fact2 = row["fact2"].strip()
            example["facts"] = facts = " ".join([fact1, fact2])
            example["combinedfact"] = combinedfact = row["combinedfact"].strip() + "."
            example["explanation"] = " ".join([facts, combinedfact])

            examples.append(example)
    return examples

def preprocess_100TFQA():
    input_path = os.path.join(input_dir, "100TFQA.jsonl")

    with jsonlines.open(input_path) as fin:
        examples = []
        for row in tqdm(fin.iter()):
            statements = {"original": row["statement"].strip()}

            example = {}
            example["id"] = row["id"]
            example["answer"] = row["answer"].strip()
            example["statements"] = statements
            example["explanation"] = row["explanation"].strip()

            examples.append(example)
    return examples

def preprocess_GSM8K():
    input_path = os.path.join(input_dir, "GSM8K_test.jsonl")

    with jsonlines.open(input_path) as fin:
        examples = []
        for idx, row in tqdm(enumerate(fin.iter())):
            questions = {"original": row["question"].strip()}
            explanation, answer = row["answer"].split("####")

            example = {}
            example["id"] = idx
            example["answer"] = answer.strip()
            example["questions"] = questions
            example["explanation"] = explanation.strip()

            examples.append(example)
    return examples