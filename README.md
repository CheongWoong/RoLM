# RoLM
This is the original implementation of [When Format Changes Meaning: Investigating Semantic Inconsistency of Large Language Models]() (EMNLP 2025 Findings).

## Installation
### Set up a Conda environment
This setup script creates an environment named "RoLM".
```
bash scripts/installation/setup_conda_env.sh
```

## Inference
Run the following script to run inference.  
The raw predictions are saved in "results/{dataset_name}/{model_name}/{prompting_strategy}_raw_predictions.jsonl".
```
# dataset_name: ["CommonsenseQA", "QASC", "100TFQA", "GSM8K", "MMLU-Pro-Law-100Q"]
bash scripts/experiments/baseline/inference/{dataset_name}/{model_name}.sh
```

## Evaluation
Run the following command to run evaluation.  
The postprocessed predictions and score files are saved in "...predictions.jsonl" and "...score.json" in the same directory.
```
bash scripts/experiments/baseline/evaluation/{model_name}.sh
```

## Analysis
Analysis codes are provided [here](https://github.com/CheongWoong/RoLM/tree/main/src/analysis).