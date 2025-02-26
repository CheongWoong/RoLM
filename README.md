# RoLM
This is a repository for analyzing the robustness (semantic consistency) of LLMs.

## Installation
### Set up a Conda environment
This setup script creates an environment named "RoLM".
```
bash scripts/installation/setup_conda_env.sh
```

### Preprocess the datasets (Optional)
Run the following command to preprocess the datasets in "original_datasets/".  
The resulting dataset is saved in "preprocessed_datasets/".
```
# dataset_name: ["CommonsenseQA", "QASC", "100TFQA", "GSM8K"]
bash scripts/installation/data_preprocessing.sh {dataset_name}
```

## Inference
Run the following script to run inference.  
The raw predictions are saved in "results/{dataset_name}/{model_name}/{prompting_strategy}_raw_predictions.jsonl".
```
bash scripts/experiments/baseline/inference/{dataset_name}/{model_name}.sh
```

## Evaluation
Run the following command to run evaluation.  
The postprocessed predictions and score files are saved in "...predictions.jsonl" and "...score.json" in the same directory.
```
bash scripts/experiments/baseline/evaluation/{model_name}.sh
```
