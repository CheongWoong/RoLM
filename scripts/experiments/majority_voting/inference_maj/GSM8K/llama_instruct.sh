dataset_name=GSM8K
model_name=Llama-3.1-8B-Instruct

max_new_tokens=512
for ps in zero-shot-cot few-shot-cot
do
    python src/inference.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps --max_new_tokens $max_new_tokens --majority_voting
done