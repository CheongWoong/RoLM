dataset_name=MMLU-Pro-Law-100Q
model_name=Phi-3.5-mini-instruct

max_new_tokens=512
for ps in zero-shot few-shot zero-shot-cot few-shot-cot
do
    python src/inference.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps --max_new_tokens $max_new_tokens
done