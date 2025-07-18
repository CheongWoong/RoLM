dataset_name=mmlu_pro_law_q100
model_name=Llama-3.1-8B-Instruct

max_new_tokens=512
for ps in zero-shot few-shot zero-shot-cot few-shot-cot
do
    python src/inference.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps --max_new_tokens $max_new_tokens --save_logprobs
done