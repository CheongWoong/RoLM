dataset_name=QASC
model_name=Llama-3.1-8B-Instruct

max_new_tokens=256
for ps in zero-shot few-shot zero-shot-cot few-shot-cot
do
    python src/inference_activation_steering.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps --max_new_tokens $max_new_tokens
done