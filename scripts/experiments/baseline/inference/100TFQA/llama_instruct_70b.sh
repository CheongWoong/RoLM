dataset_name=100TFQA
model_name=Llama-3.1-70B-Instruct

max_new_tokens=256
for ps in zero-shot few-shot zero-shot-cot few-shot-cot
do
    python src/inference.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps --max_new_tokens $max_new_tokens --save_logprobs
done