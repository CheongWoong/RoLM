dataset_name=CommonsenseQA
model_name=gpt-4o-2024-11-20

max_new_tokens=256
for ps in zero-shot few-shot zero-shot-cot
do
    python src/inference_openai.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps --max_new_tokens $max_new_tokens --save_logprobs
done