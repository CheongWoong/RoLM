dataset_name=QASC
model_name=gpt-4o-2024-11-20

max_new_tokens=256
for ps in zero-shot
do
    python src/extended_formats/inference_openai_ext.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps --max_new_tokens $max_new_tokens --save_logprobs
done