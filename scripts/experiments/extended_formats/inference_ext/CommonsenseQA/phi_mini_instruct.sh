dataset_name=CommonsenseQA
model_name=Phi-3.5-mini-instruct

max_new_tokens=256
for ps in zero-shot
do
    python src/extended_formats/inference_ext.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps --max_new_tokens $max_new_tokens --save_logprobs
done