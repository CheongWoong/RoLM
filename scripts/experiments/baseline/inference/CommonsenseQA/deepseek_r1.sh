dataset_name=CommonsenseQA
model_name=DeepSeek-R1-Distill-Llama-8B

max_new_tokens=1024
for ps in zero-shot-cot
do
    python src/inference.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps --max_new_tokens $max_new_tokens
done