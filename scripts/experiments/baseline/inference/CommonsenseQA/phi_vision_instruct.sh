dataset_name=CommonsenseQA
model_name=Phi-3.5-vision-instruct

max_new_tokens=256
for ps in zero-shot few-shot zero-shot-cot
do
    python src/inference.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps --max_new_tokens $max_new_tokens
done