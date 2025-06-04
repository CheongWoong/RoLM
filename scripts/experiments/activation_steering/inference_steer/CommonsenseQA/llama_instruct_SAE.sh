dataset_name=CommonsenseQA
model_name=Llama-3.1-8B-Instruct

alpha=25
max_new_tokens=256
# for ps in zero-shot few-shot zero-shot-cot
for ps in zero-shot
do
    python src/inference_SAE_steering.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps --max_new_tokens $max_new_tokens --alpha $alpha
done