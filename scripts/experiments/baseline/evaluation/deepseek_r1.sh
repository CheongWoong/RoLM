model_name=DeepSeek-R1-Distill-Llama-8B
for dataset_name in 100TFQA CommonsenseQA GSM8K QASC
do
    for ps in zero-shot-cot few-shot-cot
    do
        python src/evaluation.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps
    done
done