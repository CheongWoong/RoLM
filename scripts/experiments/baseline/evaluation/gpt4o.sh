model_name=gpt-4o-2024-11-20
for dataset_name in 100TFQA CommonsenseQA GSM8K QASC
do
    for ps in zero-shot few-shot zero-shot-cot few-shot-cot
    do
        python src/evaluation.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps
    done
done