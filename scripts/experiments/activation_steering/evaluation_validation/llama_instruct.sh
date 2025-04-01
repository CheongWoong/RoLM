model_name=Llama-3.1-8B-Instruct
# for dataset_name in 100TFQA CommonsenseQA GSM8K QASC
for dataset_name in CommonsenseQA
do
    # for ps in zero-shot few-shot zero-shot-cot few-shot-cot
    for ps in zero-shot zero-shot-cot
    do
        python src/evaluation_validation.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps
    done
done