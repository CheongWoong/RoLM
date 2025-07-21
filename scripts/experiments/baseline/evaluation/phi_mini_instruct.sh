model_name=Phi-3.5-mini-instruct
for dataset_name in 100TFQA CommonsenseQA GSM8K QASC MMLU-Pro-Law-100Q
do
    for ps in zero-shot few-shot zero-shot-cot few-shot-cot
    do
        python src/evaluation.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps
    done
done