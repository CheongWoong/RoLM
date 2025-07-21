model_name=Llama-3.1-70B-Instruct
for dataset_name in CommonsenseQA QASC 100TFQA GSM8K MMLU-Pro-Law-100Q
do
    for ps in zero-shot few-shot zero-shot-cot few-shot-cot
    do
        python src/evaluation.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps
    done
done