model_name=Llama-3.1-8B-Instruct
for dataset_name in 100TFQA QASC MMLU-Pro-Law-100Q
do
    for ps in zero-shot few-shot zero-shot-cot few-shot-cot
    do
        python src/get_activations.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps
    done
done
for dataset_name in CommonsenseQA
do
    for ps in zero-shot few-shot zero-shot-cot
    do
        python src/get_activations.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps
    done
done
for dataset_name in GSM8K
do
    for ps in zero-shot-cot few-shot-cot
    do
        python src/get_activations.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps
    done
done