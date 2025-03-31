for model_name in Falcon3-1B-Instruct Falcon3-3B-Instruct Falcon3-7B-Instruct
do
    for dataset_name in CommonsenseQA QASC
    do
        for ps in zero-shot few-shot zero-shot-cot few-shot-cot
        do
            python src/evaluation.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps
        done
    done
done