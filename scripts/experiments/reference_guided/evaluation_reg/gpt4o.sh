model_name=gpt-4o-2024-11-20
for dataset_name in QASC
do
    for ps in zero-shot-reference-guided
    do
        python src/evaluation.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps
    done
done