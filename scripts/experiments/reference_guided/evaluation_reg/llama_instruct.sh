model_name=Llama-3.1-8B-Instruct
for dataset_name in QASC
do
    for ps in zero-shot-reference-guided
    do
        python src/evaluation.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps
    done
done