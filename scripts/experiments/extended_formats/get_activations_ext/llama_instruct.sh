model_name=Llama-3.1-8B-Instruct
for dataset_name in CommonsenseQA QASC
do
    for ps in zero-shot
    do
        python src/extended_formats/get_activations_ext.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps
    done
done