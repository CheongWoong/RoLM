model_name=Phi-3.5-mini-instruct
for dataset_name in CommonsenseQA QASC
do
    for ps in zero-shot
    do
        python src/extended_formats/get_activations_ext.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps
    done
done