model_name=Llama-3.1-8B-Instruct
# for dataset_name in 100TFQA CommonsenseQA GSM8K QASC
for dataset_name in CommonsenseQA
do
    # for ps in zero-shot_activation_steering_5.0 few-shot_activation_steering_5.0 zero-shot-cot_activation_steering_5.0 few-shot-cot_activation_steering_5.0
    for ps in zero-shot_activation_steering_5.0 zero-shot-cot_activation_steering_5.0
    do
        python src/evaluation.py --dataset_name $dataset_name --model_name $model_name --prompting_strategy $ps
    done
done