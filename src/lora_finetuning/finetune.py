from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch
import argparse
import jsonlines


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="CommonsenseQA")
args = parser.parse_args()
DATASET_NAME = args.dataset_name

# Model and tokenizer
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Required for padding

# Dummy dataset
# raw_data = [
#     {"input": "What is the capital of France?", "output": "Paris"},
#     {"input": "Translate 'hello' to Spanish.", "output": "Hola"},
# ]
raw_data = []
with jsonlines.open(f"preprocessed_datasets/{DATASET_NAME}_train.jsonl") as fin:
    for example in fin.iter():
        raw_data.append(example)

dataset = Dataset.from_list(raw_data)

# Load model in fp16 (no quantization)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA configuration
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# Preprocess function using chat template
def formatting_func(example):
    assert len(example['input']) == len(example['output'])

    output_texts = []
    for i in range(len(example['input'])):
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": example["input"][i]},
            {"role": "assistant", "content": example["output"][i]}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        output_texts.append(full_prompt)
    return output_texts

training_args = SFTConfig(
    output_dir=f"src/REBUTTAL_emnlp/finetuned_{DATASET_NAME}",
    fp16=True,
    max_length=512 if "GSM" in DATASET_NAME else 256,
    per_device_train_batch_size=4 if "GSM" in DATASET_NAME else 8,
    gradient_accumulation_steps=2 if "GSM" in DATASET_NAME else 1,
    num_train_epochs=10 if "100TFQA" in DATASET_NAME else 3,
    save_strategy="epoch",
    report_to="none"
)

# Trainer
response_template = "<|start_header_id|>assistant<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_func,
    data_collator=collator,
    peft_config=peft_config,
)

# Start training
trainer.train()
