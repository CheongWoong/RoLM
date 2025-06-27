from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
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
def preprocess(example):
    messages = [
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]}
    ]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokenized = tokenizer(
        full_prompt,
        truncation=True,
        max_length=512 if "GSM" in DATASET_NAME else 256,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess)

# Training arguments
training_args = TrainingArguments(
    output_dir=f"src/REBUTTAL_emnlp/finetuned_{DATASET_NAME}",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# Start training
trainer.train()
