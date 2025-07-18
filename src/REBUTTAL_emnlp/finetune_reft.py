from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, AutoConfig
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model
import pyreft
from datasets import Dataset
import torch
import argparse
import jsonlines

import transformers

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="CommonsenseQA")
args = parser.parse_args()
DATASET_NAME = args.dataset_name

# Model and tokenizer
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, model_max_length=512 if "GSM" in DATASET_NAME else 256)
tokenizer.pad_token = tokenizer.eos_token  # Required for padding

# Dummy dataset
# raw_data = [
#     {"input": "What is the capital of France?", "output": "Paris"},
#     {"input": "Translate 'hello' to Spanish.", "output": "Hola"},
# ]
inputs, outputs = [], []
with jsonlines.open(f"preprocessed_datasets/{DATASET_NAME}_train.jsonl") as fin:
    for example in fin.iter():
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        sep_idx = full_prompt.find("<|start_header_id|>assistant<|end_header_id|>")
        sep_len = len("<|start_header_id|>assistant<|end_header_id|>")
        inputs.append(full_prompt[:sep_idx+sep_len])
        outputs.append(full_prompt[sep_idx+sep_len:])

# Load model in fp16 (no quantization)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # torch_dtype=torch.float16,
    torch_dtype="auto",
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

reft_config = pyreft.ReftConfig(representations=[{
    # "layer": l, "component": "block_output",
    # string component access is enforced for customized model such as a peft model!
    "layer": l, "component": f"base_model.model.model.layers[{l}].output",
    "low_rank_dimension": 64,
    "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
    low_rank_dimension=64)} for l in range(model.config.num_hidden_layers)])
reft_model = pyreft.get_reft_model(model, reft_config)
reft_model.set_device("cuda")
# you need to call this to re-enable lora grads!
reft_model.model.enable_adapter_layers()
reft_model.print_trainable_parameters()

# train enables dropout but no grads.
# this line might not be necessary since HF trainer enables this by default.
reft_model.model.train()
n_params = reft_model.count_parameters(include_model=False)
n_params_with_model = reft_model.count_parameters(include_model=True)

data_module = pyreft.make_multiple_position_supervised_data_module(
    tokenizer, model, inputs, outputs, "f7+l7", len(reft_model.interventions)*2,
    nonstop=True 
)

training_args = transformers.TrainingArguments(
    output_dir=f"src/REBUTTAL_emnlp/finetuned_reft_{DATASET_NAME}",
    # fp16=True,
    bf16=True,
    per_device_train_batch_size=4 if "GSM" in DATASET_NAME else 8,
    gradient_accumulation_steps=2 if "GSM" in DATASET_NAME else 1,
    num_train_epochs=10 if "100TFQA" in DATASET_NAME else 3,
    save_strategy="epoch",
    logging_steps=1,
    report_to="none",
    learning_rate=5e-5,
)
trainer = pyreft.ReftTrainerForCausalLM(model=reft_model, tokenizer=tokenizer, args=training_args, **data_module)

# Start training
trainer.train()
