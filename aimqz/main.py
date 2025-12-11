import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
import os


# --- 1. 加载配置 ---
def load_config(path="./config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()


# --- 2. 读取数据 ---
print("Loading dataset...")
dataset = load_dataset("json", data_files=cfg['data_path'], split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=cfg['training']['seed'])
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")


# --- 3. 加载 tokenizer 和模型 ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(
    cfg["model_name"],
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    cfg["model_name"],
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

model = prepare_model_for_kbit_training(model)


# --- 4. LoRA 配置 ---
peft_config = LoraConfig(
    r=cfg['lora']['r'],
    lora_alpha=cfg['lora']['lora_alpha'],
    lora_dropout=cfg['lora']['lora_dropout'],
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=cfg['lora']['target_modules']
)


# --- 5. formatting_func: 将 messages[] 转成模型的标准输入 ---
def formatting_func(example):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return text


# --- 6. TrainingArguments ---
training_args = TrainingArguments(
    output_dir=cfg["output_dir"],
    num_train_epochs=cfg["training"]["num_train_epochs"],
    per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
    gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
    learning_rate=cfg["training"]["learning_rate"],
    logging_steps=cfg["training"]["logging_steps"],
    eval_strategy="steps",
    eval_steps=cfg["training"]["eval_steps"],
    save_strategy="steps",
    save_steps=cfg["training"]["save_steps"],
    bf16=True,
    optim="paged_adamw_32bit",
    report_to="none",
    ddp_find_unused_parameters=False,
)


# --- 7. SFTTrainer ---
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_func,
    args=training_args,
    max_length=cfg["training"]["max_seq_length"],
)


# --- 8. 训练 ---
trainer.train()


# --- 9. 保存 ---
trainer.save_model(cfg["output_dir"])
tokenizer.save_pretrained(cfg["output_dir"])

print("Done.")
