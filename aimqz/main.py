import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os

# --- 1. 加载配置 ---
config_path = "./config.yaml"
def load_config(path="./config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

cfg = load_config()

# --- 2. 准备数据集 ---
# 假设你的数据是 JSON 格式的 List[Dict]
print("Loading dataset...")
dataset = load_dataset("json", data_files=cfg['data_path'], split="train")

# 划分训练集和验证集 (90% 训练, 10% 验证)
dataset = dataset.train_test_split(test_size=0.1, seed=cfg['training']['seed'])
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")

# --- 3. 模型与分词器 ---
print("Loading model and tokenizer...")

# 量化配置 (4-bit QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # 4090 支持 bf16
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    cfg['model_name'],
    trust_remote_code=True
)
# Qwen 的 pad token 处理
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    cfg['model_name'],
    quantization_config=bnb_config,
    device_map="auto", # accelerate 会自动处理多卡分配
    trust_remote_code=True,
    attn_implementation="flash_attention_2" # 4090 强力推荐开启
)

# 准备模型进行 k-bit 训练
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

# --- 5. 训练参数 ---
training_args = TrainingArguments(
    output_dir=cfg['output_dir'],
    num_train_epochs=cfg['training']['num_train_epochs'],
    per_device_train_batch_size=cfg['training']['per_device_train_batch_size'],
    per_device_eval_batch_size=cfg['training']['per_device_eval_batch_size'],
    gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
    learning_rate=cfg['training']['learning_rate'],
    logging_steps=cfg['training']['logging_steps'],
    # 评估策略
    eval_strategy="steps",
    eval_steps=cfg['training']['eval_steps'],
    save_strategy="steps",
    save_steps=cfg['training']['save_steps'],
    # 优化与精度
    bf16=True, # 4090 必须开启 bf16
    optim="paged_adamw_32bit",
    report_to="none", # 如果想用 wandb，改成 "wandb"
    ddp_find_unused_parameters=False, # 多卡训练需要设置为 False 以避免报错
)

# --- 6. SFT Trainer ---
# 使用 TRL 的 SFTTrainer，它会自动应用 Chat Template
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=cfg['training']['max_seq_length'],
    dataset_text_field="messages", # 指向你 json 中的 key，TRL 会自动处理 messages 格式
)

# --- 7. 开始训练 ---
print("Starting training...")
trainer.train()

# --- 8. 保存模型 ---
print("Saving model...")
trainer.save_model(cfg['output_dir'])
tokenizer.save_pretrained(cfg['output_dir'])
print(f"Done! Model saved to {cfg['output_dir']}")