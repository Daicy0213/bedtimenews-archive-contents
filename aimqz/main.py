import os
import sys
import yaml
import pynvml
import torch
from typing import Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
# 引入 datasets 库的核心功能
from datasets import load_dataset, DatasetDict

# ==========================================
# 1. 辅助函数
# ==========================================
def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def print_log(msg: str):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"\n[INFO] {msg}\n" + "-"*50)

def get_least_used_gpu():
    """获取当前使用量最小的GPU"""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    free_mem = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mem.append((info.free, i))
    # 按 free memory 降序，取第一个
    _, best_gpu = sorted(free_mem, reverse=True)[0]
    pynvml.nvmlShutdown()
    return best_gpu


# ==========================================
# 2. 数据预处理函数 (专为 map 设计)
# ==========================================
def process_func(example, tokenizer, max_len):
    """
    单条数据处理函数，将被 dataset.map 调用。
    """
    # 兼容 'messages' 或 'conversations' 字段
    messages = example.get("messages", example.get("conversations"))

    if messages is None:
        raise ValueError("数据样本中找不到 'messages' 或 'conversations' 字段")

    # Qwen2.5 对话模板处理
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding=False, # padding 交给 DataCollator 动态处理，节省空间
        add_special_tokens=True
    )

    # 构建 labels (Causal LM 任务，Labels = Input IDs)
    input_ids = tokenized["input_ids"]
    labels = list(input_ids) # 复制一份作为 labels

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

# ==========================================
# 3. 主训练流程
# ==========================================
def main():
    # --- 0. 加载配置 ---
    cfg = load_config("config.yaml")

    # --- 1. 加载 Tokenizer ---
    model_path = cfg["model"]["path"]
    print_log(f"加载 Tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. 加载与分割数据集 (使用 datasets 库) ---
    data_file = cfg["data"]["file"]
    print_log(f"正在加载数据集: {data_file}")

    # 加载本地 JSON 数据集
    # split="train" 意味着加载整个文件作为一个 Dataset 对象
    raw_dataset = load_dataset("json", data_files=data_file, split="train")

    # 验证数据列名，防止后续报错
    column_names = raw_dataset.column_names
    print_log(f"原始数据列名: {column_names}")

    # 使用 datasets 自带的 split 功能
    # 这会返回一个 DatasetDict，包含 'train' 和 'test'
    print_log(f"正在按比例 {cfg['data']['val_size']} 划分验证集...")
    dataset_dict = raw_dataset.train_test_split(test_size=cfg["data"]["val_size"], seed=42)

    print_log(f"训练集大小: {len(dataset_dict['train'])}, 验证集大小: {len(dataset_dict['test'])}")

    # --- 3. 数据映射与预处理 (Tokenization) ---
    print_log("正在进行 Tokenization (Map)...")

    # 使用偏函数或 lambda 将 tokenizer 和 max_len 传入 process_func
    # remove_columns 很重要，Tokenize 后要移除原始文本列('messages'等)，否则 Trainer 会报错
    tokenized_datasets = dataset_dict.map(
        lambda example: process_func(example, tokenizer, cfg["data"]["max_length"]),
        batched=False,
        remove_columns=column_names,
        num_proc=4, # 开启4个进程加速处理
        desc="Tokenizing dataset"
    )

    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["test"]

    # --- 4. 加载本地预训练模型 ---
    print_log("加载模型 (Flash Attention 2 + BF16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if cfg["training"]["bf16"] else torch.float16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )

    model.gradient_checkpointing_enable()

    # --- 5. 配置 LoRA ---
    print_log("配置 LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- 6. 训练参数设置 ---
    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        num_train_epochs=cfg["training"]["num_train_epochs"],
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=cfg["training"]["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["training"]["save_steps"],
        logging_steps=cfg["training"]["logging_steps"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        bf16=cfg["training"]["bf16"],
        fp16=cfg["training"]["fp16"],
        logging_dir=f"{cfg['training']['output_dir']}/logs",
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        group_by_length=True,
        # 新增：防止处理完的数据包含多余列
        remove_unused_columns=False
    )

    # --- 7. 开始训练 ---
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print_log("开始训练...")
    trainer.train()

    # --- 8. 保存模型 ---
    print_log(f"保存 LoRA 适配器至 {cfg['training']['output_dir']}")
    trainer.save_model(cfg["training"]["output_dir"])
    tokenizer.save_pretrained(cfg["training"]["output_dir"])

if __name__ == "__main__":
    main()