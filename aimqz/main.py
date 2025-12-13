import os

import pynvml
import torch
import yaml
# 引入 datasets 库的核心功能
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training  # 引入 QLoRA 模型准备函数
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig  # 引入量化配置
)


# ==========================================
# 1. 辅助函数
# ==========================================
def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def print_log(msg: str):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"\n[INFO] {msg}\n" + "-" * 50)


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
        padding=False,  # padding 交给 DataCollator 动态处理，节省空间
        add_special_tokens=True
    )

    # 构建 labels (Causal LM 任务，Labels = Input IDs)
    input_ids = tokenized["input_ids"]
    labels = list(input_ids)  # 复制一份作为 labels

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
    cfg = load_config("config_q8.yaml")

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
        num_proc=4,  # 开启4个进程加速处理
        desc="Tokenizing dataset"
    )

    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["test"]

    # --- 4. 加载本地预训练模型 ---
    print_log("创建 BitsAndBytes 量化配置...")
    quant_cfg = cfg["quantization"]

    if quant_cfg.get("use_8bit", False):
        # 使用8Bit量化模型
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,  # 仅启用 8-bit
        )
        print_log("加载模型 (QLoRA 8-bit 量化)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # 移除 torch_dtype=... 因为量化配置会接管 dtype
            attn_implementation="flash_attention_2",
            device_map=None,  # DDP 模式下仍保持 None，但加载时会使用 QLoRA 的特殊逻辑
            quantization_config=bnb_config,  # 传入量化配置
            trust_remote_code=True
        )

        # 必须使用 prepare_model_for_kbit_training 包装模型，以处理量化模型中的 LayerNorm
        # 默认启用 gradient_checkpointing
        model = prepare_model_for_kbit_training(model)
    elif quant_cfg["use_4bit"]:
        # 使用4Bit量化模型
        compute_dtype = getattr(torch, quant_cfg["bnb_4bit_compute_dtype"])

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_cfg["use_4bit"],
            bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        )

        print_log("加载模型 (QLoRA 4-bit 量化)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # 移除 torch_dtype=... 因为量化配置会接管 dtype
            attn_implementation="flash_attention_2",
            device_map=None,  # DDP 模式下仍保持 None，但加载时会使用 QLoRA 的特殊逻辑
            quantization_config=bnb_config,  # 传入量化配置
            trust_remote_code=True
        )
        # 必须使用 prepare_model_for_kbit_training 包装模型，以处理量化模型中的 LayerNorm
        # 默认启用 gradient_checkpointing
        model = prepare_model_for_kbit_training(model)
    else:
        print_log("加载模型 (Flash Attention 2 + BF16)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if cfg["training"]["bf16"] else torch.float16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )

        model.gradient_checkpointing_enable()  # 启用 gradient_checkpointing

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
