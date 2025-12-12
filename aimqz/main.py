# train_dual_4090_flash.py
import os
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

accelerator = Accelerator()

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_cartoken = tokenizer.eos_token

    # ===== 启用 FlashAttention-2 =====
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        dtype=torch.bfloat16,
        device_map={"": accelerator.local_process_index},
        trust_remote_code=True,
        use_cache=False,
        attn_implementation="flash_attention_2",  # ✅ 核心启用
    )
    # ===============================

    peft_config = LoraConfig(**config["lora"])
    model = get_peft_model(model, peft_config)

    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # 数据加载等后续步骤完全不变...
    dataset = load_dataset("json", data_files=config["data"]["dataset_path"], split="train")

    def format_conversation(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(
        format_conversation,
        batched=False,
        num_proc=config["data"]["num_proc"]
    )
    dataset = dataset.train_test_split(test_size=1 - config["data"]["train_ratio"])
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    training_args = TrainingArguments(
        **config["training"],
        fp16=False,
        tf32=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ddp_find_unused_parameters=False,
    )

    class DualCardTrainer(Trainer):
        def log(self, logs):
            if accelerator.is_main_process:
                if "loss" in logs or "eval_loss" in logs:
                    super().log(logs)

    trainer = DualCardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            padding=True,
            max_length=config["training"]["max_seq_length"],
            return_tensors="pt"
        ),
    )

    if accelerator.is_main_process:
        print("🚀 Starting LoRA + FlashAttention-2 training on dual RTX 4090...")

    trainer.train()

    if accelerator.is_main_process:
        final_path = os.path.join(config["training"]["output_dir"], "final_model_flash")
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"✅ Training completed with FlashAttention-2. Model saved to {final_path}")

if __name__ == "__main__":
    main()