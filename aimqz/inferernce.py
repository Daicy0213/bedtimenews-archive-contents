import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 配置参数 ---
# 原始预训练模型路径
BASE_MODEL_PATH = "./Qwen2.5-7B-Instruct"
# 训练保存的 LoRA 适配器路径 (与 config.yaml 中的 output_dir 对应)
LORA_ADAPTER_PATH = "./output_qwen_lora"

# --- 1. 加载 Tokenizer ---
print("🚀 正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- 2. 加载基座模型 ---
# 注意：推理通常只用单卡，所以使用 device_map="auto" 或指定具体的 "cuda:0"
# 如果你在训练时使用了 BF16，这里也建议使用 BF16 来加载，以保持精度。
print(f"🧠 正在加载基座模型 ({BASE_MODEL_PATH})...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,  # 确保与训练时的 dtype 一致 (如果是 4090 且训练时使用了 bf16)
    attn_implementation="flash_attention_2",  # 推理时也使用 Flash Attention 2 加速
    device_map="auto",  # 自动分配到 GPU 上
    trust_remote_code=True
)

# --- 3. 加载 LoRA 适配器 ---
print(f"🧩 正在加载 LoRA 适配器 ({LORA_ADAPTER_PATH})...")
model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)

# --- 4. 合并 LoRA 权重到基座模型 (可选但推荐) ---
# 合并权重后，可以移除 PeftModel 包装，提高推理速度并节省显存。
print("✨ 正在合并 LoRA 权重...")
# 必须将模型切换到 eval 模式
model.eval()
model = model.merge_and_unload()  # 合并并卸载 LoRA 结构


# --- 5. 对话推理循环 ---
def chat_loop():
    print("\n" + "=" * 50)
    print("开始对话 (输入 'exit' 退出, 'clear' 清空历史)")
    print("=" * 50)

    history = []

    while True:
        try:
            user_input = input("我: ")
            if user_input.lower() in ['exit', 'quit']:
                print("对话结束。")
                break
            if user_input.lower() == 'clear':
                history = []
                print("历史记录已清空。")
                continue

            # 构造对话历史
            history.append({"role": "user", "content": user_input})

            # 使用 tokenizer.apply_chat_template 准备输入
            input_text = tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True  # 这一步很重要，告诉模型接下来应该生成 assistant 的回复
            )

            # Tokenize 和移至设备
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            # 确保 input_ids 在与模型相同的设备上 (通常是 CUDA)
            input_ids = input_ids.to(model.device)

            # 生成回复
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            # 解码回复，跳过输入部分
            response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

            print(f"马督工: {response}")
            print("\n")

            # 将模型的回复添加到历史记录中
            history.append({"role": "assistant", "content": response})

        except EOFError:
            break
        except Exception as e:
            print(f"发生错误: {e}")
            break


if __name__ == "__main__":
    chat_loop()
