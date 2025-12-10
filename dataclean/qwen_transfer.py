import os
from pathlib import Path

from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

PROMPT_SYSTEM = """你是一个专业的数据标注助手。
你的任务是把一段文本按照问答关系拆成 user 和 assistant。
要求：
- 识别马前卒(也叫任冲昊/马督工)和提问者
- user 为明确问题（通常以问号结尾, 或者被<font color = "indigo"></font>包裹, 连续的多行和多个标签应合并为一个问题）
- assistant 为紧随其后的马前卒所说的内容
- 输出 JSON 数组，对话的格式为：
[{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]},...]
- 请遵循多轮对话数据集的原则, 将相同主题的内容合并到同一个message对象中, 不同主题的另外创建一个message
- 注意保持换行的一致性, 输出的内容统一使用两个换行符号
- 不要自行编造内容，只基于文本识别问答关系。
- 可以删除多余的标记和符号, 如网页地址/图片地址等等
"""

target_file = Path("dataset1.jsonl")
total_tokens = 0
# 逐个读取文件
for i in range(20, 20):
    with open(f"../dataclean/cleaned/{i}.txt", "r", encoding="utf-8") as f:
        content = f.read()

    # 发送请求
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen3-max",
        messages=[
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": content},
        ]
    )
    resp_content = completion.choices[0].message.content
    tokens = completion.usage.total_tokens  # 累加token
    total_tokens += tokens

    # 格式化为新的内容
    prefix = f"// ------------------------------------ {i} -------------------------------------\n"
    resp_content = prefix + resp_content.replace("```json", "")[:-3]
    print(resp_content)

    with target_file.open("a", encoding="utf-8") as f:  # 续写
        f.write(resp_content + "\n")

print("\ntotal_tokens:", total_tokens)  # 打印最终消耗的token
