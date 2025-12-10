import os
from pathlib import Path

from openai import OpenAI, APIError

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

target_file = Path("bedtimenews.jsonl")
total_tokens = 0

# 截止2025年12月10日的最高播放量视频
top_popular = [83, 94, 338, 158, 50, 682, 185, 282, 139, 685, 568, 159, 112, 672, 417, 68, 85, 77, 81, 106, 541, 95,
               100, 309, 141, 308, 71, 72, 555, 843, 182, 312, 190, 646, 658, 355, 169, 40, 671, 186, 69, 145, 655,
               125, 510, 147, 51, 883, 285, 167, 176, 97, 90, 596, 93, 78, 144, 45, 129, 114, 65, 429, 122, 842,
               152, 37, 131, 180, 149, 98, 92, 116, 218, 151, 422, 110, 113, 519, 170, 155, 846, 27, 275, 328, 385,
               117, 52, 30, 595, 134, 146, 73, 192, 88, 194, 187, 137, 680, 143, 108]
top_popular.sort()
print(top_popular)

if __name__ == '__main__':
    # 打印当前批次需要处理的文本编号0-50，80-100
    start = 50
    end = 80
    curr_list = [top_popular[i] for i in range(start, end)]
    print(f"当前批次处理的期数： {curr_list}")

    # 逐个读取文件
    for i in range(start, end):
        video_id = top_popular[i]
        with open(f"../dataclean/cleaned/{video_id}.txt", "r", encoding="utf-8") as f:
            content = f.read()

        prefix = f"// ------------------------------------ {video_id} -------------------------------------\n"
        # 发送请求
        try:
            completion = client.chat.completions.create(
                # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                model="qwen3-max",
                messages=[
                    {"role": "system", "content": PROMPT_SYSTEM},
                    {"role": "user", "content": content},
                ]
            )
        except APIError as e:
            print(f"跳过文件 {video_id}.txt（APIError: {e}）")
            with target_file.open("a", encoding="utf-8") as f:  # 如果错误，则仅标记
                f.write(prefix)
            continue
        except Exception as e:
            print(f"跳过文件 {video_id}.txt（未知错误: {e}）")
            continue

        resp_content = completion.choices[0].message.content
        tokens = completion.usage.total_tokens  # 累加token
        total_tokens += tokens

        # 格式化为新的内容
        resp_content = prefix + resp_content.replace("```json", "").replace("```", "")
        print(resp_content)

        with target_file.open("a", encoding="utf-8") as f:  # 续写
            f.write(resp_content + "\n")

    print("\ntotal_tokens:", total_tokens)  # 打印最终消耗的token
