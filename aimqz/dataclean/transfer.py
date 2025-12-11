import json

input_file = "./bedtimenews.json"  # 原始 JSON 数组文件
output_file = "./bedtimenews.jsonl"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_file, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")