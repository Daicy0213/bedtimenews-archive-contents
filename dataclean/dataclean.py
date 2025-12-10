import re
from pathlib import Path


def extract_start_number(folder_name: str) -> int:
    """从 '1-100' 提取起始编号 1"""
    match = re.match(r'^(\d+)-\d+$', folder_name)
    return int(match.group(1)) if match else float('inf')


def clean_text(text: str) -> str:
    """清除 HTML、Markdown、多余空白等，但保留 <font color="indigo">...</font> 及其内容"""
    # 删除 front matter（YAML 块）
    text = re.sub(r'^---[\s\S]*?---\s*', '', text, count=1, flags=re.MULTILINE)

    # 删除图片链接
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # 删除标题行：如 "# Tabs {.tabset}", "## B站", "# " 等
    text = re.sub(r'^#{1,6}\s*.*$', '', text, flags=re.MULTILINE)

    # 清理空白行和首尾空白
    text = re.sub(r'\n\s*\n', '\n\n', text)  # 保留段落间隔
    text = re.sub(r'\n{3,}', '\n\n', text)  # 最多两个换行

    # 找出所有 <font color="indigo">...</font> 块，并替换为占位符（保留原始内容）
    indigo_blocks = []

    def preserve_indigo(match):
        indigo_blocks.append(match.group(0))  # 保存完整标签（含内容）
        return f"__INDIGO_BLOCK_{len(indigo_blocks) - 1}__"

    # 匹配：支持单引号、双引号、大小写、空格
    pattern = r'<font\s+color\s*=\s*["\']?\s*indigo\s*["\']?\s*>(.*?)</font>'
    text = re.sub(pattern, preserve_indigo, text, flags=re.IGNORECASE | re.DOTALL)

    # 删除其他 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)

    # 删除引用块（> ...）
    text = re.sub(r'^> .*', '', text, flags=re.MULTILINE)

    # 删除带 {.xxx} 的行（如 {.is-warning}）
    text = re.sub(r'^\{\.[-\w]+\}\s*$', '', text, flags=re.MULTILINE)

    # 删除--- 和 -->等
    text = re.sub(r'^--[-|>]', '', text, flags=re.MULTILINE)

    # 再次清理空行
    text = re.sub(r'\n\s*\n', '\n\n', text)  # 保留段落间隔
    text = re.sub(r'\n{3,}', '\n\n', text)  # 最多两个换行
    text = text.strip()

    # 还原 <font color="indigo"> 块
    for i, block in enumerate(indigo_blocks):
        text = text.replace(f"__INDIGO_BLOCK_{i}__", block)

    return text


def read_all_md_files_in_order(main_dir: str):
    main_path = Path(main_dir)
    target_folder = Path("./cleaned")
    target_folder.mkdir(parents=True, exist_ok=True)

    # 获取所有子文件夹，并按起始编号排序
    subfolders = [f for f in main_path.iterdir() if f.is_dir() and re.match(r"\d+.*", f.name)]
    subfolders.sort(key=lambda x: extract_start_number(x.name))

    # 遍历每个子文件夹
    for folder in subfolders:
        print(f"Processing folder: {folder.name}")

        # 获取所有 .md 文件
        md_files = list(folder.glob("*.md"))

        # 按文件名中的数字排序（自然排序）
        def get_file_number(md_file: Path) -> int:
            # 提取纯数字部分，如 "5.md" → 5, "100.md" → 100
            name = md_file.stem  # 去掉 .md
            return int(name) if name.isdigit() else float('inf')

        md_files.sort(key=get_file_number)

        # 顺序读取每个文件
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                print(f"  Reading: {md_file.name}")  # 打印文件名
                content = f.read()
                content = clean_text(content)  # 清理数据

                output_file = target_folder / md_file.with_suffix(".txt").name
                output_file.write_text(content, encoding='utf-8')
                print("数据已保存")


if __name__ == '__main__':
    main_dir = "../main"
    read_all_md_files_in_order(main_dir)
