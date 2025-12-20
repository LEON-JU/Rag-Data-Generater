import json
from pathlib import Path
from statistics import mean, median
from transformers import AutoTokenizer
from tqdm import tqdm

# 设置本地 tokenizer 路径
tokenizer = AutoTokenizer.from_pretrained(
    "/home/juyiang/data/llm_models/deepseek-r1-0528-qwen3-8b-AddTags2",
    trust_remote_code=True
)

# JSONL 数据路径
file_path = "/home/juyiang/data/dataset/sft_data/multi_agent/hotpotqa_fullwiki/summary/train.jsonl"

# 存储每条数据的 token 长度
token_lens = []

with open(file_path, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

# 读取并处理数据
with open(file_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, total=total_lines, desc="Processing samples"):
        data = json.loads(line)
        messages = data.get("messages", [])

        # 拼接所有 message 的 content
        full_text = "\n".join([m["content"] for m in messages if "content" in m])

        # 使用 tokenizer 编码并统计 token 数量
        tokens = tokenizer.encode(full_text, add_special_tokens=False)
        token_lens.append(len(tokens))

# 打印统计结果
print(f"📊 样本总数: {len(token_lens)}")
print(f"🔢 最大长度 (max): {max(token_lens)}")
print(f"🔍 最小长度 (min): {min(token_lens)}")
print(f"📈 平均长度 (mean): {mean(token_lens):.2f}")
print(f"📉 中位数 (median): {median(token_lens)}")
