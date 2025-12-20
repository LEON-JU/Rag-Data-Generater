import json
from transformers import AutoTokenizer
from tqdm import tqdm

# 设置路径
input_path = "/home/juyiang/data/dataset/sft_data/multi_agent/twowikimultihopqa/summary/train.jsonl"
output_path = "/home/juyiang/data/dataset/sft_data/multi_agent/twowikimultihopqa/summary/train_filtered.jsonl"

# Token 限制
MAX_TOKENS = 8192

# 加载本地 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/home/juyiang/data/llm_models/deepseek-r1-0528-qwen3-8b",
    trust_remote_code=True,
    local_files_only=True
)

# 计数
total = 0
kept = 0
dropped = 0

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in tqdm(fin, desc="Filtering samples"):
        total += 1
        try:
            data = json.loads(line)
            messages = data.get("messages", [])
            full_text = "\n".join([m["content"] for m in messages if "content" in m])

            tokens = tokenizer.encode(full_text, add_special_tokens=False)

            if len(tokens) <= MAX_TOKENS:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1
        except Exception as e:
            print(f"[Error] Skipping line {total}: {e}")
            dropped += 1

# 结果统计
print("\n✅ 处理完成")
print(f"📦 总样本数: {total}")
print(f"✅ 保留样本: {kept}")
print(f"🗑️ 丢弃样本: {dropped}")
print(f"📄 输出文件: {output_path}")
