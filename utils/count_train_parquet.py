#!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd

ROOT = Path("/home/juyiang/code/Rag-Data-Generater/data")  # 修改为你的目录

TARGET_DIRS = [
    "ASearcher",
    "nq_hotpotqa_train_multi_2",
    "nq_hotpotqa_train_multi_4",
    "nq_hotpotqa_train_multi_6",
    "nq_hotpotqa_train_multi_8"
]

def count_parquet_rows(path: Path):
    try:
        df = pd.read_parquet(path)
        return len(df)
    except Exception as e:
        return f"ERROR: {e}"

def main():
    print("=== Counting train.parquet in datasets ===\n")
    for dname in TARGET_DIRS:
        dpath = ROOT / dname
        train_file = dpath / "train.parquet"

        if train_file.exists():
            count = count_parquet_rows(train_file)
            print(f"{dname:<30} -> {count} rows")
        else:
            print(f"{dname:<30} -> train.parquet NOT FOUND")

    print("\nDone.\n")

if __name__ == "__main__":
    main()
