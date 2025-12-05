#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import json

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


FILE = Path("/home/juyiang/code/Rag-Data-Generater/data/ASearcher/test_nq_rand1000.parquet")

def pretty(obj):
    """Pretty print JSON with indentation."""
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)

def main():
    if not FILE.exists():
        print(f"File not found: {FILE}")
        return

    print(f"Loading dataset: {FILE}")
    df = pd.read_parquet(FILE)

    print(f"\nTotal rows: {len(df)}")
    print("\n=== Displaying FIRST ROW ===\n")

    row = df.iloc[0]

    # Print all columns raw
    print("📌 All Columns:")
    for col in df.columns:
        print(f" - {col}")

    print("\n📌 Raw Row Content:")
    print(row)

    # Pretty visualization for structured fields
    print("\n=== Pretty Visualization ===\n")

    for field in ["question", "answer", "paths", "passages", "documents", "context"]:
        if field in row:
            print(f"\n🔹 {field.upper()}:\n")
            print(pretty(row[field]))

    print("\nDone.\n")

if __name__ == "__main__":
    main()
