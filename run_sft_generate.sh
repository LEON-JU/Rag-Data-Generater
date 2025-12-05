#!/usr/bin/env bash
set -euo pipefail

HOT_POT_DEFAULT=(
  /home/juyiang/data/hotpotqa/fullwiki/train-00000-of-00002.parquet
  /home/juyiang/data/hotpotqa/fullwiki/train-00001-of-00002.parquet
)

TWOWIKI_DEFAULT=/home/juyiang/data/2wikimultihopqa/train.parquet

run_dataset() {
  local dataset_name=$1
  shift
  python examples/multi_agent_langchain/multi_agent_sft_generate.py \
    --config configs/multi_agent_langchain.yaml \
    --output-dir sft_data/multi_agent/${dataset_name} \
    --split train \
    --num-workers 8 \
    --dataset "${dataset_name}" \
    "$@"
}

run_dataset hotpotqa_fullwiki \
  --dataset-files "${HOT_POT_DEFAULT[@]}"

run_dataset twowikimultihopqa \
  --dataset-files "${TWOWIKI_DEFAULT}"
