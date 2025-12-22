#!/bin/bash

### ----------------------------
### 路径配置
### ----------------------------
BASE_MODEL="/home/juyiang/data/llm_models/deepseek-r1-0528-qwen3-8b-AddTags2"
LORA_NAME="deepseek_lora"
LORA_PATH="/home/juyiang/data/llm_models/deepseek-r1-0528-qwen3-8b-SingleAbility/v0-20251219-150338/checkpoint-1263"
HOST="0.0.0.0"
PORT=8000

### ----------------------------
### 显卡设置（单卡）
### ----------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3

### ----------------------------
### 启动 vLLM 服务
### ----------------------------
python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 15000 \
    --dtype bfloat16 \
    --disable-log-requests \
    --host "$HOST" \
    --port "$PORT" \
    --enforce-eager \
    --max-lora-rank 64 \
    --enable-lora \
    --lora-modules "${LORA_NAME}=${LORA_PATH}" \
