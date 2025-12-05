#!/bin/bash

### ----------------------------
### Configuration
### ----------------------------
MODEL_PATH="/home/juyiang/data/llm_models/qwen25-32b-awq"
HOST="0.0.0.0"
PORT=8000

### ----------------------------
### GPU Selection
### ----------------------------
# 用两张卡部署（推荐）
export CUDA_VISIBLE_DEVICES=2,3

### ----------------------------
### Start vLLM Server (foreground)
### ----------------------------
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size 2 \
    --host $HOST \
    --port $PORT \
    --gpu-memory-utilization 0.95 \
    --max-model-len 12000 \
    --quantization awq_marlin \
    --dtype auto \
    --enforce-eager \
    --disable-log-requests
 