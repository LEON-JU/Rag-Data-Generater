export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export MASTER_PORT=12345

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

BASE_MODEL="/home/juyiang/data/llm_models/Qwen2.5-7B-Instruct-AddTags2"
MODEL_SINGLE_ABILITY_PATH="/home/juyiang/data/llm_models/Qwen2.5-7B-Instruct-SingleAbility"
DATA_DIR="/home/juyiang/data/dataset/sft_data/multi_agent"

swift sft \
  --model "${BASE_MODEL}" \
  --train_type lora \
  --dataset \
    "${DATA_DIR}/hotpotqa_fullwiki/search/train.jsonl" \
    "${DATA_DIR}/hotpotqa_fullwiki/reasoning/train.jsonl" \
    "${DATA_DIR}/hotpotqa_fullwiki/backtrack/train.jsonl" \
    "${DATA_DIR}/hotpotqa_fullwiki/summary/train_filtered.jsonl" \
    "${DATA_DIR}/twowikimultihopqa/search/train.jsonl" \
    "${DATA_DIR}/twowikimultihopqa/reasoning/train.jsonl" \
    "${DATA_DIR}/twowikimultihopqa/backtrack/train.jsonl" \
    "${DATA_DIR}/twowikimultihopqa/summary/train_filtered.jsonl" \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --max_length 8192 \
  --packing true \
  --eval_steps 200 \
  --save_steps 200 \
  --logging_steps 1 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 8 \
  --dataset_num_proc 4 \
  --save_total_limit 3 \
  --output_dir "${MODEL_SINGLE_ABILITY_PATH}" \
  --deepspeed zero2 \
  --attn_impl flash_attn \
  --use_liger_kernel true \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --model_type deepseek_r1_distill

# swift sft \
#     --model "${BASE_MODEL}" \
#     --train_type "full" \
#     --dataset \
#       "${DATA_DIR}/hotpotqa_fullwiki/search/train.jsonl" \
#       "${DATA_DIR}/hotpotqa_fullwiki/reasoning/train.jsonl" \
#       "${DATA_DIR}/hotpotqa_fullwiki/backtrack/train.jsonl" \
#       "${DATA_DIR}/hotpotqa_fullwiki/summary/train_filtered.jsonl" \
#       "${DATA_DIR}/twowikimultihopqa/search/train.jsonl" \
#       "${DATA_DIR}/twowikimultihopqa/reasoning/train.jsonl" \
#       "${DATA_DIR}/twowikimultihopqa/backtrack/train.jsonl" \
#       "${DATA_DIR}/twowikimultihopqa/summary/train_filtered.jsonl" \
#     --torch_dtype "bfloat16" \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 4 \
#     --learning_rate 5e-5 \
#     --gradient_accumulation_steps 4 \
#     --packing true \
#     --eval_steps 50 \
#     --save_steps 50 \
#     --logging_steps 1 \
#     --max_length 8192 \
#     --warmup_ratio 0.05 \
#     --dataloader_num_workers 8 \
#     --dataset_num_proc 4 \
#     --save_total_limit 3 \
#     --response_prefix "" \
#     --save_only_model false \
#     --output_dir "${MODEL_SINGLE_ABILITY_PATH}" \
#     --deepspeed "zero3" \
#     --use_liger_kernel true \
#     --attn_impl "flash_attn" \
#     --model_type "deepseek_r1_distill"