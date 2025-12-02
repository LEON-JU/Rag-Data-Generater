#!/usr/bin/env python3
"""
Quick interactive smoke test:

1. Pick one QA item (default: ASearcher hotpot subset)
2. Build the system prompt with tool descriptions
3. Run the interruption loop against the local vLLM/OpenAI-compatible endpoint
4. Stream out assistant thoughts, tool calls, retrieved observations, and final answer
5. Print the gold answer for manual inspection

Environment requirements:
    - GENERATOR_LLM_API_KEY / GENERATOR_LLM_BASE_URL pointing to your vLLM server
    - ELASTIC_PASSWORD or ELASTIC_SEARCH_PASSWORD for the wiki backend
"""

from __future__ import annotations

import argparse
import json
from typing import List
import sys
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent   # 获取 ../
sys.path.insert(0, str(ROOT))                  # 加入 PYTHONPATH

from rag_data_generator.datasets.prepare import prepare_dataset_asarcher
from rag_data_generator.llm.client import OpenAILLMClient, SiliconFlowLLMClient
from rag_data_generator.pipeline.interruption import InterruptionOrchestrator
from rag_data_generator.prompts.prompts import SYSTEM_PROMPT_TOOLS_BACKTRACK, build_system_tools
from rag_data_generator.tooling.registry import ToolRegistry


def _format_entry(role: str, content: str) -> str:
    header = f"[{role.upper()}]"
    return f"{header}\n{content.strip()}\n"

def _strip_thinking(text: str) -> str:
    observations = re.findall(r"<observation>.*?</observation>", text, flags=re.DOTALL)
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    for obs in observations:
        if obs not in stripped:
            stripped = (stripped + "\n" + obs).strip() if stripped else obs.strip()
    return stripped


def run_demo(question: str, answer: str, max_rounds: int = 4, client_mode: str = "openai") -> None:
    system_prompt = build_system_tools(SYSTEM_PROMPT_TOOLS_BACKTRACK)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    client_mode = (client_mode or os.environ.get("LLM_CLIENT", "openai")).lower()
    if client_mode == "siliconflow":
        llm_client = SiliconFlowLLMClient()
        print("[INFO] Using SiliconFlow LLM client", flush=True)
    else:
        if client_mode != "openai":
            print(f"[WARN] Unknown client '{client_mode}', defaulting to OpenAI-compatible client.", flush=True)
        llm_client = OpenAILLMClient()
        print("[INFO] Using OpenAI-compatible LLM client", flush=True)
    registry = ToolRegistry()
    orchestrator = InterruptionOrchestrator(
        llm_client=llm_client,
        tool_registry=registry,
        max_rounds=max_rounds,
        debug=True,
    )
    try:
        result = orchestrator.run(messages)
    except RuntimeError as exc:
        print(f"[ERROR] Generation failed: {exc}", flush=True)
        return

    print("=" * 80)
    print("Question:")
    print(question)
    print("=" * 80)
    print("Agent transcript:")
    for entry in result["history"]:
        role = entry.get("role", "")
        if role == "system":
            continue  # hide system prompt from output
        content = _strip_thinking(entry.get("content", ""))
        print(_format_entry(role, content))

    print("=" * 80)
    print("Model final response:")
    print(_strip_thinking(result["response"] or ""))
    print("=" * 80)
    print("Gold answer:")
    print(answer)
    print("=" * 80)

    usage = getattr(llm_client, "last_usage", None)
    if usage:
        total = usage.get("total_tokens") if isinstance(usage, dict) else None
        print("Token usage summary:")
        if total is not None:
            print(f"  total_tokens: {total}")
        print(json.dumps(usage, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single QA example through the tool-aware pipeline.")
    parser.add_argument("--subset", default="hotpotqa_rand1000", help="ASearcher subset under data/ASearcher/")
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--index", type=int, default=0, help="Row index to sample from the dataset")
    parser.add_argument("--max-rounds", type=int, default=4, help="Max interruption rounds")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional size to sample from the parquet file")
    parser.add_argument("--client", default=os.environ.get("LLM_CLIENT", "openai"), choices=["openai", "siliconflow"], help="LLM backend to use")
    args = parser.parse_args()

    dataset = prepare_dataset_asarcher(split=args.split, subset=args.subset, sample_size=args.sample_size)
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"Index {args.index} out of bounds for dataset of size {len(dataset)}")
    record = dataset[args.index]

    run_demo(question=record["question"], answer=record["answer"], max_rounds=args.max_rounds, client_mode=args.client)


if __name__ == "__main__":
    main()
