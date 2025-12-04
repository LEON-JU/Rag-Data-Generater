#!/usr/bin/env python3
"""
Quick interactive smoke test driven by configs/full_prompt_pipeline.yaml:

1. Pick one QA item (default: ASearcher hotpot subset)
2. Build the system prompt with tool descriptions
3. Run the interruption loop against the configured LLM endpoint
4. Stream out agent thoughts, tool calls, retrieved observations, and final answer
5. Print the gold answer for manual inspection
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录
sys.path.insert(0, str(ROOT))

from rag_data_generator.datasets.prepare import prepare_dataset_asarcher
from rag_data_generator.llm.client import OpenAILLMClient, SiliconFlowLLMClient
from rag_data_generator.pipeline.interruption import InterruptionOrchestrator
from rag_data_generator.prompts.prompts import SYSTEM_PROMPT_TOOLS_BACKTRACK, build_system_tools
from rag_data_generator.tooling.registry import ToolRegistry
from rag_data_generator.utils.config import (
    apply_environment_overrides,
    choose,
    load_yaml_config,
)

DEFAULT_CONFIG_PATH = ROOT / "configs/full_prompt_pipeline.yaml"

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


def build_llm_client(llm_config: dict | None, override: str | None) -> tuple[str, OpenAILLMClient | SiliconFlowLLMClient]:
    llm_config = llm_config or {}
    client_choice = choose(
        (override.lower() if isinstance(override, str) else override),
        llm_config.get("client"),
        os.environ.get("LLM_CLIENT"),
        default="openai",
    )
    client_choice = (client_choice or "openai").lower()

    if client_choice == "siliconflow":
        params = llm_config.get("siliconflow", {})
        return (
            "SiliconFlow",
            SiliconFlowLLMClient(
                model=params.get("model"),
                api_key=params.get("api_key"),
                base_url=params.get("base_url"),
                timeout=params.get("timeout", 60),
            ),
        )

    params = llm_config.get("openai", {})
    if client_choice not in {"openai", None}:
        print(f"[WARN] Unknown client '{client_choice}', defaulting to OpenAI-compatible client.", flush=True)
    return (
        "OpenAI-compatible",
        OpenAILLMClient(
            model=params.get("model"),
            api_key=params.get("api_key"),
            base_url=params.get("base_url"),
        ),
    )


def run_demo(
    question: str,
    answer: str,
    llm_client: OpenAILLMClient | SiliconFlowLLMClient,
    max_rounds: int = 4,
) -> None:
    system_prompt = build_system_tools(SYSTEM_PROMPT_TOOLS_BACKTRACK)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
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
    parser.add_argument("--config", default=None, help="Path to the YAML config (defaults to configs/full_prompt_pipeline.yaml)")
    parser.add_argument("--subset", default=None, help="ASearcher subset under data/ASearcher/")
    parser.add_argument("--split", default=None, choices=["test", "train"], help="Dataset split")
    parser.add_argument("--index", type=int, default=None, help="Row index to sample from the dataset")
    parser.add_argument("--max-rounds", type=int, default=None, help="Max interruption rounds")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional size to sample from the parquet file")
    parser.add_argument(
        "--client",
        default=None,
        choices=["openai", "siliconflow"],
        help="LLM backend to use (overrides config)",
    )
    args = parser.parse_args()

    config_path = args.config or DEFAULT_CONFIG_PATH
    config = load_yaml_config(config_path)
    apply_environment_overrides(config.get("env"))

    dataset_cfg = config.get("dataset", {})
    pipeline_cfg = config.get("pipeline", {})

    subset = choose(args.subset, dataset_cfg.get("subset"), default="hotpotqa_rand1000")
    split = choose(args.split, dataset_cfg.get("split"), default="test")
    sample_size = choose(args.sample_size, dataset_cfg.get("sample_size"))
    index = choose(args.index, dataset_cfg.get("index"), default=0)
    max_rounds = choose(args.max_rounds, pipeline_cfg.get("max_rounds"), default=4)

    dataset = prepare_dataset_asarcher(split=split, subset=subset, sample_size=sample_size)
    if index < 0 or index >= len(dataset):
        raise IndexError(f"Index {index} out of bounds for dataset of size {len(dataset)}")
    record = dataset[index]

    client_label, llm_client = build_llm_client(config.get("llm"), args.client)
    print(f"[INFO] Using {client_label} LLM client", flush=True)

    run_demo(
        question=record["question"],
        answer=record["answer"],
        llm_client=llm_client,
        max_rounds=max_rounds,
    )


if __name__ == "__main__":
    main()
