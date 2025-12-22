#!/usr/bin/env python3
"""
Single-agent streaming demo that drives a fine-tuned local model with one prompt.

The script:
1. Loads a QA record (defaults to ASearcher HotpotQA subset) by index.
2. Builds the single-agent system prompt that enforces the reasoning→question→summary→backtrack→end→answer order.
3. Streams tokens from a local OpenAI-compatible endpoint (e.g., vLLM), intercepts <question> blocks,
   executes the requested tool, and injects <observation> back into the conversation.
4. Prints the full streamed transcript plus the dataset's gold answer for manual evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag_data_generator.datasets.prepare import prepare_dataset_asarcher
from rag_data_generator.llm.client import OpenAILLMClient, SiliconFlowLLMClient
from rag_data_generator.pipeline.interruption import SearchTagDetector
from rag_data_generator.prompts.prompts import (
    SYSTEM_PROMPT_SINGLE_AGENT_STREAMING,
)
from rag_data_generator.tooling.registry import ToolRegistry
from rag_data_generator.utils.config import (
    apply_environment_overrides,
    choose,
    load_yaml_config,
)

DEFAULT_CONFIG_PATH = ROOT / "configs/single_agent_streaming.yaml"
DEFAULT_SUBSET = "hotpotqa_rand1000"
DEFAULT_SPLIT = "test"
DEFAULT_INDEX = 0
DEFAULT_MAX_ROUNDS = 6
DEFAULT_TEMPERATURE = 0.1
DEFAULT_PRINT_THINKING = True
DEFAULT_USE_STREAM = False

CLI_FLAG_MAP = {
    "config": "--config",
    "subset": "--subset",
    "split": "--split",
    "index": "--index",
    "sample_size": "--sample-size",
    "max_rounds": "--max-rounds",
    "temperature": "--temperature",
    "client": "--client",
    "print_thinking": "--print-thinking",
    "use_stream": "--stream",
}


def _flag_used(flag: str, argv: List[str] | None = None) -> bool:
    argv = argv or sys.argv[1:]
    for token in argv:
        if token == flag or token.startswith(f"{flag}="):
            return True
    return False


def _coerce_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def build_llm_client(
    llm_config: dict | None,
    override: str | None,
) -> Tuple[str, OpenAILLMClient | SiliconFlowLLMClient]:
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


class StreamingSingleAgentRunner:
    """Runs the interruption loop; streaming is optional."""

    def __init__(
        self,
        llm_client: OpenAILLMClient | SiliconFlowLLMClient,
        tool_registry: ToolRegistry,
        max_rounds: int = 6,
        temperature: float = 0.2,
        question_tag: str = "question",
        observation_template: str = "<observation>{}</observation>",
        print_thinking: bool = True,
        use_stream: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.max_rounds = max_rounds
        self.temperature = temperature
        self.detector = SearchTagDetector(tag=question_tag)
        self.observation_template = observation_template
        self.print_thinking = print_thinking
        self.use_stream = use_stream
        self._think_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)

    def _maybe_stream(
        self,
        history: List[dict],
    ) -> Tuple[str, re.Match | None, bool, bool]:
        """
        Returns (assistant text, <question> match, used_stream, saw_answer).
        """
        if not self.use_stream:
            text = self.llm_client.complete(history, temperature=self.temperature)
            if self.print_thinking:
                print(text, flush=True)
            return text, self._first_question_outside_think(text), False, "</answer>" in text

        stream_fn = getattr(self.llm_client, "stream_chat", None)
        if not callable(stream_fn):
            text = self.llm_client.complete(history, temperature=self.temperature)
            if self.print_thinking:
                print(text, flush=True)
            return text, self._first_question_outside_think(text), False, "</answer>" in text

        try:
            stream = stream_fn(history, temperature=self.temperature)
        except NotImplementedError:
            text = self.llm_client.complete(history, temperature=self.temperature)
            if self.print_thinking:
                print(text, flush=True)
            return text, self._first_question_outside_think(text), False, "</answer>" in text

        chunks: List[str] = []
        match: re.Match | None = None
        saw_answer = False
        try:
            for chunk in stream:
                if not chunk:
                    continue
                chunks.append(chunk)
                if self.print_thinking:
                    print(chunk, end="", flush=True)
                text = "".join(chunks)
                match = self._first_question_outside_think(text)
                if "</answer>" in text:
                    saw_answer = True
                    break
                if match:
                    break
        finally:
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                close_stream()
        text = "".join(chunks)
        if not match:
            match = self._first_question_outside_think(text)
        if self.print_thinking:
            print(flush=True)
        return text, match, True, saw_answer

    def run(self, messages: List[dict]) -> Dict[str, List[dict] | str]:
        history = list(messages)
        if self.print_thinking:
            for entry in history:
                print(entry["content"], end="\n\n", flush=True)
        transcript: List[dict] = []
        final_response: str | None = None

        for round_idx in range(self.max_rounds):
            response_text, question_match, used_stream, saw_answer = self._maybe_stream(history)
            if not response_text.strip():
                print("[WARN] Assistant returned an empty response, stopping.", flush=True)
                break

            if question_match:
                query = question_match.group(1).strip()
                result = self.tool_registry.invoke("Wiki_RAG", json.dumps({"input": query or ""}, ensure_ascii=False))
                observation = self.observation_template.format(result.to_observation())

                assistant_chunk = response_text[: question_match.end()].rstrip()
                if not assistant_chunk.endswith("\n"):
                    assistant_chunk += "\n"
                assistant_chunk = f"{assistant_chunk}{observation}\n"

                if self.print_thinking:
                    print(assistant_chunk, end="", flush=True)

                history.append({"role": "assistant", "content": assistant_chunk})
                transcript.append({"role": "assistant", "content": assistant_chunk})
                continue

            # No <question> tag detected—treat as a normal assistant turn.
            text = response_text
            answer_close = text.find("</answer>")
            if answer_close != -1:
                text = text[: answer_close + len("</answer>")]
                final_response = text

            if self.print_thinking:
                print(text, end="" if text.endswith("\n") else "\n", flush=True)
            history.append({"role": "assistant", "content": text})
            transcript.append({"role": "assistant", "content": text})
            if final_response or saw_answer:
                break

        return {"history": history, "response": final_response or "", "turns": transcript}

    def _first_question_outside_think(self, text: str) -> re.Match | None:
        matches = self.detector.matches(text)
        if not matches:
            return None
        think_spans = [(m.start(), m.end()) for m in self._think_pattern.finditer(text)]
        if not think_spans:
            return matches[0]
        for match in matches:
            start, end = match.start(), match.end()
            inside = False
            for t_start, t_end in think_spans:
                if t_start <= start and end <= t_end:
                    inside = True
                    break
            if not inside:
                return match
        return None


def run_single_agent_demo(
    question: str,
    answer: str,
    llm_client: OpenAILLMClient | SiliconFlowLLMClient,
    max_rounds: int,
    temperature: float,
    print_thinking: bool,
    use_stream: bool,
) -> None:
    system_prompt = SYSTEM_PROMPT_SINGLE_AGENT_STREAMING
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    runner = StreamingSingleAgentRunner(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
        max_rounds=max_rounds,
        temperature=temperature,
        print_thinking=print_thinking,
        use_stream=use_stream,
    )
    result = runner.run(messages)

    print("\n" + "=" * 80)
    print("Question:")
    print(question)
    print("=" * 80)
    print("Agent transcript:")
    for turn in result["turns"]:
        content = turn.get("content", "")
        print(f"[ASSISTANT]\n{content.strip()}\n")
    if not result["turns"]:
        print("[No assistant turns captured]\n")

    print("=" * 80)
    print("Model final response (trimmed to </answer>):")
    print(result["response"])
    print("=" * 80)
    print("Gold answer:")
    print(answer)
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-agent streaming QA demo.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to the YAML config file (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--subset",
        default=DEFAULT_SUBSET,
        help=f"ASearcher subset under data/ASearcher/ (default: {DEFAULT_SUBSET}).",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=["test", "train"],
        help=f"Dataset split (default: {DEFAULT_SPLIT}).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=DEFAULT_INDEX,
        help=f"Row index to sample from the dataset (default: {DEFAULT_INDEX}).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional size to sample from the parquet file (default: use full file).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=DEFAULT_MAX_ROUNDS,
        help=f"Maximum assistant rounds (default: {DEFAULT_MAX_ROUNDS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--client",
        default="openai",
        choices=["openai", "siliconflow"],
        help="LLM backend override (default: openai).",
    )
    parser.add_argument(
        "--print-thinking",
        default=str(DEFAULT_PRINT_THINKING).lower(),
        choices=["true", "false"],
        help="Whether to stream the model's thinking/output (default: true).",
    )
    parser.add_argument(
        "--stream",
        default=str(DEFAULT_USE_STREAM).lower(),
        choices=["true", "false"],
        help="Enable token streaming (default: false).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml_config(config_path)
    apply_environment_overrides(config.get("env"))

    dataset_cfg = config.get("dataset", {})
    pipeline_cfg = config.get("pipeline", {})

    subset_cli = args.subset if _flag_used(CLI_FLAG_MAP["subset"]) else None
    split_cli = args.split if _flag_used(CLI_FLAG_MAP["split"]) else None
    sample_cli = args.sample_size if _flag_used(CLI_FLAG_MAP["sample_size"]) else None
    index_cli = args.index if _flag_used(CLI_FLAG_MAP["index"]) else None
    rounds_cli = args.max_rounds if _flag_used(CLI_FLAG_MAP["max_rounds"]) else None
    temp_cli = args.temperature if _flag_used(CLI_FLAG_MAP["temperature"]) else None
    client_cli = args.client if _flag_used(CLI_FLAG_MAP["client"]) else None
    thinking_cli = (
        _coerce_bool(args.print_thinking) if _flag_used(CLI_FLAG_MAP["print_thinking"]) else None
    )
    stream_cli = (
        _coerce_bool(args.stream) if _flag_used(CLI_FLAG_MAP["use_stream"]) else None
    )

    subset = choose(subset_cli, dataset_cfg.get("subset"), default=DEFAULT_SUBSET)
    split = choose(split_cli, dataset_cfg.get("split"), default=DEFAULT_SPLIT)
    sample_size = choose(sample_cli, dataset_cfg.get("sample_size"))
    index = choose(index_cli, dataset_cfg.get("index"), default=DEFAULT_INDEX)
    max_rounds = choose(rounds_cli, pipeline_cfg.get("max_rounds"), default=DEFAULT_MAX_ROUNDS)
    temperature = choose(temp_cli, pipeline_cfg.get("temperature"), default=DEFAULT_TEMPERATURE)
    print_thinking = choose(
        thinking_cli,
        _coerce_bool(pipeline_cfg.get("print_thinking")),
        default=DEFAULT_PRINT_THINKING,
    )
    use_stream = choose(
        stream_cli,
        _coerce_bool(pipeline_cfg.get("stream")),
        default=DEFAULT_USE_STREAM,
    )

    dataset = prepare_dataset_asarcher(split=split, subset=subset, sample_size=sample_size)
    if index < 0 or index >= len(dataset):
        raise IndexError(f"Index {index} out of bounds for dataset of size {len(dataset)}")
    record = dataset[index]

    client_label, llm_client = build_llm_client(config.get("llm"), client_cli)
    print(f"[INFO] Using {client_label} LLM client", flush=True)

    run_single_agent_demo(
        question=record["question"],
        answer=record["answer"],
        llm_client=llm_client,
        max_rounds=max_rounds,
        temperature=temperature,
        print_thinking=print_thinking,
        use_stream=use_stream,
    )


if __name__ == "__main__":
    main()
