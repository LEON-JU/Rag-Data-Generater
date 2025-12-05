#!/usr/bin/env python3
"""
Multi-agent SFT data generator (multi-threaded).

Usage example:

  cd /path/to/Rag-Data-Generater
  python examples/multi_agent_langchain/multi_agent_sft_generate.py \
      --config configs/multi_agent_langchain.yaml \
      --output-dir sft_data/multi_agent \
      --split train \
      --num-workers 8 \
      --include-answer

会在 output-dir 下生成结构：
sft_data/multi_agent/
  reasoning/train.jsonl
  search/train.jsonl
  summary/train.jsonl
  backtrack/train.jsonl
  answer/train.jsonl   (如果 --include-answer)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:  # optional dependency used for progress display
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore

# ---- set repo root so we can import rag_data_generator* ----
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# third-party / project imports
from datasets import Dataset  # type: ignore
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from rag_data_generator.datasets.prepare import prepare_dataset_asarcher
from rag_data_generator.tooling.registry import ToolRegistry
from rag_data_generator.utils.config import (
    apply_environment_overrides,
    choose,
    load_yaml_config,
)

DEFAULT_CONFIG_PATH = ROOT / "configs/multi_agent_langchain.yaml"

DEFAULT_HOTPOTQA_FILES = (
    Path("/home/juyiang/data/hotpotqa/fullwiki/train-00000-of-00002.parquet"),
    Path("/home/juyiang/data/hotpotqa/fullwiki/train-00001-of-00002.parquet"),
)
DEFAULT_TWOWIKI_FILE = Path("/home/juyiang/data/2wikimultihopqa/train.parquet")

DATASET_CHOICES = (
    "asarcher",
    "hotpotqa_fullwiki",
    "twowikimultihopqa",
)


# ===================== 工具函数 =====================

TAG_PATTERN_CACHE: Dict[str, Any] = {}
_thread_local = threading.local()
LLM_CONFIG: Dict[str, Any] = {}
REGISTRY_ARGS: Dict[str, Any] = {}  # 目前没用，占位方便以后扩展


def extract_between_tags(text: str, tag: str) -> str:
    """Extract content between <tag>...</tag>, return '' if not found."""
    import re

    if tag not in TAG_PATTERN_CACHE:
        TAG_PATTERN_CACHE[tag] = re.compile(
            rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE
        )
    match = TAG_PATTERN_CACHE[tag].search(text)
    return match.group(1).strip() if match else ""


def build_chain(prompt: ChatPromptTemplate, llm: ChatOpenAI):
    return prompt | llm | StrOutputParser()


def build_llm_from_config(llm_config: dict | None) -> ChatOpenAI:
    llm_config = llm_config or {}
    kwargs: Dict[str, Any] = {}
    for field in ("model", "base_url", "api_key", "temperature", "timeout"):
        if field in llm_config and llm_config[field] is not None:
            kwargs[field] = llm_config[field]
    if not kwargs:
        raise ValueError(
            "Please set LLM configuration in configs/multi_agent_langchain.yaml under `llm`."
        )
    return ChatOpenAI(**kwargs)


def get_pipeline() -> "SingleRoundPipeline":
    """Per-thread lazy construction of the multi-agent pipeline."""
    if not hasattr(_thread_local, "pipeline"):
        llm = build_llm_from_config(LLM_CONFIG)
        registry = ToolRegistry(**REGISTRY_ARGS)
        _thread_local.pipeline = SingleRoundPipeline(llm=llm, registry=registry)
    return _thread_local.pipeline


# ===================== 各个 Agent 的 Prompt =====================

# Reasoning Agent
reasoning_prompt = ChatPromptTemplate.from_template(
    """
You are Agent 1 (Reasoning Agent).
You receive ONLY the user's question.
Your job is to perform internal reasoning and planning, and produce a single <reasoning>...</reasoning> block.

Your reasoning must:
• Analyze what the question is fundamentally asking.
• Identify the essential knowledge pieces required to answer it.
• Decide exactly which entities or concepts must be looked up.
• DO NOT instruct to search the full question verbatim.
• DO NOT phrase the search items as questions.
• Instead, decompose the question into atomic lookup targets (usually entities, names, organizations, locations, medical terms, or option labels).
• If multiple lookups are needed, explicitly enumerate them as:
    1. I need to check …
    2. I also need to verify …
    3. Next I should look up …

Format rules:
• Output ONLY one block: <reasoning>...</reasoning>, put your reasoning in this block.
• No other tags and no text outside the block.

Question:
{question}
""".strip()
)

# Search Agent
search_prompt = ChatPromptTemplate.from_template(
    """
You are Agent 2 (Search Agent).
You receive the question, reasoning and history.
Your job is to follow the instructions of reasoning or backtrack, and form a search query string consisting of at most 5 keywords.

Your search query must:
1. Follow the instructions in <reasoning></reasoning> or <backtrack></backtrack>, and search for the required topics.
2. The search query is a few key-words: "keyword_1 keyword_2 ...".
   - No verbs like "search", "find", "look up".
   - No full sentences, no question marks.
3. Input context may contain previous search plans; you should check the history and see what topics have not been searched yet.
   You should only search for ONE specific topic in one search.
4. The search query will be used in a Wikipedia-based retriever, so it should be specific and compact.

Output format (MUST):
- Output ONLY one block: <search>...</search>
- NO extra text, NO explanation.

Question:
{question}

Reasoning:
{reasoning}

History:
{history}
""".strip()
)

# Summary Agent
summary_prompt = ChatPromptTemplate.from_template(
    """
You are Agent 3 (Summary Agent).
You receive the question, reasoning, search query, and the observation (search results).
Your job is to extract and condense ONLY the key factual information from the observation, focusing on what is relevant for resolving the question.

Requirements:
• Do NOT perform further reasoning.
• Do NOT introduce new inferences.
• Do NOT restate the entire observation—summarize only the crucial facts.
• Preserve names, dates, functions, relationships, and definitions that are essential to answering the original question.
• If the observation contains irrelevant details, omit them.

Output:
• Produce exactly one <summary>...</summary> block, put your summary in this block.
• No additional text or tags.

Question:
{question}

Reasoning:
{reasoning}

Search:
{search}

Observation:
{observation}
""".strip()
)

# Backtrack Agent
backtrack_prompt = ChatPromptTemplate.from_template(
    """
You are Agent 4 (Verification & Backtracking Agent).
You receive the question, the initial reasoning, the search query, and the summary.
Your job is to decide whether the information in the summary is sufficient to answer the question.

Your responsibilities:
• Re-evaluate the plan made in the initial reasoning. If the search and summary already cover necessary information, output a single <backtrack>...</backtrack> block explaining why the information is adequate, followed by an <end> tag to indicate that no further search is needed. Produce exactly one <backtrack> block and one <end> tag.\n
• Check whether the recent search and summary cover the required entities. If not, produce a <backtrack> block that:
    – states what should be searched next, using explicit entity names (not the full question)
• Check whether the recent search and summary is related to the required entities. If not, then produce a <backtrack> block that:
    – explains what went wrong (unclear target, irrelevant article, missing entity, wrong search target, etc.)
    – revises the plan if necessary (e.g., choose a more specific entity or break it down further, correct wrong search query)

Format rules:
• Based on the above conditions, output either a single <backtrack> block, or a <backtrack> block followed by an <end> tag.
eg1: <backtrack>The summary provides ... , thus it is sufficient to answer the question.</backtrack><end>
eg2: <backtrack>The summary provides ... , it is not sufficient to answer the question. Therefore, the next step should focus on finding out ...</backtrack><end>
• You are not allowed to include <search>...</search> in your output.

Question:
{question}

Reasoning:
{reasoning}

Search:
{search}

Summary:
{summary}
""".strip()
)

# Answer Agent
answer_prompt = ChatPromptTemplate.from_template(
    """
You are the final Answer Agent.
You receive the question, the initial reasoning, and the summary of relevant information.
Your task is to provide the final answer to the user.

Rules:
• Base your answer strictly on the information in the summary and generally accepted knowledge.
• Do NOT expose internal reasoning or planning.
• The answer should be concise and directly address the question.

Format:
• Output exactly one <answer>...</answer> block, put your answer in this block.
• No other tags or explanations.

Question:
{question}

Reasoning:
{reasoning}

Summary:
{summary}
""".strip()
)


# ===================== 单轮 Pipeline =====================


class SingleRoundPipeline:
    """只跑一轮的 multi-agent pipeline，用于合成 SFT 轨迹。"""

    def __init__(self, llm: ChatOpenAI, registry: ToolRegistry | None = None) -> None:
        self.reasoning_chain = build_chain(reasoning_prompt, llm)
        self.search_chain = build_chain(search_prompt, llm)
        self.summary_chain = build_chain(summary_prompt, llm)
        self.backtrack_chain = build_chain(backtrack_prompt, llm)
        self.answer_chain = build_chain(answer_prompt, llm)
        self.registry = registry or ToolRegistry()

    def run_registry_search(self, search_block: str) -> str:
        """Extract <search> query and call Wiki_RAG via ToolRegistry."""
        query = extract_between_tags(search_block, "search")
        if not query:
            return ""
        try:
            payload = json.dumps({"input": query}, ensure_ascii=False)
            result = self.registry.invoke("Wiki_RAG", payload)
            return result.to_observation()
        except Exception as exc:  # pragma: no cover
            return json.dumps(
                {"plugin": "Wiki_RAG", "ok": False, "error": str(exc)},
                ensure_ascii=False,
            )

    def run_single_round(self, question: str) -> Dict[str, Any]:
        """Run reasoning → search → observation → summary → backtrack → answer once."""
        history: List[Dict[str, str]] = []

        reasoning = self.reasoning_chain.invoke({"question": question})
        search = self.search_chain.invoke(
            {"question": question, "reasoning": reasoning, "history": history}
        )

        observation = self.run_registry_search(search)

        summary = self.summary_chain.invoke(
            {
                "question": question,
                "reasoning": reasoning,
                "search": search,
                "observation": observation,
            }
        )

        backtrack = self.backtrack_chain.invoke(
            {
                "question": question,
                "reasoning": reasoning,
                "search": search,
                "summary": summary,
            }
        )

        answer = self.answer_chain.invoke(
            {"question": question, "reasoning": reasoning, "summary": summary}
        )

        return {
            "question": question,
            "reasoning": reasoning,
            "search": search,
            "observation": observation,
            "summary": summary,
            "backtrack": backtrack,
            "answer": answer,
        }


# ===================== SFT User Prompt 模板 =====================


def build_reasoning_user_prompt(question: str) -> str:
    return (
        "For the following question, decompose the key concepts and entities that need "
        "to be looked up in external tools. Do NOT answer the question itself.\n\n"
        f"Question:\n{question}\n\n"
        "You must output exactly one <reasoning>...</reasoning> block and nothing else."
    )


def build_search_user_prompt(question: str, reasoning: str) -> str:
    return (
        "Given the user's question and the previous <reasoning>...</reasoning> block, "
        "produce a search query consisting of at most 5 keywords.\n\n"
        "The search query:\n"
        "- should only contain keywords separated by spaces;\n"
        "- should not be a full sentence or a question;\n"
        "- should focus on a single topic to look up.\n\n"
        f"Question:\n{question}\n\n"
        f"Reasoning:\n{reasoning.strip()}\n\n"
        "Output exactly one <search>...</search> block and nothing else."
    )


def build_summary_user_prompt(
    question: str, reasoning: str, search: str, observation: str
) -> str:
    obs_block = f"<observation>{observation}</observation>"
    return (
        "You are given the internal reasoning, the search query, and the observation "
        "returned by a search tool. Your task is to summarize ONLY the key factual "
        "information from the observation that is relevant to answering the question.\n\n"
        "Do NOT add any new reasoning or speculation.\n\n"
        f"Question:\n{question}\n\n"
        f"{reasoning.strip()}\n\n"
        f"{search.strip()}\n\n"
        f"{obs_block}\n\n"
        "Output exactly one <summary>...</summary> block and nothing else."
    )


def build_backtrack_user_prompt(
    question: str, reasoning: str, search: str, summary: str
) -> str:
    return (
        "Given the question, the initial <reasoning>...</reasoning>, the <search>...</search> "
        "query, and the <summary>...</summary> of the observations, determine whether the "
        "available information is sufficient to answer the question.\n\n"
        
        "If it IS sufficient, output a single <backtrack>...</backtrack> block explaining why "
        "the information is adequate, followed by an <end> tag to indicate that no further "
        "search is needed. Produce exactly one <backtrack> block and one <end> tag.\n"
        
        "If it is NOT sufficient, output a single <backtrack>...</backtrack> block explaining "
        "what additional entities or topics should be searched next. Produce exactly one "
        "<backtrack> block and do NOT output an <end> tag.\n\n"
        
        f"Question:\n{question}\n\n"
        f"{reasoning.strip()}\n\n"
        f"{search.strip()}\n\n"
        f"{summary.strip()}\n\n"
        
        "Based on the above conditions, output either a single <backtrack> block, or a "
        "<backtrack> block followed by an <end> tag."
        "eg1: <backtrack>The summary provides ... , thus it is sufficient to answer the question.</backtrack><end>"
        "eg2: <backtrack>The summary provides ... , it is not sufficient to answer the question. Therefore, the next step should focus on finding out ...</backtrack><end>"
    )


def build_answer_user_prompt(question: str, reasoning: str, summary: str) -> str:
    return (
        "You are the final Answer Agent in a multi-agent RAG system.\n\n"
        "Based on the question, the internal reasoning and the factual <summary>...</summary> "
        "information extracted from retrieved documents, provide the final answer to the user.\n\n"
        "Use only generally accepted knowledge and the information in the summary. "
        "Do NOT include your internal reasoning.\n\n"
        f"Question:\n{question}\n\n"
        f"{reasoning.strip()}\n\n"
        f"{summary.strip()}\n\n"
        "Output exactly one <answer>...</answer> block and nothing else."
    )


# ===================== I/O & 多线程逻辑 =====================


def load_existing_ids(path: Path) -> set[int]:
    ids: set[int] = set()
    if not path.is_file():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ids.add(int(obj["id"]))
            except Exception:
                continue
    return ids


def prepare_output_files(base_dir: Path, split: str, include_answer: bool):
    abilities = ["reasoning", "search", "summary", "backtrack"]
    if include_answer:
        abilities.append("answer")

    paths: Dict[str, Path] = {}
    for ability in abilities:
        subdir = base_dir / ability
        subdir.mkdir(parents=True, exist_ok=True)
        paths[ability] = subdir / f"{split}.jsonl"
    return paths


def process_example(
    example_id: int,
    question: str,
    out_files: Dict[str, Any],
    existing_ids: Dict[str, set[int]],
    include_answer: bool,
    write_lock: threading.Lock,
) -> None:
    """Worker: run pipeline for one question & write SFT records."""
    pipeline = get_pipeline()
    try:
        result = pipeline.run_single_round(question)
    except Exception as e:
        logging.exception(f"[id={example_id}] pipeline error: {e}")
        return

    reasoning = result["reasoning"]
    search = result["search"]
    observation = result["observation"]
    summary = result["summary"]
    backtrack = result["backtrack"]
    answer = result["answer"]

    records: Dict[str, Dict[str, Any] | None] = {
        "reasoning": None,
        "search": None,
        "summary": None,
        "backtrack": None,
        "answer": None,
    }

    # Build records only if this id is not yet present for that ability
    if example_id not in existing_ids["reasoning"]:
        records["reasoning"] = {
            "id": example_id,
            "messages": [
                {
                    "role": "user",
                    "content": build_reasoning_user_prompt(question),
                },
                {"role": "assistant", "content": reasoning.strip()},
            ],
        }

    if example_id not in existing_ids["search"]:
        records["search"] = {
            "id": example_id,
            "messages": [
                {
                    "role": "user",
                    "content": build_search_user_prompt(question, reasoning),
                },
                {"role": "assistant", "content": search.strip()},
            ],
        }

    if example_id not in existing_ids["summary"]:
        records["summary"] = {
            "id": example_id,
            "messages": [
                {
                    "role": "user",
                    "content": build_summary_user_prompt(
                        question, reasoning, search, observation
                    ),
                },
                {"role": "assistant", "content": summary.strip()},
            ],
        }

    if example_id not in existing_ids["backtrack"]:
        records["backtrack"] = {
            "id": example_id,
            "messages": [
                {
                    "role": "user",
                    "content": build_backtrack_user_prompt(
                        question, reasoning, search, summary
                    ),
                },
                {"role": "assistant", "content": backtrack.strip()},
            ],
        }

    if include_answer and example_id not in existing_ids["answer"]:
        records["answer"] = {
            "id": example_id,
            "messages": [
                {
                    "role": "user",
                    "content": build_answer_user_prompt(
                        question, reasoning, summary
                    ),
                },
                {"role": "assistant", "content": answer.strip()},
            ],
        }

    # write to files with a global lock
    with write_lock:
        for ability, rec in records.items():
            if rec is None:
                continue
            if ability not in out_files:
                continue
            fh = out_files[ability]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

            # update in-memory existing_ids so后面的任务不会重复写
            existing_ids[ability].add(example_id)


def load_asarcher_questions(
    split: str, subset: str | None, sample_size: int | None
) -> List[Tuple[int, str]]:
    dataset: Dataset = prepare_dataset_asarcher(
        split=split, subset=subset, sample_size=sample_size
    )
    questions: List[Tuple[int, str]] = []
    for idx, row in enumerate(dataset):
        questions.append((idx, str(row["question"])))
    return questions


def resolve_dataset_files(
    dataset_name: str, provided_paths: List[str] | None
) -> List[Path]:
    if dataset_name == "asarcher":
        return []

    if provided_paths:
        paths = [Path(p).expanduser().resolve() for p in provided_paths]
    else:
        if dataset_name == "hotpotqa_fullwiki":
            paths = [Path(p) for p in DEFAULT_HOTPOTQA_FILES]
        else:
            paths = [Path(DEFAULT_TWOWIKI_FILE)]

    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing parquet files for dataset "
            f"{dataset_name}: {', '.join(missing)}. "
            "Please supply --dataset-files pointing to the correct location."
        )

    if dataset_name == "twowikimultihopqa" and len(paths) != 1:
        raise ValueError(
            "2WikiMultiHopQA expects exactly one parquet file. "
            "Provide a single path via --dataset-files."
        )

    return paths


def load_hotpotqa_questions(
    parquet_files: List[Path], sample_size: int | None
) -> List[Tuple[int, str]]:
    questions: List[Tuple[int, str]] = []
    next_id = 0
    for file_path in parquet_files:
        logging.info(f"Loading HotpotQA data from {file_path}")
        dataset = Dataset.from_parquet(str(file_path))
        for row in dataset:
            questions.append((next_id, str(row["question"])))
            next_id += 1
            if sample_size is not None and len(questions) >= sample_size:
                return questions
    return questions


def load_twowiki_questions(
    parquet_file: Path, sample_size: int | None
) -> List[Tuple[int, str]]:
    logging.info(f"Loading 2WikiMultiHopQA data from {parquet_file}")
    dataset = Dataset.from_parquet(str(parquet_file))
    questions: List[Tuple[int, str]] = []
    for idx, row in enumerate(dataset):
        questions.append((idx, str(row["question"])))
        if sample_size is not None and len(questions) >= sample_size:
            break
    return questions


# ===================== MAIN =====================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-agent SFT dataset generator (multi-threaded)."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config (default: configs/multi_agent_langchain.yaml)",
    )
    parser.add_argument(
        "--dataset",
        default="asarcher",
        choices=DATASET_CHOICES,
        help="Dataset to read questions from.",
    )
    parser.add_argument(
        "--dataset-files",
        nargs="+",
        default=None,
        help=(
            "Parquet file(s) for datasets backed by local files. "
            "Used when --dataset is hotpotqa_fullwiki or twowikimultihopqa."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="sft_data/multi_agent",
        help="Base directory to save SFT jsonl files.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Which split of ASearcher to use (default: train). Ignored by other datasets.",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Subset name for test split, e.g. 'hotpotqa_rand1000'. "
        "Applicable only when --dataset asarcher and split=test.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optionally subsample N examples from the dataset for debugging.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker threads to use.",
    )
    parser.add_argument(
        "--include-answer",
        action="store_true",
        help="Also generate SFT data for the Answer agent.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log every N finished examples.",
    )
    parser.add_argument(
        "--progress-bar",
        dest="progress_bar",
        action="store_true",
        help="Display a tqdm progress bar while generating.",
    )
    parser.add_argument(
        "--no-progress-bar",
        dest="progress_bar",
        action="store_false",
        help="Disable tqdm progress output.",
    )
    parser.set_defaults(progress_bar=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    # ---- load config & apply env overrides ----
    config = load_yaml_config(args.config)
    apply_environment_overrides(config.get("env"))

    global LLM_CONFIG
    LLM_CONFIG = config.get("llm") or {}
    if not LLM_CONFIG:
        raise ValueError("LLM config is empty in YAML file!")

    # ---- prepare dataset ----
    if args.dataset == "asarcher":
        questions = load_asarcher_questions(
            split=args.split, subset=args.subset, sample_size=args.sample_size
        )
        dataset_label = f"ASearcher ({args.split})"
    elif args.dataset == "hotpotqa_fullwiki":
        file_paths = resolve_dataset_files(args.dataset, args.dataset_files)
        questions = load_hotpotqa_questions(file_paths, sample_size=args.sample_size)
        dataset_label = (
            f"HotpotQA fullwiki ({len(file_paths)} parquet files)"
        )
    else:
        file_paths = resolve_dataset_files(args.dataset, args.dataset_files)
        questions = load_twowiki_questions(
            parquet_file=file_paths[0], sample_size=args.sample_size
        )
        dataset_label = "2WikiMultiHopQA"

    logging.info(
        f"Loaded {len(questions)} questions from {dataset_label} ({args.dataset})."
    )

    # ---- prepare output paths & already processed ids ----
    output_base = Path(args.output_dir).resolve()
    output_paths = prepare_output_files(
        output_base, split=args.split, include_answer=args.include_answer
    )

    existing_ids: Dict[str, set[int]] = {}
    for ability, path in output_paths.items():
        ids = load_existing_ids(path)
        existing_ids[ability] = ids
        logging.info(
            f"[{ability}] existing records: {len(ids)} at {os.fspath(path)}"
        )

    # open file handles once
    out_files: Dict[str, Any] = {}
    for ability, path in output_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        # append mode so我们可以断点续跑
        out_files[ability] = path.open("a", encoding="utf-8")

    # figure out which examples still need work (至少有一个能力没写)
    todo: List[Tuple[int, str]] = []
    for idx, q in questions:
        needed = False
        for ability in existing_ids.keys():
            if idx not in existing_ids[ability]:
                needed = True
                break
        if needed:
            todo.append((idx, q))

    logging.info(f"Total {len(questions)} questions, {len(todo)} left to process.")

    if not todo:
        logging.info("Nothing to do. All examples already processed.")
        for fh in out_files.values():
            fh.close()
        return

    write_lock = threading.Lock()

    # ---- multi-threaded execution ----
    total = len(todo)
    finished = 0

    progress_bar = None
    if args.progress_bar:
        if tqdm is None:
            logging.warning("tqdm is not installed, disabling progress bar output.")
        else:
            progress_bar = tqdm(total=total, desc="Examples", unit="ex")

    try:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_id = {
                executor.submit(
                    process_example,
                    idx,
                    q,
                    out_files,
                    existing_ids,
                    args.include_answer,
                    write_lock,
                ): idx
                for idx, q in todo
            }

            for future in as_completed(future_to_id):
                _ = future_to_id[future]
                # if there was an exception it has already been logged in process_example
                finished += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                if finished % max(1, args.log_interval) == 0:
                    logging.info(f"Progress: {finished}/{total} examples done.")
    except KeyboardInterrupt:
        logging.warning("Interrupted by user, stopping early. Partial results are saved.")
    finally:
        if progress_bar is not None:
            progress_bar.close()
        for fh in out_files.values():
            fh.close()
        logging.info("All file handles closed.")


if __name__ == "__main__":
    main()
