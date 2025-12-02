"""Dataset loaders reused for generation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from datasets import Dataset, DownloadMode, load_dataset

from rag_data_generator.prompts.prompts import (
    SYSTEM_PROMPT_TOOLS_BACKTRACK,
    SYSTEM_PROMPT_TOOLS_SSRL,
    build_prompt,
    build_system_tools,
)

if os.environ.get("ASARCHER_DATA_DIR"):
    DEFAULT_ASARCHER_ROOT = Path(os.environ["ASARCHER_DATA_DIR"]).expanduser().resolve()
else:
    # fall back to the Agentic-RAG checkout that already bundles this dataset
    DEFAULT_ASARCHER_ROOT = (Path(__file__).resolve().parents[2] / "data" / "ASearcher").resolve()


def _resolve_system_prompt(use_ssrl: bool) -> str:
    if use_ssrl:
        return build_system_tools(SYSTEM_PROMPT_TOOLS_SSRL)
    return build_system_tools(SYSTEM_PROMPT_TOOLS_BACKTRACK)


def prepare_dataset(name: str = "gsm8k", split: str = "train", eval_size: int = 10, use_ssrl: bool = False, **kwargs):
    loaders = {
        "gsm8k": prepare_dataset_gsm8k,
        "medmcqa": prepare_dataset_medmcqa,
        "medqa": prepare_dataset_medqa,
        "asarcher": prepare_dataset_asarcher,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset name: {name}")
    return loaders[name](split=split, eval_size=eval_size, use_ssrl=use_ssrl, **kwargs)


def prepare_dataset_gsm8k(split: str = "train", eval_size: int = 10, use_ssrl: bool = False):
    data = load_dataset("openai/gsm8k", "main")[split]
    system_prompt = _resolve_system_prompt(use_ssrl)
    formatted = []
    for example in data:
        prompt = build_prompt(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["question"]},
            ]
        )
        formatted.append({"prompt": prompt, "answer": _extract_answer(example["answer"])})
    return formatted


def prepare_dataset_medmcqa(split: str = "train", eval_size: int = 10, use_ssrl: bool = False):
    data = load_dataset("medmcqa", split=split)
    system_prompt = _resolve_system_prompt(use_ssrl)
    formatted = []
    for example in data:
        options = "\n".join([f"{key}. {example[f'op{key.lower()}']}" for key in ["A", "B", "C", "D"]])
        question = f"Question: {example['question']}\nOptions:\n{options}"
        prompt = build_prompt(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        )
        correct_idx = int(example["cop"])
        answers = [example["opa"], example["opb"], example["opc"], example["opd"]]
        formatted.append({"prompt": prompt, "question": question, "answer": str(answers[correct_idx])})
    return formatted


def prepare_dataset_medqa(
    split: str = "train",
    eval_size: int = 10,
    use_ssrl: bool = False,
    dataset_name: str = "fzkuji/MedQA",
    dataset_config: str = "med_qa_zh_4options_bigbio_qa",
):
    cache_dir = os.path.join(".hf_cache", "medqa")
    try:
        data = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            cache_dir=cache_dir,
            download_mode=DownloadMode.FORCE_REDOWNLOAD,
        )
    except Exception:
        fallback = prepare_dataset_medmcqa(split=split, use_ssrl=use_ssrl)
        eval_data = fallback[:eval_size]
        train_data = fallback[eval_size:]
        return Dataset.from_list(train_data), Dataset.from_list(eval_data)

    system_prompt = _resolve_system_prompt(use_ssrl)
    formatted = []
    for idx, example in enumerate(data):
        options = "".join([f"{chr(65 + i)}. {choice}\n" for i, choice in enumerate(example["choices"])])
        user_content = f"Question: {example['question']}\nOptions:\n{options}"
        prompt = build_prompt(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        )
        formatted.append({"id": idx + 1, "prompt": prompt, "question": user_content, "answer": str(example["answer"][0])})
    eval_data = formatted[:eval_size]
    train_data = formatted[eval_size:]
    return Dataset.from_list(train_data), Dataset.from_list(eval_data)


def prepare_dataset_asarcher(
    split: str = "train",
    eval_size: int = 10,
    use_ssrl: bool = False,
    subset: str | None = None,
    root_dir: Path | None = None,
    sample_size: int | None = None,
):
    root = root_dir or DEFAULT_ASARCHER_ROOT
    if not root.exists():
        raise FileNotFoundError(f"ASearcher root not found: {root}")

    if split == "train":
        file_path = root / "train.parquet"
    else:
        subset = subset or "hotpotqa_rand1000"
        file_path = root / f"test_{subset}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"No subset named '{subset}' under {root}")

    df = pd.read_parquet(file_path)
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    system_prompt = _resolve_system_prompt(use_ssrl)
    records = []
    for _, row in df.iterrows():
        question = str(row["question"])
        answer = str(row["answer"])
        prompt = build_prompt(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        )
        records.append({"prompt": prompt, "question": question, "answer": answer})
    return Dataset.from_list(records)


def _extract_answer(answer_text: str) -> str:
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return answer_text.strip()


__all__ = [
    "prepare_dataset",
    "prepare_dataset_medqa",
    "prepare_dataset_medmcqa",
    "prepare_dataset_gsm8k",
    "prepare_dataset_asarcher",
]
