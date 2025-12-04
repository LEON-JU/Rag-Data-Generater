#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag_data_generator.tooling.registry import ToolRegistry
from rag_data_generator.utils.config import (
    apply_environment_overrides,
    choose,
    load_yaml_config,
)

DEFAULT_CONFIG_PATH = ROOT / "configs/multi_agent_langchain.yaml"


# =============== 工具函数 ===============

TAG_PATTERN_CACHE: Dict[str, re.Pattern[str]] = {}


def extract_between_tags(text: str, tag: str) -> str:
    """
    从文本中抽取 <tag>...</tag> 之间的内容。
    如果不存在对应 tag，则返回空字符串。
    """

    if tag not in TAG_PATTERN_CACHE:
        TAG_PATTERN_CACHE[tag] = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
    match = TAG_PATTERN_CACHE[tag].search(text)
    return match.group(1).strip() if match else ""


def build_chain(prompt: ChatPromptTemplate, llm: ChatOpenAI):
    return prompt | llm | StrOutputParser()


# =============== 各个 Agent 的 Prompt ===============

# ---- Agent 1: Reasoning Agent ----
reasoning_prompt = ChatPromptTemplate.from_template("""
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
""")


# ---- Agent 2: Search Agent ----
search_prompt = ChatPromptTemplate.from_template("""
You are Agent 2 (Search Agent).
You receive the question, reasoning and history.
Your job is to follow the instructions of reasoning or backtrack, form a string format search query consisted of at most 5 keywords.

Your search query must:
1. follows the insturctions in <reasoning></reasoning> or <backtrack></backtrack>, search for the required topics.
2. Your search query is consisted of some key-words: "keyword_1 keyword_2 ..."
   - No verbs like "search", "find", "look up".
   - No full sentences, no question marks.
3. input context may contain search plans, so you should check the history and see what topics have not been searched yet, you should only search for one specific topic in one search.
4. the search query is later searched in wikipedia, it prefers specific and compact queries.
                                                 
Output format (MUST):
- Output ONLY one block: <search>...</search>
- NO extra text, NO explanation.
                                                 

Question:
{question}

Reasoning:
{reasoning}

History:
{history}
""")


# ---- Agent 3: Summary Agent ----
summary_prompt = ChatPromptTemplate.from_template("""
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
""")


# ---- Agent 4: Backtrack / Verification Agent ----
backtrack_prompt = ChatPromptTemplate.from_template("""
You are Agent 4 (Verification & Backtracking Agent).  
You receive the question, the initial reasoning, the search query, and the summary.  
Your job is to decide whether the information in the summary is sufficient to answer the question.

Your responsibilities:
• Re-evaluate the plan made in the initial reasoning, if the search and summary already covers necessary information, produce <backtrack></backtrack> to indicate no correction needed.
• Check whether the recent search and summary cover the required entities, if not, produce a <backtrack> block that:  
    – states what should be searched next, using explicit entity names (not the full question)  
• Check whether the recent search and summary is related to the required entities, if not, then produce a <backtrack> block that:                                        
    – explains what went wrong (unclear target, irrelevant article, missing entity, wrong search target, etc.)  
    – revises the plan if necessary (e.g., choose a more specific entity or break it down further, correct wrong search query)

Format rules:
• Output exactly one <backtrack>...</backtrack> block, put your backtrack thoughts in this block.
• You are not allowed to include <search>...</search> in your output.

Question:
{question}

Reasoning:
{reasoning}

Search:
{search}

Summary:
{summary}
""")


# ---- Agent 5: Answer Agent（可选，但SFT一般需要） ----
answer_prompt = ChatPromptTemplate.from_template("""
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
""")


class MultiAgentPipeline:
    def __init__(self, llm: ChatOpenAI, registry: ToolRegistry | None = None) -> None:
        self.reasoning_chain = build_chain(reasoning_prompt, llm)
        self.search_chain = build_chain(search_prompt, llm)
        self.summary_chain = build_chain(summary_prompt, llm)
        self.backtrack_chain = build_chain(backtrack_prompt, llm)
        self.answer_chain = build_chain(answer_prompt, llm)
        self.registry = registry or ToolRegistry()

    def run_registry_search(self, search_block: str) -> str:
        """
        从 <search> 标签中抽取 query，调用已有的 ToolRegistry(Wiki_RAG)。
        """

        query = extract_between_tags(search_block, "search")
        if not query:
            return ""
        try:
            payload = json.dumps({"input": query}, ensure_ascii=False)
            result = self.registry.invoke("Wiki_RAG", payload)
            return result.to_observation()
        except Exception as exc:  # pragma: no cover - defensive
            return json.dumps({"plugin": "Wiki_RAG", "ok": False, "error": str(exc)}, ensure_ascii=False)

    def run_pipeline_for_question(self, question: str, max_rounds: int = 5, verbose: bool = False) -> Dict[str, Any]:
        """
        完整 pipeline：
        reasoning → 多轮(search, observation, summary, backtrack) → answer
        最终输出是线性扁平化结构，而不是多层结构。
        """

        if verbose:
            print(f"\nQuestion: {question}", flush=True)
        reasoning = self.reasoning_chain.invoke({"question": question})
        if verbose:
            print(reasoning.strip(), flush=True)

        history: List[Dict[str, str]] = []
        flat: Dict[str, Any] = {"question": question, "reasoning": reasoning, "steps": []}

        search = self.search_chain.invoke({"question": question, "reasoning": reasoning, "history": history})
        if verbose:
            print(search.strip(), flush=True)

        round_idx = 1
        final_summary = ""

        while round_idx <= max_rounds:
            if verbose:
                print(f"\nRound {round_idx}", flush=True)

            observation = self.run_registry_search(search)
            if verbose:
                print(observation, flush=True)

            summary = self.summary_chain.invoke(
                {
                    "question": question,
                    "reasoning": reasoning,
                    "history": history,
                    "search": search,
                    "observation": observation,
                }
            )
            if verbose:
                print(summary.strip(), flush=True)

            backtrack = self.backtrack_chain.invoke(
                {
                    "question": question,
                    "reasoning": reasoning,
                    "history": history,
                    "search": search,
                    "summary": summary,
                }
            )
            if verbose:
                print(backtrack.strip(), flush=True)

            flat["steps"].append(
                {
                    "search": search,
                    "observation": observation,
                    "summary": summary,
                    "backtrack": backtrack,
                }
            )

            history.append({"search": search, "summary": summary, "backtrack": backtrack})

            final_summary = summary
            extracted = extract_between_tags(backtrack, "backtrack")
            if "<backtrack" in backtrack.lower():
                backtrack_content = extracted
            else:
                backtrack_content = backtrack
            if not backtrack_content.strip():
                break

            round_idx += 1
            if round_idx > max_rounds:
                break

            search = self.search_chain.invoke({"question": question, "reasoning": reasoning, "history": history})
            if verbose:
                print(search.strip(), flush=True)

        answer = self.answer_chain.invoke(
            {"question": question, "reasoning": reasoning, "history": history, "summary": final_summary}
        )
        if verbose:
            print(answer.strip(), flush=True)

        flat["answer"] = answer
        flat["flattened_text"] = flatten_pipeline_output(flat)
        if verbose:
            print("\n[Pipeline] flattened output ready.", flush=True)

        return flat


def flatten_pipeline_output(flat: Dict[str, Any]) -> str:
    """
    将 run_pipeline_for_question() 的返回值扁平化成
    完整的 tags 序列文本，用于 SFT。
    """

    parts: List[str] = []
    parts.append(flat["reasoning"].strip())
    for step in flat["steps"]:
        parts.append(step["search"].strip())
        obs_block = f"<observation>{step['observation']}</observation>"
        parts.append(obs_block)
        parts.append(step["summary"].strip())
        parts.append(step["backtrack"].strip())
    parts.append(flat["answer"].strip())
    return "\n".join(parts)


def generate_dataset_jsonl(
    questions: List[str],
    output_path: str,
    pipeline: MultiAgentPipeline,
    max_rounds: int,
    verbose: bool = False,
) -> None:
    """
    生成 SFT JSONL，每条数据结构为：
    {
      "id": n,
      "messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": flattened_text}
      ]
    }
    """

    with open(output_path, "w", encoding="utf-8") as handle:
        for idx, question in enumerate(questions):
            flat = pipeline.run_pipeline_for_question(question, max_rounds=max_rounds, verbose=verbose)
            record = {
                "id": idx,
                "messages": [
                    {"role": "user", "content": question.strip()},
                    {"role": "assistant", "content": flat["flattened_text"].strip()},
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def build_llm_from_config(llm_config: dict | None) -> ChatOpenAI:
    llm_config = llm_config or {}
    kwargs: Dict[str, Any] = {}
    for field in ("model", "base_url", "api_key", "temperature"):
        if field in llm_config and llm_config[field] is not None:
            kwargs[field] = llm_config[field]
    if not kwargs:
        raise ValueError("请在 configs/multi_agent_langchain.yaml 中设置 llm 配置。")
    return ChatOpenAI(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent dataset generator demo.")
    parser.add_argument("--config", default=None, help="Path to the YAML config (defaults to configs/multi_agent_langchain.yaml)")
    parser.add_argument("--output", default=None, help="Destination JSONL path")
    parser.add_argument("--max-rounds", type=int, default=None, help="Max reasoning rounds per question")
    parser.add_argument("--verbose", action="store_true", help="Print per-agent progress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or DEFAULT_CONFIG_PATH
    config = load_yaml_config(config_path)
    apply_environment_overrides(config.get("env"))

    llm = build_llm_from_config(config.get("llm"))
    pipeline = MultiAgentPipeline(llm=llm)

    dataset_cfg = config.get("dataset", {})
    questions = dataset_cfg.get("questions") or []
    if not questions:
        raise ValueError("请在 config 的 dataset.questions 中至少写入一个问题。")

    pipeline_cfg = config.get("pipeline", {})
    max_rounds = choose(args.max_rounds, pipeline_cfg.get("max_rounds"), default=5)
    output_path = choose(args.output, config.get("output", {}).get("path"), default="multi_agent_sft_dataset.jsonl")

    verbose = pipeline_cfg.get("verbose")

    generate_dataset_jsonl(
        questions=questions,
        output_path=output_path,
        pipeline=pipeline,
        max_rounds=max_rounds,
        verbose=verbose,
    )

    print(f"Done. Dataset written to {output_path}")


if __name__ == "__main__":
    main()
