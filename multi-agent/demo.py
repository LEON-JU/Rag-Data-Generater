import json
import re
import argparse
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # 获取 ../
sys.path.insert(0, str(ROOT))                  # 加入 PYTHONPATH

from rag_data_generator.tooling.registry import ToolRegistry


# =============== 工具函数 ===============

TAG_PATTERN_CACHE = {}

def extract_between_tags(text: str, tag: str) -> str:
    """
    从文本中抽取 <tag>...</tag> 之间的内容。
    如果不存在对应 tag，则返回空字符串。
    """
    if tag not in TAG_PATTERN_CACHE:
        TAG_PATTERN_CACHE[tag] = re.compile(
            rf"<{tag}>(.*?)</{tag}>",
            re.DOTALL | re.IGNORECASE
        )
    m = TAG_PATTERN_CACHE[tag].search(text)
    return m.group(1).strip() if m else ""


# =============== 初始化 LLM & Wikipedia 工具 ===============

# # Qwen/Qwen2.5-72B-Instruct-128K Qwen/Qwen3-32B
# llm = ChatOpenAI(
#     model='Qwen/Qwen2.5-72B-Instruct-128K',
#     base_url='https://api.siliconflow.cn/v1',
#     api_key='sk-kiwmknhfsbjtmqojloxxoyhxnbkhrmqnhpcwxwzigpmyczto',
#     temperature=0.3
# )

llm = ChatOpenAI(
    model='/home/juyiang/data/llm_models/qwen25-32b-awq',
    base_url='http://localhost:8000/v1',
    api_key='dummy'
)

str_parser = StrOutputParser()

registry = ToolRegistry()


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

reasoning_chain = reasoning_prompt | llm | str_parser


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


search_chain = search_prompt | llm | str_parser


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

summary_chain = summary_prompt | llm | str_parser


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
• You are not allowed to include <search>...<\search> in your output.

Question:
{question}

Reasoning:
{reasoning}

Search:
{search}

Summary:
{summary}
""")

backtrack_chain = backtrack_prompt | llm | str_parser


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

answer_chain = answer_prompt | llm | str_parser


# =============== Wiki_RAG 检索 ===============

def run_registry_search(search_block: str) -> str:
    """
    从 <search> 标签中抽取 query，调用已有的 ToolRegistry(Wiki_RAG)。
    """
    query = extract_between_tags(search_block, "search")
    if not query:
        return ""
    try:
        payload = json.dumps({"input": query}, ensure_ascii=False)
        result = registry.invoke("Wiki_RAG", payload)
        return result.to_observation()
    except Exception as e:
        return json.dumps({"plugin": "Wiki_RAG", "ok": False, "error": str(e)}, ensure_ascii=False)


# =============== 单问题 Pipeline ===============
def flatten_pipeline_output(flat: Dict[str, Any]) -> str:
    """
    将 run_pipeline_for_question() 的返回值扁平化成
    完整的 tags 序列文本，用于 SFT。
    """

    parts = []

    # 1) reasoning（不在 steps 里）
    parts.append(flat["reasoning"].strip())

    # 2) 多轮 search/obs/summary/backtrack
    for step in flat["steps"]:
        parts.append(step["search"].strip())

        # observation 本不是 tag，但你希望完整串联，所以保持原样
        obs_block = f"<observation>{step['observation']}</observation>"
        parts.append(obs_block)

        parts.append(step["summary"].strip())
        parts.append(step["backtrack"].strip())

    # 3) answer（最终）
    parts.append(flat["answer"].strip())

    # 用两行分隔更清晰，也可以换成 '\n' 单行
    return "\n".join(parts)


def build_search_block(query: str) -> str:
    return f"<search>{query}</search>"


def run_pipeline_for_question(question: str, max_rounds: int = 5, verbose: bool = False) -> Dict[str, Any]:
    """
    完整 pipeline：
    reasoning → 多轮(search, observation, summary, backtrack) → answer
    最终输出是线性扁平化结构，而不是多层结构。
    """

    # =========================================
    # Step 1: reasoning (only once)
    # =========================================
    if verbose:
        print(f"\nQuestion: {question}", flush=True)
    reasoning = reasoning_chain.invoke({"question": question})
    if verbose:
        print(reasoning.strip(), flush=True)

    # 用来内部给 search/summary/backtrack 追加历史
    history = []

    # 用来最终输出扁平化字段
    flat = {
        "question": question,
        "reasoning": reasoning,
        "steps": []   # 每一步包含 search/obs/summary/backtrack
    }

    # =========================================
    # Step 2: 第一轮 search from reasoning
    # =========================================
    search = search_chain.invoke({
        "question": question,
        "reasoning": reasoning,
        "history": history,
    })
    if verbose:
        print(search.strip(), flush=True)

    round_idx = 1
    final_summary = ""

    while round_idx <= max_rounds:
        if verbose:
            print(f"\nRound {round_idx}", flush=True)

        # =======================
        # Observation
        # =======================
        observation = run_registry_search(search)
        if verbose:
            print(observation, flush=True)

        # =======================
        # Summary
        # =======================
        summary = summary_chain.invoke({
            "question": question,
            "reasoning": reasoning,
            "history": history,
            "search": search,
            "observation": observation,
        })
        if verbose:
            print(summary.strip(), flush=True)

        # =======================
        # Backtrack
        # =======================
        backtrack = backtrack_chain.invoke({
            "question": question,
            "reasoning": reasoning,
            "history": history,
            "search": search,
            "summary": summary,
        })
        if verbose:
            print(backtrack.strip(), flush=True)

        # ===== 保存扁平化一条轨迹 ======
        flat["steps"].append({
            "search": search,
            "observation": observation,
            "summary": summary,
            "backtrack": backtrack,
        })

        # ===== 更新 history（给下一轮 prompt 用） =====
        history.append({
            "search": search,
            "summary": summary,
            "backtrack": backtrack,
        })

        # ===== 检查 Backtrack 是否需要继续，非空则触发下一轮 search =====
        final_summary = summary
        backtrack_content = extract_between_tags(backtrack, "backtrack") or backtrack
        if not backtrack_content.strip():
            break

        round_idx += 1
        if round_idx > max_rounds:
            break

        search = search_chain.invoke({
            "question": question,
            "reasoning": reasoning,
            "history": history,
        })
        if verbose:
            print(search.strip(), flush=True)

    # =========================================
    # Step 3: Answer
    # =========================================
    answer = answer_chain.invoke({
        "question": question,
        "reasoning": reasoning,
        "history": history,
        "summary": final_summary,
    })
    if verbose:
        print(answer.strip(), flush=True)

    flat["answer"] = answer
    flat["flattened_text"] = flatten_pipeline_output(flat)
    if verbose:
        print("\n[Pipeline] flattened output ready.", flush=True)

    return flat


# =============== 批处理 & JSONL导出 ===============

def generate_dataset_jsonl(
    questions: List[str],
    output_path: str,
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
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, q in enumerate(questions):
            flat = run_pipeline_for_question(q, verbose=verbose)

            # 构建 messages
            json_record = {
                "id": idx,
                "messages": [
                    {
                        "role": "user",
                        "content": q.strip()
                    },
                    {
                        "role": "assistant",
                        "content": flat["flattened_text"].strip()
                    }
                ]
            }

            f.write(json.dumps(json_record, ensure_ascii=False) + "\n")
            f.flush()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent dataset generator demo.")
    parser.add_argument("--output", default="multi_agent_sft_dataset.jsonl", help="Destination JSONL path")
    parser.add_argument("--verbose", action="store_true", help="Print per-agent progress")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # 一个小demo：你可以换成自己的问题列表（HotpotQA / ASearcher等）
    demo_questions = [
        # "Which two teams did the head coach of the 2007 San Diego State Aztecs play for professionally?",
        "United States Air Force 432nd Wing location",
    ]

    generate_dataset_jsonl(
        questions=demo_questions,
        output_path=args.output,
        verbose=args.verbose,
    )

    print(f"Done. Dataset written to {args.output}")
