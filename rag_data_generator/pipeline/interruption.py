"""Tag detection + interruption loop for tool-calling models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from rag_data_generator.llm.client import ChatClient
from rag_data_generator.tooling.registry import ToolRegistry


@dataclass
class ToolCall:
    raw_block: str
    plugin_name: str
    plugin_args: str


class SearchTagDetector:
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)

    def extract(self, text: str) -> List[str]:
        return [match.group(1).strip() for match in self.pattern.finditer(text)]

    def matches(self, text: str) -> List[re.Match]:
        return list(self.pattern.finditer(text))

    def first(self, text: str) -> re.Match | None:
        return self.pattern.search(text)


class ToolCallParser:
    tool_pattern = re.compile(r"\[(.*?)\]:\s*(?:(?:\"(.*?)\")|(.*))", re.DOTALL)

    def parse(self, block: str) -> ToolCall:
        match = self.tool_pattern.match(block.strip())
        if match:
            name = re.sub(r"[^a-zA-Z_]", "", match.group(1))
            payload = match.group(2) or match.group(3) or ""
        else:
            name = "Wiki_RAG"
            payload = block
        return ToolCall(raw_block=block, plugin_name=name, plugin_args=payload.strip())


class InterruptionOrchestrator:
    def __init__(
        self,
        llm_client: ChatClient,
        tool_registry: ToolRegistry,
        max_rounds: int = 4,
        observation_template: str = "<observation>{}</observation>",
        debug: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.max_rounds = max_rounds
        self.observation_template = observation_template
        self.detector = SearchTagDetector()
        self.parser = ToolCallParser()
        self.debug = debug

    def run(self, messages: List[dict]) -> dict:
        history = list(messages)
        final_response = None

        for round_idx in range(self.max_rounds):

            # Step 1 — LLM生成
            assistant_output = self.llm_client.complete(history)
            text = assistant_output or ""

            # Step 2 — 查找 <search>...</search>
            match = self.detector.first(text)          # = StoppingCriteria equivalent

            if match and round_idx < self.max_rounds - 1:
                inner = match.group(1).strip()
                call = self.parser.parse(inner)
                result = self.tool_registry.invoke(call.plugin_name, call.plugin_args)
                obs_text = self.observation_template.format(result.to_observation())

                prefix = text[: match.end()]
                if not prefix.endswith("\n"):
                    prefix = f"{prefix}\n"
                merged = f"{prefix}{obs_text}\n"

                # 写回历史，相当于 tokenizer.encode → 继续生成的 prompt
                history.append({
                    "role": "assistant",
                    "content": merged
                })
                final_response = merged
                continue

            # Step 3 — 如果包含 </answer> → 截断并停止
            if "</answer>" in text:
                end = text.index("</answer>") + len("</answer>")
                final_response = text[:end].strip()

                history.append({
                    "role": "assistant",
                    "content": final_response
                })
                break

            # 没有 search 或无法继续 → 直接写入
            history.append({"role": "assistant", "content": text})
            final_response = text

        return {"history": history, "response": final_response}



__all__ = [
    "ToolCall",
    "SearchTagDetector",
    "ToolCallParser",
    "InterruptionOrchestrator",
]
