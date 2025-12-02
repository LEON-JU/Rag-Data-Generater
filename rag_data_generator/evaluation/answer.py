"""LLM-based answer evaluation helper."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Tuple

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover - provide nicer error if used
    OpenAI = None

from rag_data_generator.prompts.prompts import LLM_EVAL_PROMPT


class EvalClient(Protocol):  # pragma: no cover - typing helper
    def complete(self, prompt: str) -> str:
        ...


class OpenAIEvalClient:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is required for OpenAIEvalClient but is not installed.")
        model = model or os.getenv("EVAL_LLM_MODEL_NAME")
        api_key = api_key or os.getenv("EVAL_LLM_API_KEY")
        base_url = base_url or os.getenv("EVAL_LLM_BASE_URL")
        if not model:
            raise RuntimeError("Set EVAL_LLM_MODEL_NAME for evaluation.")
        if not api_key:
            raise RuntimeError("Set EVAL_LLM_API_KEY for evaluation.")
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content or ""


@dataclass
class EvaluationItem:
    question: str
    expected: str
    predicted: str


class AnswerEvaluator:
    def __init__(self, prompt_template: str = LLM_EVAL_PROMPT, client: Optional[EvalClient] = None) -> None:
        self.prompt_template = prompt_template
        self.client = client or OpenAIEvalClient()

    def evaluate(self, items: Iterable[EvaluationItem]) -> Tuple[int, int, float, List[dict]]:
        correct = 0
        total = 0
        records: List[dict] = []
        for item in items:
            prompt = self.prompt_template.format(
                question=item.question,
                expected=item.expected,
                predicted=item.predicted,
            )
            response = self.client.complete(prompt)
            is_correct = response.strip() == "Yes"
            total += 1
            if is_correct:
                correct += 1
            records.append(
                {
                    "question": item.question,
                    "expected": item.expected,
                    "predicted": item.predicted,
                    "eval": "true" if is_correct else "false",
                }
            )
        accuracy = correct / total if total else 0.0
        return correct, total, accuracy, records


__all__ = ["AnswerEvaluator", "EvaluationItem", "OpenAIEvalClient"]
