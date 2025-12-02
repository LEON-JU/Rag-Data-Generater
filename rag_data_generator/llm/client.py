"""LLM client abstractions used by the orchestration pipeline."""

from __future__ import annotations

import os
from typing import Iterable, List, Protocol

import requests

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover - provide nicer error later
    OpenAI = None


class ChatClient(Protocol):  # pragma: no cover - structural typing helper
    def complete(self, messages: List[dict], **kwargs) -> str:
        ...


class OpenAILLMClient:
    def __init__(
        self,
        model: str | None = "/home/juyiang/data/llm_models/qwen3-32b",
        api_key: str | None = "none",
        base_url: str | None = 'http://localhost:8000/v1',
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is required for OpenAILLMClient but is not installed.")
        self.model = model or os.getenv("GENERATOR_LLM_MODEL") or os.getenv("EVAL_LLM_MODEL_NAME")
        api_key = api_key or os.getenv("GENERATOR_LLM_API_KEY")
        base_url = base_url or os.getenv("GENERATOR_LLM_BASE_URL")
        if not self.model:
            raise RuntimeError("No model name configured for the generator LLM")
        if not api_key:
            raise RuntimeError("Set GENERATOR_LLM_API_KEY before generating data")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.last_usage: dict | None = None

    def complete(self, messages: List[dict], **kwargs) -> str:
        temperature = kwargs.get("temperature", 0.7)
        response = self._client.chat.completions.create(model=self.model, messages=messages, temperature=temperature)
        usage = getattr(response, "usage", None)
        if usage is not None:
            try:
                usage = usage.model_dump()
            except AttributeError:
                usage = dict(usage)
        self.last_usage = usage
        return response.choices[0].message.content or ""


class SiliconFlowLLMClient:
    """Minimal client for SiliconFlow's OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model: str | None = 'Qwen/Qwen2.5-72B-Instruct-128K',
        api_key: str | None = 'sk-kiwmknhfsbjtmqojloxxoyhxnbkhrmqnhpcwxwzigpmyczto',
        base_url: str | None = 'https://api.siliconflow.cn/v1',
        timeout: int = 60,
    ) -> None:
        self.model = model or os.getenv("SILICONFLOW_LLM_MODEL")
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        self.base_url = base_url or os.getenv("SILICONFLOW_API_BASE")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("Set SILICONFLOW_API_KEY before using SiliconFlowLLMClient.")
        self.last_usage: dict | None = None

    def complete(self, messages: List[dict], **kwargs) -> str:
        temperature = kwargs.get("temperature", 0.7)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - external service reported error
            detail = exc.response.text if exc.response is not None else str(exc)
            raise RuntimeError(f"SiliconFlow request failed: {detail}") from exc
        data = response.json()
        self.last_usage = data.get("usage")
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"No completion returned from SiliconFlow: {data}")
        return choices[0].get("message", {}).get("content", "")


__all__ = ["ChatClient", "OpenAILLMClient", "SiliconFlowLLMClient"]
