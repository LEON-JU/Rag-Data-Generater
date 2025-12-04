"""Built-in tool implementations."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

import numpy as np

from rag_data_generator.search.wiki import WikiSearcher, create_wiki_searcher

try:  # pragma: no cover
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None


@dataclass
class ToolSpec:
    name_for_human: str
    name_for_model: str
    description_for_model: str
    parameters: List[Dict[str, Any]]


class Tool(Protocol):
    spec: ToolSpec

    def invoke(self, payload: Dict[str, Any]) -> Any:  # pragma: no cover - interface definition
        ...


class WikiRAGTool:
    def __init__(
        self,
        language: str = "en",
        spec: ToolSpec | None = None,
        default_size: int = 3,
        max_sentences: int = 5,
        max_candidates: int = 256,
        embed_model: str | None = None,
    ) -> None:
        spec = spec or ToolSpec(
            name_for_human="维基百科知识检索模块",
            name_for_model="Wiki_RAG",
            description_for_model="通过 Elasticsearch 查询百科知识并返回与实体最相关的段落。",
            parameters=[
                {
                    "name": "input",
                    "description": "规范化的实体或问题",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        )
        self.spec = spec
        self.language = language
        self.default_size = default_size
        self.max_sentences = max_sentences
        self.max_candidates = max_candidates
        self.embed_model = embed_model or os.getenv(
            "WIKI_RAG_EMBED_MODEL", "sentence-transformers/paraphrase-MiniLM-L6-v2"
        )
        self._searcher: WikiSearcher | None = None
        self._embedder: SentenceTransformer | None = None

    def _ensure_searcher(self) -> WikiSearcher:
        if self._searcher is None:
            self._searcher = create_wiki_searcher(self.language)
        return self._searcher

    def invoke(self, payload: Dict[str, Any]) -> Any:
        query = str(payload.get("input", "")).strip()
        if not query:
            raise ValueError("Wiki_RAG requires a non-empty 'input'.")
        size = payload.get("size")
        size = int(size) if size is not None else self.default_size
        size = max(1, min(size, self.default_size))
        hits = self._ensure_searcher().search(query, size=size)
        if hits:
            hits = [hits[0]]  # only keep the top ranked document
        refined = hits
        return json.dumps(refined, ensure_ascii=False)

    def _ensure_embedder(self) -> SentenceTransformer:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is required for Wiki_RAG but is not installed.")
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embed_model)
        return self._embedder

    def _split_sentences(self, text: str) -> List[str]:
        # Lightweight splitter: split on common sentence punctuation, avoid heavy NLP pipelines
        chunks = re.split(r"(?<=[\\.?!])\s+", text)
        sentences = [chunk.strip() for chunk in chunks if chunk.strip()]
        if not sentences:
            return [text.strip()]
        return sentences

    def _select_sentences(self, query: str, hits: List[dict]) -> List[dict]:
        candidates: List[dict] = []
        for doc in hits:
            title = doc.get("title", "")
            text = doc.get("text", "")
            for sent in self._split_sentences(text):
                candidates.append({"title": title, "sentence": sent})
                if len(candidates) >= self.max_candidates:
                    break
            if len(candidates) >= self.max_candidates:
                break
        if not candidates:
            return hits
        embedder = self._ensure_embedder()
        sentences = [c["sentence"] for c in candidates]
        query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sent_vecs = embedder.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
        scores = np.dot(sent_vecs, query_vec)
        top_k = min(self.max_sentences, len(candidates))
        top_indices = np.argsort(-scores)[:top_k]
        results = []
        for i in top_indices:
            results.append({"title": candidates[i]["title"], "sentence": candidates[i]["sentence"]})
        return results


DEFAULT_TOOL_SPECS = [
    ToolSpec(
        name_for_human="维基百科知识检索模块",
        name_for_model="Wiki_RAG",
        description_for_model="通过 Elasticsearch 查询百科文档并返回 title 和 text 字段。",
        parameters=[
            {
                "name": "input",
                "description": "需要检索的实体或问题",
                "required": True,
                "schema": {"type": "string"},
            }
        ],
    ),
]


def build_default_tools() -> List[Tool]:
    return [WikiRAGTool(language="en", spec=DEFAULT_TOOL_SPECS[0], default_size=3)]


__all__ = [
    "Tool",
    "ToolSpec",
    "WikiRAGTool",
    "DEFAULT_TOOL_SPECS",
    "build_default_tools",
]
