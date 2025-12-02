"""Wiki search helper that wraps an Elasticsearch backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

try:  # pragma: no cover
    from elasticsearch import Elasticsearch
except ImportError:  # pragma: no cover
    Elasticsearch = None  # type: ignore

from .elastic import create_es_client


@dataclass
class WikiSearcher:
    index_name: str
    client: Elasticsearch

    def search(self, query: str, size: int = 1) -> List[dict]:
        body = {"query": {"multi_match": {"query": query, "fields": ["title", "text"]}}, "size": size}
        response = self.client.search(index=self.index_name, body=body)
        hits = response.get("hits", {}).get("hits", [])
        return [hit.get("_source", {}) for hit in hits]

    def close(self) -> None:
        self.client.close()


def create_wiki_searcher(language: str = "en", url: Optional[str] = None) -> WikiSearcher:
    """Factory used by the tool registry. Mirrors the original implementation."""

    index_name = f"wiki_{language.lower()}"
    if Elasticsearch is None:
        raise RuntimeError("Install the 'elasticsearch' package to use Wiki_RAG.")
    client = create_es_client(url=url)
    return WikiSearcher(index_name=index_name, client=client)


__all__ = ["WikiSearcher", "create_wiki_searcher"]
