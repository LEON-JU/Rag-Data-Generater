"""Elasticsearch helpers copied from the Agentic-RAG project.

This module centralises all logic required to talk to the local
Elasticsearch instance, including:

* client creation with environment-based credentials
* simple semantic search helpers
* bulk indexing utilities that mirror the original es_wiki_build.py script
* sanity-check utilities that expose the original es_wiki_test.py behaviour
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import pandas as pd
try:  # pragma: no cover - optional dependency
    from elasticsearch import Elasticsearch, helpers
except ImportError:  # pragma: no cover
    Elasticsearch = None  # type: ignore
    helpers = None  # type: ignore

DEFAULT_ELASTIC_URL = "http://localhost:9200"
DEFAULT_ELASTIC_USERNAME = "elastic"


class MissingElasticsearchPassword(RuntimeError):
    """Raised when no password is provided via the environment."""


def create_es_client(
    url: str | None = None,
    username: str | None = None,
    password: str | None = None,
    verify_certs: bool = False,
) -> Elasticsearch:
    """Build a configured Elasticsearch client.

    Args:
        url: Endpoint for the Elasticsearch cluster.
        username: Authentication user (defaults to ``elastic``).
        password: Authentication password. Required.
        verify_certs: Whether to verify TLS certificates.
    """

    # Resolve credentials lazily so runtime env overrides (e.g. from configs) take effect.
    url = url or os.environ.get("ELASTIC_URL", DEFAULT_ELASTIC_URL)
    username = username or os.environ.get("ELASTIC_USERNAME", DEFAULT_ELASTIC_USERNAME)
    password = password or os.environ.get("ELASTIC_PASSWORD") or os.environ.get("ELASTIC_SEARCH_PASSWORD")
    if not password:
        raise MissingElasticsearchPassword(
            "Set ELASTIC_PASSWORD or ELASTIC_SEARCH_PASSWORD before using Elasticsearch tools."
        )
    if Elasticsearch is None:  # pragma: no cover - triggered when dependency missing
        raise RuntimeError("Install the 'elasticsearch' package to enable ES-backed search.")

    return Elasticsearch(
        url,
        basic_auth=(username, password),
        verify_certs=verify_certs,
        ssl_show_warn=False,
    )


def semantic_search(query: str, index_name: str, num_results: int = 10, client: Optional[Elasticsearch] = None) -> List[Dict]:
    """Run a multi-field query over ``title`` and ``text`` fields."""

    close_client = False
    if client is None:
        client = create_es_client()
        close_client = True
    try:
        search_body = {
            "query": {"multi_match": {"query": query, "fields": ["title", "text"]}},
            "size": num_results,
        }
        response = client.search(index=index_name, body=search_body)
        hits = response.get("hits", {}).get("hits", [])
        return [hit.get("_source", {}) for hit in hits]
    finally:
        if close_client:
            client.close()


def _iter_parquet_rows(path: Path, index_name: str) -> Iterator[Dict]:
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        payload = row.to_dict()
        payload["_index"] = index_name
        yield payload


def bulk_index_parquet(
    *,
    language: str,
    corpus_pattern: str | None = None,
    index_name: str | None = None,
    client: Optional[Elasticsearch] = None,
) -> None:
    """Mimics ``ArtSearch/es_wiki_build.py`` for convenience."""

    index_name = index_name or f"wiki_{language}"
    corpus_pattern = corpus_pattern or f"data/wikipedia/20231101.{language}/*.parquet"
    files = sorted(glob.glob(corpus_pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found with pattern: {corpus_pattern}")
    if helpers is None:  # pragma: no cover
        raise RuntimeError("Install the 'elasticsearch' package to run bulk indexing.")

    close_client = False
    if client is None:
        client = create_es_client()
        close_client = True

    try:
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name, ignore_unavailable=True)
        client.indices.create(index=index_name)

        for file_path in files:
            actions = _iter_parquet_rows(Path(file_path), index_name)
            helpers.bulk(client=client, actions=actions)
    finally:
        if close_client:
            client.close()


def snapshot_index(index_name: str, sample_size: int = 2, client: Optional[Elasticsearch] = None) -> Dict[str, List[Dict]]:
    """Return document counts and a few sample entries for quick debugging."""

    close_client = False
    if client is None:
        client = create_es_client()
        close_client = True

    try:
        count = client.count(index=index_name).get("count", 0)
        hits = client.search(index=index_name, size=sample_size).get("hits", {}).get("hits", [])
        return {"count": count, "documents": [hit.get("_source", {}) for hit in hits]}
    finally:
        if close_client:
            client.close()


__all__ = [
    "create_es_client",
    "semantic_search",
    "bulk_index_parquet",
    "snapshot_index",
    "MissingElasticsearchPassword",
]
