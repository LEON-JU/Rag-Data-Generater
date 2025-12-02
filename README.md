# Rag-Data-Generater

This repository focuses on building tool-augmented data generation pipelines for Retrieval-Augmented Generation (RAG) experiments. The goal is to produce high-quality prompts, model responses, and auxiliary supervision signals without coupling the logic to any specific training loop.

The code here extracts and repackages the utility pieces from the Agentic-RAG-R1 project:

- Elasticsearch-backed wiki/document search helpers
- Prompt templates for agent-style tool calling
- Tool registries with deterministic plugin triggering
- LLM-based answer evaluation helpers

See `docs/data_formats.md` for the currently supported dataset descriptions.
