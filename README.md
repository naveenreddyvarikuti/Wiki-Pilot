# WikiPilot

Production-grade RAG system over the GitLab Handbook.

Demonstrates: hybrid retrieval (BM25 + dense), RRF fusion, cross-encoder reranking,
citation enforcement, prompt versioning, and threshold-based eval gating.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Structure

```
data/          raw handbook clone + processed chunks
src/ingest/    clone, parse, chunk
src/index/     BM25 + Qdrant vector store
src/retrieve/  hybrid retrieval + RRF fusion + reranking
src/generate/  local LLM + prompt versioning + citation validation
src/eval/      golden set + Ragas + CI gate
prompts/       versioned YAML prompt templates
ui/            Streamlit demo
```
