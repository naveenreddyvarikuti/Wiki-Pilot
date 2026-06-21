from pathlib import Path
from langchain_community.llms import LlamaCpp
from langchain_core.documents import Document

MODEL_PATH = str(Path("models/qwen2.5-7b-instruct-q3_k_m.gguf"))

SYSTEM_PROMPT = """You are a helpful assistant answering questions about the GitLab Handbook.

Rules:
1. Use ONLY the provided sources to answer. Do not use outside knowledge.
2. Cite sources using [Source N] for every claim you make.
3. If the sources don't contain the answer, say "I don't have enough information to answer this question."
4. Be concise and direct."""


def build_llm() -> LlamaCpp:
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=8,
        temperature=0.1,
        max_tokens=512,
        verbose=False,
    )


def _build_prompt(query: str, sources: list[Document]) -> str:
    source_blocks = []
    for i, doc in enumerate(sources, 1):
        hierarchy = doc.metadata.get("heading_hierarchy", "Unknown")
        source_blocks.append(f"[Source {i}] {hierarchy}\n{doc.page_content}")

    sources_text = "\n\n".join(source_blocks)

    return f"""{SYSTEM_PROMPT}

Sources:
{sources_text}

Question: {query}

Answer:"""


def generate_answer(query: str, sources: list[Document], llm: LlamaCpp) -> str:
    prompt = _build_prompt(query, sources)
    return llm.invoke(prompt)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from src.ingest.parser import parse_file
    from src.ingest.chunker import chunk_sections
    from src.index.bm25_store import build_bm25_retriever
    from src.index.vector_store import build_vector_store, get_dense_retriever
    from src.retrieve.hybrid import hybrid_retrieve
    from src.retrieve.rerank import build_reranker, rerank

    test_file = Path("data/processed/raw_md/content/handbook/engineering/ai/agent-foundations/_index.md")
    sections = parse_file(test_file, base_dir=Path("data/processed/raw_md"))
    chunks = chunk_sections(sections)

    print("Building indexes...")
    bm25_retriever = build_bm25_retriever(chunks, k=15)
    vector_store = build_vector_store(chunks)
    dense_retriever = get_dense_retriever(vector_store, k=15)
    reranker = build_reranker()
    llm = build_llm()

    query = "How does the Agent Foundations team handle incoming requests?"
    print(f"\nQuery: {query}\n")

    hybrid_results = hybrid_retrieve(query, bm25_retriever, dense_retriever, k=10)
    reranked = rerank(query, hybrid_results, reranker, top_k=5)

    print("Sources used:")
    for i, doc in enumerate(reranked, 1):
        print(f"  [Source {i}] {doc.metadata['heading']}")

    print("\nGenerating answer...\n")
    answer = generate_answer(query, reranked, llm)
    print(answer)
