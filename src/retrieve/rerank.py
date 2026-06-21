

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

RERANKER_MODEL = "BAAI/bge-reranker-base"


def build_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL, device="cpu")


def rerank(
    query: str,
    candidates: list[Document],
    reranker: CrossEncoder,
    top_k: int = 5,
) -> list[Document]:

    if not candidates:
        return []

    pairs = [(query, doc.page_content) for doc in candidates]

    scores = reranker.predict(pairs)

    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored[:top_k]]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from pathlib import Path
    from src.ingest.parser import parse_file
    from src.ingest.chunker import chunk_sections
    from src.index.bm25_store import build_bm25_retriever
    from src.index.vector_store import build_vector_store, get_dense_retriever
    from src.retrieve.hybrid import hybrid_retrieve

    test_file = Path("data/processed/raw_md/content/handbook/engineering/ai/agent-foundations/_index.md")
    sections = parse_file(test_file, base_dir=Path("data/processed/raw_md"))
    chunks = chunk_sections(sections)

    bm25_retriever = build_bm25_retriever(chunks, k=15)
    vector_store = build_vector_store(chunks)
    dense_retriever = get_dense_retriever(vector_store, k=15)

    reranker = build_reranker()

    query = "how does the team handle incoming requests"
    print(f"Query: '{query}'\n")

    hybrid_results = hybrid_retrieve(query, bm25_retriever, dense_retriever, k=10)
    print("Before reranking:")
    for i, doc in enumerate(hybrid_results[:5], 1):
        print(f"  {i}. [{doc.metadata['heading']}]")

    reranked = rerank(query, hybrid_results, reranker, top_k=5)
    print("\nAfter reranking:")
    for i, doc in enumerate(reranked, 1):
        print(f"  {i}. [{doc.metadata['heading']}]")
