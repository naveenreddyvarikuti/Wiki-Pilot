from langchain_core.documents import Document


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:

    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc.page_content
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [doc_map[doc_id] for doc_id, _ in sorted_docs]


def hybrid_retrieve(
    query: str,
    bm25_retriever,
    dense_retriever,
    k: int = 5,
    rrf_k: int = 60,
) -> list[Document]:

    fetch_k = k * 3

    bm25_retriever.k = fetch_k
    bm25_results = bm25_retriever.invoke(query)

    dense_results = dense_retriever.invoke(query)

    fused = reciprocal_rank_fusion([bm25_results, dense_results], k=rrf_k)

    return fused[:k]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from pathlib import Path
    from src.ingest.parser import parse_file
    from src.ingest.chunker import chunk_sections
    from src.index.bm25_store import build_bm25_retriever
    from src.index.vector_store import build_vector_store, get_dense_retriever

    test_file = Path("data/processed/raw_md/content/handbook/engineering/ai/agent-foundations/_index.md")
    sections = parse_file(test_file, base_dir=Path("data/processed/raw_md"))
    chunks = chunk_sections(sections)

    bm25_retriever = build_bm25_retriever(chunks, k=15)
    vector_store = build_vector_store(chunks)
    dense_retriever = get_dense_retriever(vector_store, k=15)

    queries = ["goalkeeper rotation", "how does the team communicate", "gitlab-lsp"]
    for q in queries:
        print(f"Query: '{q}'")
        results = hybrid_retrieve(q, bm25_retriever, dense_retriever, k=3)
        for i, doc in enumerate(results, 1):
            print(f"  {i}. [{doc.metadata['heading']}] {doc.page_content[:80]}...")
        print()
