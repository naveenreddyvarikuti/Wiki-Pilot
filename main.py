import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from src.ingest.parser import parse_directory
from src.ingest.chunker import chunk_sections
from src.index.bm25_store import build_bm25_retriever
from src.index.vector_store import build_vector_store, get_dense_retriever
from src.retrieve.hybrid import hybrid_retrieve
from src.retrieve.rerank import build_reranker, rerank
from src.generate.llm import build_llm, generate_answer

DATA_DIR = Path("data/processed/raw_md")


def build_pipeline():
    print("[1/6] Parsing markdown files...")
    sections = parse_directory(DATA_DIR)
    print(f"       {len(sections)} sections from {DATA_DIR}")

    print("[2/6] Chunking sections...")
    chunks = chunk_sections(sections)
    print(f"       {len(chunks)} chunks")

    print("[3/6] Building BM25 index...")
    bm25_retriever = build_bm25_retriever(chunks, k=15)

    print("[4/6] Building dense index (embedding chunks)...")
    vector_store = build_vector_store(chunks)
    dense_retriever = get_dense_retriever(vector_store, k=15)

    print("[5/6] Loading reranker...")
    reranker = build_reranker()

    print("[6/6] Loading LLM...")
    llm = build_llm()

    print("\nPipeline ready.\n")
    return bm25_retriever, dense_retriever, reranker, llm


def ask(query, bm25_retriever, dense_retriever, reranker, llm):
    hybrid_results = hybrid_retrieve(query, bm25_retriever, dense_retriever, k=10)
    reranked_docs = rerank(query, hybrid_results, reranker, top_k=5)

    print("\nSources:")
    for i, doc in enumerate(reranked_docs, 1):
        heading = doc.metadata.get("heading", "Unknown")
        source = doc.metadata.get("source_file", "")
        print(f"  [Source {i}] {heading} ({source})")

    print("\nAnswer:")
    answer = generate_answer(query, reranked_docs, llm)
    print(answer)
    print()


def main():
    bm25_retriever, dense_retriever, reranker, llm = build_pipeline()

    print("Type your question (or 'quit' to exit):\n")
    while True:
        try:
            query = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        ask(query, bm25_retriever, dense_retriever, reranker, llm)


if __name__ == "__main__":
    main()
