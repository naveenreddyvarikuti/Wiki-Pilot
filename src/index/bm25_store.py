
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.ingest.chunker import Chunk


def chunks_to_documents(chunks: list[Chunk]) -> list[Document]:
    """Convert our Chunks into LangChain Documents with metadata."""
    docs = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk.text,
            metadata={
                "heading": chunk.heading,
                "heading_hierarchy": " > ".join(chunk.heading_hierarchy),
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
            },
        )
        docs.append(doc)
    return docs


def build_bm25_retriever(chunks: list[Chunk], k: int = 5) -> BM25Retriever:
    """Build a LangChain BM25Retriever from our Chunks."""
    docs = chunks_to_documents(chunks)
    retriever = BM25Retriever.from_documents(docs, k=k)
    return retriever


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from pathlib import Path
    from src.ingest.parser import parse_file
    from src.ingest.chunker import chunk_sections

    test_file = Path("data/processed/raw_md/content/handbook/engineering/ai/agent-foundations/_index.md")
    sections = parse_file(test_file, base_dir=Path("data/processed/raw_md"))
    chunks = chunk_sections(sections)

    retriever = build_bm25_retriever(chunks, k=3)

    queries = ["goalkeeper rotation", "retrospective", "slack channel"]
    for q in queries:
        print(f"Query: '{q}'")
        results = retriever.invoke(q)
        for doc in results:
            print(f"  [{doc.metadata['heading']}] {doc.page_content[:80]}...")
        print()
