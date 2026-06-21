from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from src.ingest.chunker import Chunk
from src.index.bm25_store import chunks_to_documents

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
COLLECTION_NAME = "handbook_chunks"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def build_vector_store(chunks: list[Chunk]) -> QdrantVectorStore:
    """Embed chunks and store in a local in-memory Qdrant instance."""
    docs = chunks_to_documents(chunks)
    embeddings = get_embeddings()

    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        location=":memory:",
        collection_name=COLLECTION_NAME,
    )

    return vector_store


def get_dense_retriever(vector_store: QdrantVectorStore, k: int = 5):
    """Get a LangChain retriever from the vector store."""
    return vector_store.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from pathlib import Path
    from src.ingest.parser import parse_file
    from src.ingest.chunker import chunk_sections

    test_file = Path("data/processed/raw_md/content/handbook/engineering/ai/agent-foundations/_index.md")
    sections = parse_file(test_file, base_dir=Path("data/processed/raw_md"))
    chunks = chunk_sections(sections)

    print(f"Building vector store from {len(chunks)} chunks...")
    vector_store = build_vector_store(chunks)
    retriever = get_dense_retriever(vector_store, k=3)

    queries = ["goalkeeper rotation", "how does the team communicate", "time off policy"]
    for q in queries:
        print(f"\nQuery: '{q}'")
        results = retriever.invoke(q)
        for doc in results:
            print(f"  [{doc.metadata['heading']}] {doc.page_content[:80]}...")
