import re
from dataclasses import dataclass
from src.ingest.parser import Section


CHUNK_TARGET_TOKENS = 256
TOKENS_PER_CHAR = 0.25  # 1 token ≈ 4 chars → 1 char ≈ 0.25 tokens
OVERLAP_SENTENCES = 2


@dataclass
class Chunk:
    text: str
    heading: str
    heading_hierarchy: list[str]
    source_file: str
    chunk_index: int  # position within the section (0, 1, 2, ...)


def _estimate_tokens(text: str) -> int:
    return int(len(text) * TOKENS_PER_CHAR)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Handles common abbreviations and edge cases."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _build_chunk_text(heading_hierarchy: list[str], body: str) -> str:
    """Prepend the heading hierarchy to chunk body for richer embeddings."""
    prefix = " > ".join(heading_hierarchy)
    return f"{prefix}\n\n{body}"


def chunk_section(section: Section) -> list[Chunk]:
    """
    Convert a single Section into one or more Chunks.

    - If the section fits within the token budget, return it as one chunk.
    - Otherwise, split at sentence boundaries with overlap.
    """
    full_text = _build_chunk_text(section.heading_hierarchy, section.content)

    if _estimate_tokens(full_text) <= CHUNK_TARGET_TOKENS:
        return [Chunk(
            text=full_text,
            heading=section.heading,
            heading_hierarchy=list(section.heading_hierarchy),
            source_file=section.source_file,
            chunk_index=0,
        )]

    sentences = _split_sentences(section.content)

    if not sentences:
        return []

    chunks = []
    start = 0

    while start < len(sentences):
        current_sentences = []
        i = start

        while i < len(sentences):
            current_sentences.append(sentences[i])
            body = " ".join(current_sentences)
            candidate = _build_chunk_text(section.heading_hierarchy, body)

            if _estimate_tokens(candidate) >= CHUNK_TARGET_TOKENS:
                break
            i += 1

        body = " ".join(current_sentences)
        chunk_text = _build_chunk_text(section.heading_hierarchy, body)

        chunks.append(Chunk(
            text=chunk_text,
            heading=section.heading,
            heading_hierarchy=list(section.heading_hierarchy),
            source_file=section.source_file,
            chunk_index=len(chunks),
        ))

        next_start = i + 1 - OVERLAP_SENTENCES
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def chunk_sections(sections: list[Section]) -> list[Chunk]:
    """Convert a list of Sections into Chunks."""
    all_chunks = []
    for section in sections:
        all_chunks.extend(chunk_section(section))
    return all_chunks


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from pathlib import Path
    from src.ingest.parser import parse_file

    test_file = Path("data/processed/raw_md/content/handbook/engineering/ai/agent-foundations/_index.md")
    sections = parse_file(test_file, base_dir=Path("data/processed/raw_md"))
    chunks = chunk_sections(sections)

    print(f"Sections: {len(sections)} → Chunks: {len(chunks)}\n")
    for c in chunks[:5]:
        tokens = _estimate_tokens(c.text)
        print(f"[chunk {c.chunk_index}] {c.heading} (~{tokens} tokens)")
        print(f"  {c.text[:120]}...")
        print()
