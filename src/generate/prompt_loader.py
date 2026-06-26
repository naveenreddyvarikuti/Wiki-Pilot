from pathlib import Path
import yaml
from langchain_core.documents import Document

PROMPTS_DIR = Path("prompts/rag_qa")


def load_prompt(version: str = "active") -> dict:
    path = PROMPTS_DIR / f"{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt version not found: {path}")

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_sources(sources: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(sources, 1):
        hierarchy = doc.metadata.get("heading_hierarchy", "Unknown")
        blocks.append(f"[Source {i}] {hierarchy}\n{doc.page_content}")
    return "\n\n".join(blocks)


def format_prompt(prompt_config: dict, query: str, sources: list[Document]) -> str:
    system = prompt_config["system"].strip()
    template = prompt_config["template"]
    sources_text = format_sources(sources)

    return template.format(
        system=system,
        sources=sources_text,
        query=query,
    )


def list_versions() -> list[str]:
    return sorted(
        p.stem for p in PROMPTS_DIR.glob("*.yaml")
    )
