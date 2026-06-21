from dataclasses import dataclass, field
from pathlib import Path

import frontmatter
import mistune


@dataclass
class Section:
    heading: str
    heading_hierarchy: list[str]
    level: int
    content: str
    source_file: str


def _extract_text(tokens: list[dict]) -> str:
    """Recursively extract plain text from mistune AST tokens."""
    parts = []
    for tok in tokens:
        if tok["type"] == "text":
            parts.append(tok["raw"])
        elif tok["type"] == "codespan":
            parts.append(tok["raw"])
        elif tok["type"] == "code_block":
            parts.append(tok["raw"])
        elif "children" in tok:
            parts.append(_extract_text(tok["children"]))
        elif "raw" in tok:
            parts.append(tok["raw"])
    return "".join(parts)


def _walk_tokens(tokens: list[dict]) -> list[tuple[int, str, str]]:
    """
    Walk the AST and yield (level, heading_text, body_text) for each section.

    A 'section' is a heading plus all content until the next heading.
    Level 0 means content before the first heading (preamble).
    """
    sections: list[tuple[int, str, str]] = []
    current_level = 0
    current_heading = ""
    current_body_parts: list[str] = []

    for tok in tokens:
        if tok["type"] == "heading":
            sections.append((current_level, current_heading, "\n".join(current_body_parts).strip()))
            current_level = tok["attrs"]["level"]
            current_heading = _extract_text(tok["children"])
            current_body_parts = []
        elif tok["type"] == "code_block":
            current_body_parts.append(tok["raw"])
        elif "children" in tok:
            current_body_parts.append(_extract_text(tok["children"]))
        elif "raw" in tok:
            current_body_parts.append(tok["raw"])

    sections.append((current_level, current_heading, "\n".join(current_body_parts).strip()))
    return sections


def _build_hierarchy(raw_sections: list[tuple[int, str, str]], title: str) -> list[dict]:
    """
    Convert flat (level, heading, content) tuples into Sections with heading hierarchy.

    Uses the heading stack approach: when encountering level N,
    pop everything >= N from the stack, then push.
    """
    stack: list[tuple[int, str]] = [(0, title)]
    result = []

    for level, heading, content in raw_sections:
        if level == 0:
            if content:
                result.append({
                    "heading": title,
                    "heading_hierarchy": [title],
                    "level": 0,
                    "content": content,
                })
            continue

        while stack and stack[-1][0] >= level:
            stack.pop()

        stack.append((level, heading))

        hierarchy = [name for _, name in stack]

        if content:
            result.append({
                "heading": heading,
                "heading_hierarchy": hierarchy,
                "level": level,
                "content": content,
            })

    return result


def parse_file(file_path: Path, base_dir: Path | None = None) -> list[Section]:
    """
    Parse a single markdown file into a list of Sections.

    Args:
        file_path: Path to the markdown file.
        base_dir: If provided, source_file is stored relative to this directory.
    """
    raw_text = file_path.read_text(encoding="utf-8")

    post = frontmatter.loads(raw_text)
    title = post.metadata.get("title", file_path.stem)
    body = post.content

    md = mistune.create_markdown(renderer="ast")
    tokens = md(body)

    raw_sections = _walk_tokens(tokens)

    sections_dicts = _build_hierarchy(raw_sections, title)

    source = str(file_path.relative_to(base_dir)) if base_dir else str(file_path)

    return [
        Section(
            heading=s["heading"],
            heading_hierarchy=list(s["heading_hierarchy"]),
            level=s["level"],
            content=s["content"],
            source_file=source,
        )
        for s in sections_dicts
    ]


def parse_directory(dir_path: Path) -> list[Section]:
    """Parse all markdown files in a directory tree."""
    all_sections = []
    md_files = sorted(dir_path.rglob("*.md"))
    for f in md_files:
        all_sections.extend(parse_file(f, base_dir=dir_path))
    return all_sections


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    test_file = Path("data/processed/raw_md/content/handbook/engineering/ai/agent-foundations/_index.md")
    sections = parse_file(test_file, base_dir=Path("data/processed/raw_md"))

    print(f"Parsed {len(sections)} sections from {test_file.name}\n")
    for s in sections[:5]:
        hierarchy_str = " > ".join(s.heading_hierarchy)
        print(f"[L{s.level}] {hierarchy_str}")
        print(f"     {s.content[:100]}...")
        print()
