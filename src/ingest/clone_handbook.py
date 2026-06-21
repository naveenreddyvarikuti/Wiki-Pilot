"""
Clone the GitLab Handbook repo and filter to engineering + people sections.

Usage:
    python -m src.ingest.clone_handbook
"""

import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://gitlab.com/gitlab-com/content-sites/handbook.git"
RAW_DIR = Path("data/raw/handbook")
FILTERED_DIR = Path("data/processed/raw_md")

SECTIONS_TO_KEEP = [
    "content/handbook/engineering",
    "content/handbook/people-group",
]


def clone_repo(target: Path = RAW_DIR) -> None:
    if target.exists():
        print(f"[skip] Repo already cloned at {target}")
        return

    print(f"[clone] Shallow-cloning handbook into {target} ...")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(target)],
        check=True,
    )
    print(f"[clone] Done.")


def filter_sections(source: Path = RAW_DIR, dest: Path = FILTERED_DIR) -> int:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    for section in SECTIONS_TO_KEEP:
        section_path = source / section
        if not section_path.exists():
            print(f"[warn] Section not found: {section_path}")
            continue

        for md_file in section_path.rglob("*.md"):
            relative = md_file.relative_to(source)
            out_path = dest / relative
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(md_file, out_path)
            copied += 1

    print(f"[filter] Copied {copied} markdown files to {dest}")
    return copied


if __name__ == "__main__":
    clone_repo()
    filter_sections()
