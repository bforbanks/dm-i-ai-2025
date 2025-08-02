"""Copy all markdown topic files into a new folder `data/condenced_topics/` preserving
folder hierarchy. This allows experimentation (e.g. pruning references) without touching
the original scraped articles.

Run once:
    python create_condensed_topics.py
"""
from pathlib import Path
import shutil

SRC_DIR = Path("data/topics")
DST_DIR = Path("data/condenced_topics")


def copy_markdown_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source directory {src} not found")

    for md_path in src.rglob("*.md"):
        rel_path = md_path.relative_to(src)
        target_path = dst / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md_path, target_path)

    print(f"Copied all markdown files from {src} to {dst}")


def main() -> None:
    if DST_DIR.exists():
        print(f"Destination {DST_DIR} already exists – skipping copy to avoid overwriting.")
        return

    copy_markdown_tree(SRC_DIR, DST_DIR)


if __name__ == "__main__":
    main()
