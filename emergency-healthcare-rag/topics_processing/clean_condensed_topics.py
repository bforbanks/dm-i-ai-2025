"""Clean the condensed StatPearls markdown copies by dropping noisy sections.

Usage (run once):
    python clean_condensed_topics.py

It walks through `data/condensed_topics` (or the fallback misspelled
`data/condenced_topics`) and removes the following from every *.md file:
    • YAML front-matter delimited by '---' lines.
    • Author / affiliation blocks before the first `#` heading.
    • Trailing sections beginning with any of these headings:
        References, Review Questions, Enhancing Healthcare Team Outcomes,
        Pearls and Other Issues, Disclosure, Author Information, Outcomes
    • Empty leading/trailing whitespace lines.

The script overwrites files **in place** in the condensed copy only, leaving
`data/topics` untouched.
"""
from __future__ import annotations

import re
from pathlib import Path
import sys
from typing import Iterable

PRIMARY_DIR = Path("data/condensed_topics")

CUTOFF_PATTERNS: Iterable[str] = (
    r"^##\s+References",
    r"^##\s+Review Questions",
    r"^##\s+Enhancing Healthcare Team Outcomes",
    r"^##\s+Pearls and Other Issues",
    r"^##\s+Disclosure",
    r"^##\s+Author",
    r"^##\s+Outcomes",
)
CUTOFF_REGEX = re.compile("|".join(CUTOFF_PATTERNS), re.IGNORECASE)


def clean_markdown(raw: str) -> str:
    lines = raw.splitlines()

    # Remove YAML front-matter if present
    if lines and lines[0].strip() == "---":
        try:
            idx2 = lines.index("---", 1)
            lines = lines[idx2 + 1 :]
        except ValueError:
            # only one delimiter → drop it
            lines = lines[1:]

    # Find the first level-1 heading (# …) which is the article title
    first_h1_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("# "):
            first_h1_idx = i
            break
    if first_h1_idx is None:
        return "\n".join(lines)  # no heading → return as-is

    # Keep the title line
    lines_after_title = lines[first_h1_idx + 1 :]

    # Skip everything up to the first level-2 heading (## …) – this discards
    # author names, affiliations, CE activity blurbs, etc.
    for j, ln in enumerate(lines_after_title):
        if ln.startswith("## "):
            lines = [lines[first_h1_idx]] + lines_after_title[j:]
            break
    else:
        lines = [lines[first_h1_idx]] + lines_after_title  # no level-2 heading found

    # Strip out **Objectives** block if present but keep Continuing Education Activity
    cleaned_lines = []
    skip_objectives = False
    for ln in lines:
        # Detect start of Objectives block
        if re.match(r"^\*\*Objectives?:\*\*", ln, re.IGNORECASE):
            skip_objectives = True
            continue
        # End skipping when we hit the next level-2 heading
        if skip_objectives and ln.startswith("## "):
            skip_objectives = False
        if not skip_objectives:
            cleaned_lines.append(ln)
    lines = cleaned_lines

    # Stop at first unwanted trailing section
    for i, ln in enumerate(lines):
        if CUTOFF_REGEX.match(ln):
            lines = lines[:i]
            break

    # Strip leading/trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines) + "\n"  # ensure trailing newline


def main() -> None:
    base = PRIMARY_DIR
    md_files = list(base.rglob("*.md"))
    if not md_files:
        print(f"No markdown files found under {base}")
        return

    for md_path in md_files:
        raw = md_path.read_text(encoding="utf-8")
        cleaned = clean_markdown(raw)
        md_path.write_text(cleaned, encoding="utf-8")

    print(f"Cleaned {len(md_files)} markdown files in {base}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.exit(str(exc))
