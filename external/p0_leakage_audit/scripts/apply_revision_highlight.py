#!/usr/bin/env python3
"""Wrap git-diff additions in \\rev{...} (yellow highlight) for revision markup."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def git_show(rev: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{rev}:{path}"], text=True, encoding="utf-8", errors="replace"
    )


def git_diff(rev: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "diff", f"{rev}..HEAD", "--", path],
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def parse_added_blocks(diff_text: str) -> list[str]:
    """Return contiguous added-line blocks from a unified diff (joined, no trailing NL)."""
    blocks: list[str] = []
    current: list[str] = []

    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("@@"):
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        if line.startswith("+") and not line.startswith("+++"):
            current.append(line[1:])
        else:
            if current:
                blocks.append("\n".join(current))
                current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


def should_skip_block(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return True
    # Skip pure macro / input lines (highlight prose only).
    if stripped.startswith("\\newcommand") or stripped.startswith("\\IfFileExists"):
        return True
    if stripped.startswith("\\input{") and stripped.endswith("}"):
        return True
    return False


def is_inside_rev(text: str, index: int) -> bool:
    """True if index falls inside an existing \\rev{...} argument (brace depth)."""
    depth = 0
    i = 0
    while i < index and i < len(text):
        if text.startswith("\\rev{", i):
            depth += 1
            i += 5
            continue
        if depth and text[i] == "{":
            depth += 1
        elif depth and text[i] == "}":
            depth -= 1
        i += 1
    return depth > 0


def wrap_block(text: str, block: str, start: int = 0) -> tuple[str, bool]:
    if not block.strip():
        return text, False
    search_from = start
    while True:
        idx = text.find(block, search_from)
        if idx == -1:
            break
        if not is_inside_rev(text, idx):
            wrapped = "\\rev{" + block + "}"
            return text[:idx] + wrapped + text[idx + len(block) :], True
        search_from = idx + 1
    lines = block.split("\n")
    if len(lines) == 1:
        line = lines[0]
        search_from = start
        while True:
            pos = text.find(line, search_from)
            if pos == -1:
                break
            if not is_inside_rev(text, pos):
                wrapped = "\\rev{" + line + "}"
                return text[:pos] + wrapped + text[pos + len(line) :], True
            search_from = pos + 1
    return text, False


def apply_highlights(content: str, blocks: list[str]) -> tuple[str, list[str]]:
    missed: list[str] = []
    ordered = sorted(
        [b for b in blocks if not should_skip_block(b) and "\\rev{" not in b],
        key=len,
        reverse=True,
    )
    for block in ordered:
        new_content, ok = wrap_block(content, block)
        if ok:
            content = new_content
        else:
            missed.append(block[:120] + ("..." if len(block) > 120 else ""))
    return content, missed


def inject_preamble(content: str) -> str:
    if "\\newcommand{\\rev}" in content:
        return content
    rev_block = (
        "\\usepackage[framemethod=tikz]{mdframed}\n"
        "\\mdfdefinestyle{revhighlight}{\n"
        "  backgroundcolor=yellow!45,\n"
        "  linecolor=yellow!45,\n"
        "  linewidth=0pt,\n"
        "  innerleftmargin=3pt,\n"
        "  innerrightmargin=3pt,\n"
        "  innertopmargin=2pt,\n"
        "  innerbottommargin=2pt,\n"
        "  skipabove=3pt,\n"
        "  skipbelow=3pt,\n"
        "}\n"
        "\\newcommand{\\rev}[1]{\\begin{mdframed}[style=revhighlight]#1\\end{mdframed}}\n"
        "% Revision highlight (base rev 501e18c6): remove \\rev{} before final submission.\n"
    )
    marker = "\\usepackage{xcolor}"
    if marker in content:
        return content.replace(marker, marker + "\n" + rev_block, 1)
    marker = "\\usepackage{graphicx}"
    if marker in content:
        return content.replace(marker, marker + "\n" + rev_block, 1)
    return content


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-rev", default="501e18c6")
    parser.add_argument(
        "--files",
        nargs="+",
        default=["paper/main_APIN.tex", "paper/supplementary.tex"],
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    all_missed: list[tuple[str, str]] = []

    for rel in args.files:
        path = root / rel
        diff = git_diff(args.base_rev, rel)
        if not diff.strip():
            print(f"No diff for {rel}; skipping.")
            continue
        blocks = parse_added_blocks(diff)
        content = path.read_text(encoding="utf-8")
        content, missed = apply_highlights(content, blocks)
        if rel.endswith("main_APIN.tex") or rel.endswith("supplementary.tex"):
            content = inject_preamble(content)
        path.write_text(content, encoding="utf-8", newline="\n")
        print(f"Highlighted {rel}: {len(blocks)} diff blocks, {len(missed)} missed.")
        for m in missed:
            all_missed.append((rel, m))

    if all_missed:
        print("\nMissed blocks (apply \\rev{} manually if needed):")
        for rel, m in all_missed:
            print(f"  [{rel}] {m}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
