#!/usr/bin/env python3
"""Merge supplementary.tex appendix into main_APIN.tex (single document)."""

from __future__ import annotations

import re
from pathlib import Path


def strip_rev_markup(text: str) -> str:
    """Remove \\rev{...} wrappers (appendix may carry revision highlights)."""
    out: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith("\\rev{", i):
            i += 5
            depth = 1
            start = i
            while i < len(text) and depth:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            out.append(text[start : i - 1])
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def extract_supplement_body(supp_text: str) -> str:
    start = supp_text.index("\\section*{Supplementary index}")
    end = supp_text.rindex("\\end{document}")
    body = supp_text[start:end].strip()
    body = body.replace(
        "\\section*{Supplementary index}",
        "\\section*{Appendix overview}",
    )
    body = body.replace("\\label{sec:supp-index}", "\\label{sec:appendix-index}")
    body = re.sub(
        r"\\rev\{\\section\{([^}]+)\}\}",
        r"\\section{\1}",
        body,
    )
    body = body.replace("main manuscript", "main text")
    body = body.replace("main-text", "main text")
    body = body.replace("in the main manuscript", "in the main text")
    body = body.replace("The main manuscript ", "The main text ")
    body = body.replace("The main manuscript", "The main text")
    return strip_rev_markup(body)


def patch_main(main_text: str, appendix_body: str) -> str:
    main_text = main_text.replace("\\usepackage{xr}\n\\externaldocument{supplementary}\n", "")
    main_text = main_text.replace(
        "  3. Compile: pdflatex -> bibtex -> pdflatex -> pdflatex.",
        "  3. Compile: pdflatex -> bibtex -> pdflatex -> pdflatex.\n"
        "%     (Supplementary tables/figures are in the Appendix; no separate supp PDF.)",
    )

    marker = "%-----------------------------------------------------------------------------\n\\backmatter"
    if marker not in main_text:
        raise ValueError("Declarations section not found in main_APIN.tex")
    appendix_block = (
        "%-----------------------------------------------------------------------------\n"
        "\\appendix\n"
        f"{appendix_body}\n"
        "%-----------------------------------------------------------------------------\n\n"
    )
    main_text = main_text.replace(marker, appendix_block + marker, 1)

    main_text = main_text.replace(
        "Supplementary tables and figures are provided in the separate supplementary material document.",
        "Supplementary tables and figures are included in the Appendix of this document.",
    )
    old_supp = (
        "\\medskip\\noindent\\textbf{Supplementary material.} The supplementary material document\n"
        "is provided alongside this manuscript and is indexed at its opening page.\n"
    )
    new_supp = (
        "\\medskip\\noindent\\textbf{Appendix.} Supplementary tables and figures are merged into\n"
        "the Appendix (overview at Section~\\ref{sec:appendix-index}).\n"
    )
    main_text = main_text.replace(old_supp, new_supp)
    main_text = main_text.replace(
        "(Section~\\ref{sec:supp-anova} in the supplementary material).",
        "(Section~\\ref{sec:supp-anova} in the Appendix).",
    )
    main_text = main_text.replace(
        "Cold-start panels in the main\nmanuscript (Figure~\\ref{fig:coldstart-viz})",
        "Cold-start panels in the main text (Figure~\\ref{fig:coldstart-viz})",
    )
    return main_text


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    main_path = root / "paper" / "main_APIN.tex"
    supp_path = root / "paper" / "supplementary.tex"

    main_text = main_path.read_text(encoding="utf-8")
    supp_text = supp_path.read_text(encoding="utf-8")
    appendix_body = extract_supplement_body(supp_text)
    merged = patch_main(main_text, appendix_body)
    main_path.write_text(merged, encoding="utf-8", newline="\n")
    print(f"Merged appendix into {main_path} ({len(appendix_body.splitlines())} lines)")


if __name__ == "__main__":
    main()
