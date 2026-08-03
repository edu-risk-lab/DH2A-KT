"""Strip revision-highlight markup from a LaTeX file to recover a clean version.

Removes: the revision-highlight preamble block, \begin{revbox}/\end{revbox}
lines, and unwraps \rev{...} and \mbox{...}. Baseline manuscript had zero
\mbox, so every \mbox present is highlight-related and safe to unwrap.
"""
import re
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("paper/main_APIN.tex")
text = path.read_text(encoding="utf-8")


def unwrap(s: str, token: str) -> str:
    """Remove `token{...}` wrappers, keeping the inner content (brace-matched)."""
    out = []
    i, n = 0, len(s)
    while i < n:
        if s.startswith(token, i):
            depth = 1
            j = i + len(token)
            while j < n and depth > 0:
                c = s[j]
                if c == "\\":  # skip escaped char (\{ \} \_ \% ...)
                    j += 2
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            out.append(s[i + len(token):j])
            i = j + 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


# 1. Remove the revision-highlight preamble block (marker .. closing \fi).
text, n_pre = re.subn(
    r"% --- Revision highlighting \(yellow\).*?\\fi\n",
    "",
    text,
    flags=re.DOTALL,
)

# 2. Drop standalone \begin{revbox} / \end{revbox} lines.
lines = text.split("\n")
kept = [ln for ln in lines if ln.strip() not in ("\\begin{revbox}", "\\end{revbox}")]
n_box = len(lines) - len(kept)
text = "\n".join(kept)

# 3. Unwrap \rev{...} then \mbox{...}.
before_rev = text.count("\\rev{")
text = unwrap(text, "\\rev{")
before_mbox = text.count("\\mbox{")
text = unwrap(text, "\\mbox{")

# 4. Revert the one wording tweak made only to satisfy soul.
text = text.replace("(for example, GIKT)", "(e.g.\\ GIKT)")

path.write_text(text, encoding="utf-8")
print(f"preamble blocks removed : {n_pre}")
print(f"revbox lines removed    : {n_box}")
print(f"\\rev{{}} unwrapped        : {before_rev}")
print(f"\\mbox{{}} unwrapped       : {before_mbox}")
print(f"residual \\rev{{  : {text.count(chr(92)+'rev{')}")
print(f"residual \\mbox{{ : {text.count(chr(92)+'mbox{')}")
print(f"residual revbox  : {text.count('revbox')}")
