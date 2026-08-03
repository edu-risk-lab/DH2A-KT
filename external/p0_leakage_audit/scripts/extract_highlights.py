"""Extract highlighted revision regions (\rev{...} and revbox) with the enclosing
section/subsection heading, for report writing."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper" / "main_APIN_highlight.tex"
text = SRC.read_text(encoding="utf-8")
lines = text.split("\n")

# Track current section context per line.
sec = ""
sub = ""
line_ctx = []
sec_re = re.compile(r"\\section\*?\{(.+?)\}")
sub_re = re.compile(r"\\subsection\*?\{(.+?)\}")
for ln in lines:
    ms = sec_re.search(ln)
    if ms:
        sec = ms.group(1)
        sub = ""
    msub = sub_re.search(ln)
    if msub:
        sub = msub.group(1)
    line_ctx.append((sec, sub))


def find_matching(s, start):
    depth = 1
    j = start
    n = len(s)
    while j < n and depth > 0:
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return -1


# Build offset -> line index map.
offsets = []
pos = 0
for i, ln in enumerate(lines):
    offsets.append(pos)
    pos += len(ln) + 1


def line_of(off):
    import bisect
    return bisect.bisect_right(offsets, off) - 1


items = []
# Inline \rev{...}
for m in re.finditer(r"\\rev\{", text):
    s = m.end()
    e = find_matching(text, s)
    inner = text[s:e]
    li = line_of(m.start())
    items.append(("INLINE", li, line_ctx[li], inner.strip()))

# revbox blocks
box_re = re.compile(r"\\begin\{revbox\}(.*?)\\end\{revbox\}", re.DOTALL)
for m in box_re.finditer(text):
    inner = m.group(1)
    li = line_of(m.start())
    items.append(("BLOCK", li, line_ctx[li], inner.strip()))

items.sort(key=lambda x: x[1])

out = []
for kind, li, (sec, sub), body in items:
    out.append(f"\n===== [{kind}] line {li+1} | SEC: {sec} | SUB: {sub}")
    out.append(body)

report = "\n".join(out)
Path(ROOT / "paper" / "_highlights_extract.txt").write_text(report, encoding="utf-8")
print(f"total highlighted regions: {len(items)}")
print(f"  inline: {sum(1 for x in items if x[0]=='INLINE')}")
print(f"  block : {sum(1 for x in items if x[0]=='BLOCK')}")
