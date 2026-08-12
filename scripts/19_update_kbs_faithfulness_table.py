#!/usr/bin/env python3
"""Fill paper/tables/table_kc_jaccard_ig.tex from KBS faithfulness comparison JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--comparison",
        type=Path,
        default=REPO / "results" / "tables" / "xes3g5m_fold0_kbs_faithfulness_comparison.json",
    )
    p.add_argument(
        "--table",
        type=Path,
        default=REPO / "paper" / "tables" / "table_kc_jaccard_ig.tex",
    )
    p.add_argument("--backend-label", default="qwen2.5:7b (Ollama)")
    args = p.parse_args()

    cmp = json.loads(args.comparison.read_text(encoding="utf-8"))
    by = {a["label"]: a for a in cmp["arms"]}
    g, i = by["grounded"], by["ig"]

    # Preserve stub rows if present in existing table; otherwise keep harness defaults.
    stub_g_n, stub_g_flag, stub_g_jc = 20, "0.000", "0.567"
    stub_i_n, stub_i_flag, stub_i_jc = 20, "1.000", "0.567"
    old = args.table.read_text(encoding="utf-8") if args.table.exists() else ""
    for line in old.splitlines():
        if "StubLLM" in line and "Grounded" in line:
            parts = [x.strip() for x in line.split("&")]
            if len(parts) >= 5:
                stub_g_n, stub_g_flag, stub_g_jc = parts[2], parts[3], parts[4].rstrip(" \\")
        if "StubLLM" in line and line.strip().startswith("IG"):
            parts = [x.strip() for x in line.split("&")]
            if len(parts) >= 5:
                stub_i_n, stub_i_flag, stub_i_jc = parts[2], parts[3], parts[4].rstrip(" \\")

    tex = f"""% Auto-updated by scripts/19_update_kbs_faithfulness_table.py
\\begin{{table}}[!t]
\\caption{{Tier~2 explanation faithfulness: grounded Diagnostician vs.\\ independent
generation (IG). ID-anchored KC-Jaccard (JC) measures overlap between
\\texttt{{GroundedKCs}} trailer IDs and the support set
$\\{{\\mathrm{{target}}\\}}\\cup\\mathrm{{history}}\\cup 1$-hop $E_{{\\mathrm{{pre}}}}$.
IG omits frozen $P(\\mathrm{{correct}})$ from the Diagnostician prompt; the Critic
still sees the true Tier-1 score.}}
\\label{{tab:kc-jaccard-ig}}
\\centering
\\begin{{tabular}}{{@{{}}llccc@{{}}}}
\\toprule
Arm & Backend & $n$ & Flag rate & Mean JC \\\\
\\midrule
Grounded & StubLLM (harness) & {stub_g_n} & {stub_g_flag} & {stub_g_jc} \\\\
IG & StubLLM (harness) & {stub_i_n} & {stub_i_flag} & {stub_i_jc} \\\\
Grounded & {args.backend_label} & {g['n_samples']} & {g['flag_rate']:.3f} & {g['mean_kc_jaccard']:.3f} \\\\
IG & {args.backend_label} & {i['n_samples']} & {i['flag_rate']:.3f} & {i['mean_kc_jaccard']:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    args.table.write_text(tex, encoding="utf-8")
    print(f"Wrote {args.table}")
    print(
        f"  grounded JC={g['mean_kc_jaccard']:.4f} flag={g['flag_rate']:.4f} | "
        f"IG JC={i['mean_kc_jaccard']:.4f} flag={i['flag_rate']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
