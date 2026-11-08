#!/usr/bin/env python3
"""Junyi seconds unit check (docs/preregistration_primary_rule.md Section 8.1;
docs/preregistration_2026-11-08.md).

No training. Pairs group junyi-sec (time_done // 1e6) with the existing
junyi-partial microsecond arms, same 50,000 learners, fold 0, corpus
configuration, ten seeds.

    python scripts/67_junyi_sec_check.py

Writes results/tables/junyi_sec_check.json and
paper/tables/table_junyi_sec.tex.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from scipy import stats

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results" / "tables"
SEEDS = (0, 17, 42, 1234, 2024, 7, 99, 314, 2718, 3407)
TAU = 0.002
OUT_JSON = TABLES / "junyi_sec_check.json"
OUT_TEX = REPO / "paper" / "tables" / "table_junyi_sec.tex"

CONTRASTS = (
    (
        "Seconds minus microseconds, package",
        "junyi-sec_time-package",
        "junyi-partial_time-package",
        "E",
    ),
    (
        "Seconds minus microseconds, T-zero",
        "junyi-sec_t-zero",
        "junyi-partial_t-zero",
        "E",
    ),
    (
        "Package vs.\\ no-time, seconds",
        "junyi-sec_time-package",
        "junyi-partial_no-time",
        "D",
    ),
    (
        r"Aligned gap vs.\ T-zero, seconds",
        "junyi-sec_time-package",
        "junyi-sec_t-zero",
        "D",
    ),
)


def _read(arm: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for seed in SEEDS:
        path = TABLES / f"dh2_kt_ladder_{arm}_s{seed}_vs_p0.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8", newline="") as fh:
            row = next(r for r in csv.DictReader(fh) if r.get("fold", "0") != "mean")
        out[seed] = float(row["dh2_kt_auc"])
    return out


def _score(on_arm: str, ctrl_arm: str) -> dict:
    on, ctrl = _read(on_arm), _read(ctrl_arm)
    deltas = [on[s] - ctrl[s] for s in SEEDS]
    n = len(deltas)
    mean = statistics.mean(deltas)
    sd = statistics.stdev(deltas)
    half = stats.t.ppf(0.975, df=n - 1) * sd / math.sqrt(n)
    return {
        "on": on_arm,
        "control": ctrl_arm,
        "deltas": deltas,
        "mean": mean,
        "sd": sd,
        "lcb95": mean - half,
        "ucb95": mean + half,
        "seed_k": sum(d >= TAU for d in deltas),
        "all_positive": all(d > 0 for d in deltas),
    }


def _cell(s: dict) -> str:
    return (
        f"${1e3 * s['mean']:+.2f}$ "
        f"[${1e3 * s['lcb95']:+.2f}$, ${1e3 * s['ucb95']:+.2f}$] "
        f"({s['seed_k']}/10)"
    )


def main() -> int:
    rows = []
    for contrast, on_arm, ctrl_arm, fam in CONTRASTS:
        scored = _score(on_arm, ctrl_arm)
        scored.update({"contrast": contrast, "family": fam})
        rows.append(scored)

    payload = {
        "preregistration": "docs/preregistration_2026-11-08.md",
        "metric": "pooled held-out evaluation AUC (dh2_kt_auc)",
        "tau": TAU,
        "n_seeds": len(SEEDS),
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    body = []
    for r in rows:
        body.append(
            f"{r['contrast']} & {r['family']} & {_cell(r)} \\\\"
        )
    tex = rf"""% Auto: scripts/67_junyi_sec_check.py
\begin{{table}}[tbp]
\caption{{Junyi completion-time unit check (first $50{{,}}000$ learners,
fold~0, corpus configuration, ten paired seeds;
\texttt{{docs/preregistration\_primary\_rule.md}} Section~8.1).
Seconds arms use \texttt{{time\_done//1e6}}; microsecond arms are the
existing \texttt{{junyi-partial}} runs. Differences are mean paired
pooled held-out evaluation AUC ($10^{{-3}}$) with two-sided 95\%
intervals; $k/10$ is the seed-stability descriptor at
$\tau{{=}}{{+}}2.0\times 10^{{-3}}$. The two seconds-minus-microseconds
rows are family~E (unit check); the two map rows in seconds are
family~D (completion-time, no credit). The invariance display is the
interval around zero, not a test of $H_0{{:}}\,\mu\le\tau$. These rows
are not in family~C.}}
\label{{tab:junyi-sec}}
\centering
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}llr@{{}}}}
\toprule
Contrast & Fam. & Mean [95\% CI] ($k/10$) \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    OUT_TEX.write_text(tex, encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TEX}")
    for r in rows:
        print(
            f"{r['family']} {r['contrast']}: "
            f"{1e3 * r['mean']:+.3f} "
            f"[{1e3 * r['lcb95']:+.3f},{1e3 * r['ucb95']:+.3f}] "
            f"k={r['seed_k']}/10"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
