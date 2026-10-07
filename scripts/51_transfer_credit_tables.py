"""Credit outcomes for the transfer ladders, the second encoder, and the padded arm.

Rows whose five seed CSVs are not all present are listed as pending and
left out of the table.

Reads the per-seed comparison CSVs that ``scripts/50_credit_ladder_plan.py``
jobs write, pairs on/control arms by seed, and applies two rules:

  seed-wise  every paired delta_s >= tau, tau = +0.002
  interval   mean(delta) - t_{0.975,4} * SD(delta) / sqrt(5) >= tau

SD(delta) is the sample SD of the paired per-seed deltas, so the interval
rule scales with the noise of the contrast itself and has no free
constant. It replaces an earlier max(0.002, K * SD_control) cut whose K
had been chosen to keep the XES3G5M map. The XES rows are recomputed here
from the frozen CSVs.

Writes results/tables/credit_ladder_transfer.json and
paper/tables/table_credit_transfer.tex. Does not retrain anything.

    python scripts/51_transfer_credit_tables.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results" / "tables"
SEEDS = (0, 17, 42, 1234, 2024)
TAU = 0.002
# Two-sided 95% Student t quantile, df = len(SEEDS) - 1 = 4.
T_975 = 2.7764451051977987
OUT_JSON = TABLES / "credit_ladder_transfer.json"
OUT_TEX = REPO / "paper" / "tables" / "table_credit_transfer.tex"


def _ladder(arm: str) -> str:
    return f"dh2_kt_ladder_{arm}_s{{seed}}_vs_p0.csv"


def _frozen(prefix: str) -> str:
    return f"{prefix}_s{{seed}}.csv"


# (corpus, label, rung, on pattern, control pattern)
CONTRASTS = [
    ("XES3G5M", "Q$\\leftarrow$KC incidence", "Capacity",
     _frozen("qkc_cm_observed"), _frozen("qkc_cm_zero")),
    ("XES3G5M", "Package, observed incidence", "Package", None, None),
    ("XES3G5M", "Package, zero incidence", "Package",
     _frozen("qkc_cm_zero_time"), _frozen("qkc_cm_zero")),
    ("XES3G5M", "Aligned gap vs.\\ T-zero", "Architecture",
     _frozen("qkc_cm_zero_time"), _frozen("a1_t_zero")),
    ("XES3G5M", "Aligned gap vs.\\ T-misaligned", "Architecture",
     _frozen("qkc_cm_zero_time"), _frozen("a1_t_misaligned")),
    ("XES3G5M", "Package vs.\\ param-padded no-time", "Package (param.\\ matched)",
     _frozen("qkc_cm_zero_time"), _ladder("package-matched_no-time-param-padded")),
    ("XES3G5M", "Aligned gap vs.\\ T-shuffled", "Architecture",
     _frozen("qkc_cm_zero_time"), _ladder("xes-gap-controls_t-shuffled")),
    ("XES3G5M", "Aligned gap vs.\\ T-boundary", "Architecture",
     _frozen("qkc_cm_zero_time"), _ladder("xes-gap-controls_t-boundary")),
    ("XES3G5M", "T-boundary vs.\\ T-zero", "Architecture",
     _ladder("xes-gap-controls_t-boundary"), _frozen("a1_t_zero")),
    ("XES3G5M fold 1", "Package, zero incidence", "Package",
     _ladder("xes-fold1_time-package"), _ladder("xes-fold1_no-time")),
    ("XES3G5M fold 1", "Aligned gap vs.\\ T-zero", "Architecture",
     _ladder("xes-fold1_time-package"), _ladder("xes-fold1_t-zero")),
    ("XES3G5M fold 2", "Package, zero incidence", "Package",
     _ladder("xes-fold2_time-package"), _ladder("xes-fold2_no-time")),
    ("XES3G5M fold 2", "Aligned gap vs.\\ T-zero", "Architecture",
     _ladder("xes-fold2_time-package"), _ladder("xes-fold2_t-zero")),
    ("XES3G5M attn.", "Package, zero incidence", "Package (backbone)",
     _ladder("backbone-attn_time-package"), _ladder("backbone-attn_no-time")),
    ("XES3G5M attn.", "Aligned gap vs.\\ T-zero", "Architecture",
     _ladder("backbone-attn_time-package"), _ladder("backbone-attn_t-zero")),
    ("XES3G5M attn.", "Aligned gap vs.\\ T-misaligned", "Architecture",
     _ladder("backbone-attn_time-package"), _ladder("backbone-attn_t-misaligned")),
    ("ASSIST2012", "Q$\\leftarrow$KC incidence", "Capacity",
     _ladder("assist_incidence-on"), _ladder("assist_incidence-off")),
    ("ASSIST2012", "Package, observed incidence", "Package",
     _ladder("assist_time-on-observed"), _ladder("assist_incidence-on")),
    ("ASSIST2012", "Package, zero incidence", "Package",
     _ladder("assist_time-on-zero"), _ladder("assist_incidence-off")),
    ("ASSIST2012", "Aligned gap vs.\\ T-zero", "Architecture",
     _ladder("assist_time-on-zero"), _ladder("assist_t-zero")),
    ("ASSIST2012", "Aligned gap vs.\\ T-misaligned", "Architecture",
     _ladder("assist_time-on-zero"), _ladder("assist_t-misaligned")),
    ("ASSIST2012 ref.", "Q$\\leftarrow$KC incidence", "Capacity",
     _ladder("assist-ref_incidence-on"), _ladder("assist-ref_incidence-off")),
    ("ASSIST2012 ref.", "Package, observed incidence", "Package",
     _ladder("assist-ref_time-on-observed"), _ladder("assist-ref_incidence-on")),
    ("ASSIST2012 ref.", "Package, zero incidence", "Package",
     _ladder("assist-ref_time-on-zero"), _ladder("assist-ref_incidence-off")),
    ("ASSIST2012 ref.", "Aligned gap vs.\\ T-zero", "Architecture",
     _ladder("assist-ref_time-on-zero"), _ladder("assist-ref_t-zero")),
    ("ASSIST2012 ref.", "Aligned gap vs.\\ T-misaligned", "Architecture",
     _ladder("assist-ref_time-on-zero"), _ladder("assist-ref_t-misaligned")),
    *[
        (corpus, label, rung, _ladder(f"{group}_{on}"), _ladder(f"{group}_{ctrl}"))
        for group, corpus in (("assist-sec", "ASSIST2012 1\\,s"), ("assist-sec-ref", "ASSIST2012 1\\,s ref."))
        for label, rung, on, ctrl in (
            ("Q$\\leftarrow$KC incidence", "Capacity", "incidence-on", "incidence-off"),
            ("Package, observed incidence", "Package", "time-on-observed", "incidence-on"),
            ("Package, zero incidence", "Package", "time-on-zero", "incidence-off"),
            ("Aligned gap vs.\\ T-zero", "Architecture", "time-on-zero", "t-zero"),
            ("Aligned gap vs.\\ T-misaligned", "Architecture", "time-on-zero", "t-misaligned"),
        )
    ],
    ("Junyi (50k)", "Package, zero incidence", "Package",
     _ladder("junyi-partial_time-package"), _ladder("junyi-partial_no-time")),
    ("Junyi (50k)", "Aligned gap vs.\\ T-zero", "Architecture",
     _ladder("junyi-partial_time-package"), _ladder("junyi-partial_t-zero")),
    ("Junyi (all)", "Package, zero incidence", "Package",
     _ladder("junyi-full_time-package"), _ladder("junyi-full_no-time")),
    ("Junyi (all)", "Aligned gap vs.\\ T-zero", "Architecture",
     _ladder("junyi-full_time-package"), _ladder("junyi-full_t-zero")),
]


def _available(pattern: str | None) -> bool:
    if pattern is None:
        return True
    return all((TABLES / pattern.format(seed=s)).exists() for s in SEEDS)


def _read(pattern: str) -> dict[int, dict]:
    out = {}
    for seed in SEEDS:
        path = TABLES / pattern.format(seed=seed)
        with path.open(encoding="utf-8") as fh:
            row = next(csv.DictReader(fh))
        out[seed] = {
            "val_auc": float(row["val_auc"]),
            "auc": float(row["dh2_kt_auc"]),
            "n": int(row["n_predictions"]),
            "batch": int(row["batch_size"]),
            "epochs": int(row["epochs"]),
            "file": path.name,
        }
    return out


def _outcome(credited: bool, deltas: list[float]) -> str:
    if credited:
        return "Credited"
    if all(d > 0 for d in deltas):
        return "Consistent-positive, not credited"
    return "Unsupported"


def _a7_both() -> tuple[dict[int, float], dict[int, dict]]:
    # QKC-T (observed incidence, both-mode gap) and its no-time twin, as
    # printed in table_a7_query_dt_seeds.tex.
    path = TABLES / "a7_query_dt_ablation.csv"
    on: dict[int, float] = {}
    ctrl: dict[int, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["time_gap_mode"] != "both":
                continue
            seed = int(row["seed"])
            on[seed] = float(row["val_auc"])
            ctrl[seed] = {"val_auc": float(row["twin_val_auc"]), "n": 0, "batch": 16, "epochs": 30}
    return on, ctrl


def main() -> int:
    rows = []
    pending = []
    for corpus, label, rung, on_pat, ctrl_pat in CONTRASTS:
        if not (_available(on_pat) and _available(ctrl_pat)):
            pending.append(f"{corpus}: {label}")
            continue
        if on_pat is None:
            on_val, ctrl = _a7_both()
        else:
            ctrl = _read(ctrl_pat)
            on_val = {s: v["val_auc"] for s, v in _read(on_pat).items()}
        deltas = [on_val[s] - ctrl[s]["val_auc"] for s in SEEDS]
        sd_ctrl = statistics.stdev(ctrl[s]["val_auc"] for s in SEEDS)
        mean = statistics.mean(deltas)
        sd_delta = statistics.stdev(deltas)
        lcb = mean - T_975 * sd_delta / math.sqrt(len(SEEDS))
        rows.append(
            {
                "corpus": corpus,
                "contrast": label,
                "rung": rung,
                "seeds": list(SEEDS),
                "delta_val": deltas,
                "mean_delta_val": mean,
                "min_delta_val": min(deltas),
                "sd_control_val": sd_ctrl,
                "sd_delta_val": sd_delta,
                "lcb95_delta_val": lcb,
                "tau_abs": TAU,
                "pass_abs": sum(d >= TAU for d in deltas),
                "outcome_abs": _outcome(all(d >= TAU for d in deltas), deltas),
                "outcome_ci": _outcome(lcb >= TAU, deltas),
                "control_batch": ctrl[SEEDS[0]]["batch"],
                "control_epochs": ctrl[SEEDS[0]]["epochs"],
                "n_scored": ctrl[SEEDS[0]]["n"],
            }
        )

    payload = {
        "tau_abs": TAU,
        "t_975_df4": T_975,
        "interval_rule": "mean - t_975 * SD(delta) / sqrt(5) >= tau",
        "rows": rows,
        "pending": pending,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    short = {"Credited": "Credited", "Consistent-positive, not credited": "Cons.-pos.", "Unsupported": "Unsupp."}
    lines = [
        "\\begin{table}[tbp]",
        "\\caption{Credit map across settings. Five paired training seeds,",
        "learner fold~0 unless marked, validation AUC. Seed rule: every",
        "$\\Delta_s\\ge{+}0.002$ (passing seeds shown). Interval rule: the lower",
        "95\\% bound of the mean paired $\\Delta$ ($t_{0.975,4}$) is at least",
        "$+0.002$. Cons.-pos.: consistent-positive, not credited; Unsupp.:",
        "unsupported. T-shuffled permutes gaps across all rows and learners;",
        "T-boundary feeds only $1[\\Delta t{=}0]$. ``1\\,s'': ASSIST2012 re-exported",
        "at one-second timestamp resolution (the original export has 1{,}000\\,s).",
        "``Ref.'' and ``(all)'' rows, and all XES3G5M rows, use the",
        "reference recipe; the other ASSIST2012 and Junyi rows use corpus",
        "configurations (Section~\\ref{sec:credit-gate}).}",
        "\\label{tab:credit-transfer}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\renewcommand{\\arraystretch}{0.88}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}llrrll@{}}",
        "\\toprule",
        "Setting & Contrast & Mean $\\Delta$ & Lower 95\\% & Seed rule & Interval rule \\\\",
        "\\midrule",
    ]
    last = None
    for r in rows:
        corpus = r["corpus"] if r["corpus"] != last else ""
        if last is not None and r["corpus"] != last:
            lines.append("\\addlinespace")
        last = r["corpus"]
        lines.append(
            f"{corpus} & {r['contrast']} & ${r['mean_delta_val']:+.5f}$ & ${r['lcb95_delta_val']:+.5f}$ & "
            f"{short[r['outcome_abs']]} ${r['pass_abs']}/5$ & {short[r['outcome_ci']]} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")

    for r in rows:
        print(
            f"{r['corpus']:15} {r['contrast']:40} mean={r['mean_delta_val']:+.6f} "
            f"min={r['min_delta_val']:+.6f} sd_delta={r['sd_delta_val']:.6f} "
            f"lcb95={r['lcb95_delta_val']:+.6f} seed={r['pass_abs']}/5 {r['outcome_abs']} | "
            f"interval {r['outcome_ci']}"
        )
    for p in pending:
        print(f"PENDING {p}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
