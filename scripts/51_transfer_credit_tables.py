"""Credit outcomes for the transfer ladders and the reference-budget padded arm.

Reads the per-seed comparison CSVs that ``scripts/50_credit_ladder_plan.py``
jobs write, pairs on/control arms by seed, and applies both cuts:

  absolute  tau = +0.002
  relative  tau_rel = max(0.002, K * SD_seed)

SD_seed is the sample SD of the control arm's validation AUC over the five
seeds. K = 10 is the largest integer that keeps the XES3G5M fold-0 map
(both packages credited, aligned gap not credited); the XES rows are
recomputed here from the frozen CSVs so that check is reproducible.

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
K = 10
OUT_JSON = TABLES / "credit_ladder_transfer.json"
OUT_TEX = REPO / "paper" / "tables" / "table_credit_transfer.tex"


def _ladder(arm: str) -> str:
    return f"dh2_kt_ladder_{arm}_s{{seed}}_vs_p0.csv"


def _frozen(prefix: str) -> str:
    return f"{prefix}_s{{seed}}.csv"


# (corpus, label, rung, on pattern, control pattern)
CONTRASTS = [
    ("XES3G5M", "Package, observed incidence", "Package", None, None),
    ("XES3G5M", "Package, zero incidence", "Package",
     _frozen("qkc_cm_zero_time"), _frozen("qkc_cm_zero")),
    ("XES3G5M", "Aligned gap vs.\\ T-zero", "Architecture",
     _frozen("qkc_cm_zero_time"), _frozen("a1_t_zero")),
    ("XES3G5M", "Aligned gap vs.\\ T-misaligned", "Architecture",
     _frozen("qkc_cm_zero_time"), _frozen("a1_t_misaligned")),
    ("XES3G5M", "Package vs.\\ param-padded no-time", "Package (param.\\ matched)",
     _frozen("qkc_cm_zero_time"), _ladder("package-refbudget_no-time-param-padded")),
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
    ("Junyi (50k)", "Package, zero incidence", "Package",
     _ladder("junyi-partial_time-package"), _ladder("junyi-partial_no-time")),
    ("Junyi (50k)", "Aligned gap vs.\\ T-zero", "Architecture",
     _ladder("junyi-partial_time-package"), _ladder("junyi-partial_t-zero")),
]


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


def _outcome(deltas: list[float], tau: float) -> str:
    if all(d >= tau for d in deltas):
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
    for corpus, label, rung, on_pat, ctrl_pat in CONTRASTS:
        if on_pat is None:
            on_val, ctrl = _a7_both()
        else:
            ctrl = _read(ctrl_pat)
            on_val = {s: v["val_auc"] for s, v in _read(on_pat).items()}
        deltas = [on_val[s] - ctrl[s]["val_auc"] for s in SEEDS]
        sd_ctrl = statistics.stdev(ctrl[s]["val_auc"] for s in SEEDS)
        tau_rel = max(TAU, K * sd_ctrl)
        rows.append(
            {
                "corpus": corpus,
                "contrast": label,
                "rung": rung,
                "seeds": list(SEEDS),
                "delta_val": deltas,
                "mean_delta_val": statistics.mean(deltas),
                "min_delta_val": min(deltas),
                "sd_control_val": sd_ctrl,
                "tau_abs": TAU,
                "tau_rel": tau_rel,
                "pass_abs": sum(d >= TAU for d in deltas),
                "pass_rel": sum(d >= tau_rel for d in deltas),
                "outcome_abs": _outcome(deltas, TAU),
                "outcome_rel": _outcome(deltas, tau_rel),
                "control_batch": ctrl[SEEDS[0]]["batch"],
                "control_epochs": ctrl[SEEDS[0]]["epochs"],
                "n_scored": ctrl[SEEDS[0]]["n"],
            }
        )

    xes = [r for r in rows if r["corpus"] == "XES3G5M" and r["rung"] == "Package"]
    max_k = min(r["min_delta_val"] / r["sd_control_val"] for r in xes)
    payload = {"K": K, "tau_abs": TAU, "max_K_preserving_xes_packages": max_k, "rows": rows}
    if K > math.floor(max_k):
        raise SystemExit(f"K={K} no longer preserves the XES packages (max {max_k:.2f})")
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    short = {"Credited": "Credited", "Consistent-positive, not credited": "Cons.-pos.", "Unsupported": "Unsupp."}
    lines = [
        "\\begin{table}[t]",
        "\\caption{Credit outcomes under both cuts. Five paired training seeds",
        "$\\{0,17,42,1234,2024\\}$, learner fold~0, validation AUC. Absolute cut",
        "$\\tau{=}{+}0.002$; relative cut",
        f"$\\tau_{{\\mathrm{{rel}}}}{{=}}\\max(0.002,\\,{K}\\cdot\\mathrm{{SD}}_{{\\mathrm{{seed}}}})$,",
        "with $\\mathrm{SD}_{\\mathrm{seed}}$ the sample SD of the control arm's",
        f"validation AUC. $K{{=}}{K}$ is the largest integer that keeps both XES3G5M",
        f"packages credited (bound {max_k:.1f}). Cons.-pos.\\ is consistent-positive,",
        "not credited; Unsupp.\\ is unsupported. The param-padded arm adds the",
        "same 256 parameters as the timed arm and multiplies their output by",
        "zero. XES3G5M rows use batch~16, 30~epochs, patience~5; ASSIST2012",
        "and Junyi use their corpus configs, batch~32 and 10~epochs. Junyi uses",
        "the first 50{,}000 train and 50{,}000 evaluation learners, a partial",
        "replication; its items map one-to-one to KCs, so no incidence",
        "contrast is run there. Source:",
        "\\protect\\path{results/tables/credit_ladder_transfer.json}.}",
        "\\label{tab:credit-transfer}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{@{}llrrcc@{}}",
        "\\toprule",
        "Corpus & Contrast & Mean $\\Delta$val & $\\tau_{\\mathrm{rel}}$ & Outcome at $+0.002$ & Outcome at $\\tau_{\\mathrm{rel}}$ \\\\",
        "\\midrule",
    ]
    last = None
    for r in rows:
        corpus = r["corpus"] if r["corpus"] != last else ""
        if last is not None and r["corpus"] != last:
            lines.append("\\addlinespace")
        last = r["corpus"]
        lines.append(
            f"{corpus} & {r['contrast']} & ${r['mean_delta_val']:+.5f}$ & ${r['tau_rel']:.4f}$ & "
            f"{short[r['outcome_abs']]} ${r['pass_abs']}/5$ & {short[r['outcome_rel']]} ${r['pass_rel']}/5$ \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")

    print(f"K={K} max_K={max_k:.3f}")
    for r in rows:
        print(
            f"{r['corpus']:12} {r['contrast']:40} mean={r['mean_delta_val']:+.6f} "
            f"min={r['min_delta_val']:+.6f} sd_ctrl={r['sd_control_val']:.6f} "
            f"tau_rel={r['tau_rel']:.6f} abs={r['pass_abs']}/5 {r['outcome_abs']} | "
            f"rel={r['pass_rel']}/5 {r['outcome_rel']}"
        )
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
