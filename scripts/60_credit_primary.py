#!/usr/bin/env python3
"""Primary credit rule of docs/preregistration_primary_rule.md (Sections 3-6).

No training. Reads the ten-seed credit JSONs (pooled held-out evaluation
AUC stored as dh2_kt_auc), Junyi history-only CSVs, and attention-ASSISTments
ladder CSVs; assigns each contrast to family C (primary), E (exploratory),
or D (diagnostic, no credit); and applies the one-sided credit test
(H0: mu <= tau) and positivity test (H0: mu <= 0) with Holm within family C
and Benjamini-Hochberg within family E, alpha = 0.025 one-sided.

    python scripts/60_credit_primary.py

Writes results/tables/credit_primary_test.json.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path

from scipy import stats

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results" / "tables"
TEST_JSON = TABLES / "credit_ladder_transfer_10seed_test.json"
VAL_JSON = TABLES / "credit_ladder_transfer_10seed.json"
HISTORY_JSON = TABLES / "junyi_history_check.json"
OUT_JSON = TABLES / "credit_primary_test.json"

TAU = 0.002
ALPHA = 0.025
TAU_GRID = (0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0040)
SEEDS = (0, 17, 42, 1234, 2024, 7, 99, 314, 2718, 3407)
CONFIRMATORY_CONTRASTS = (
    "Package, zero incidence",
    "Aligned gap vs.\\ T-zero",
    "Q$\\leftarrow$KC incidence",
)
FOLD_RE = re.compile(r"(\\)?\s*fold\s+(\d)$")

# Extra family-E contrasts not present in the transfer JSON.
ATTN_CONTRASTS = (
    ("Package vs.\\ no-time", "Package", "time-package", "no-time"),
    ("Aligned gap vs.\\ T-zero", "Architecture", "time-package", "t-zero"),
    ("Aligned gap vs.\\ T-misaligned", "Architecture", "time-package", "t-misaligned"),
)
JUNYI_FULL_HIST = (
    (
        "Junyi (full) history-only fold 0",
        0,
        "junyi-hist-full-fold0_time-history",
        "junyi-full_no-time",
        "junyi-hist-full-fold0_t-zero-history",
    ),
    (
        "Junyi (full) history-only fold 1",
        1,
        "junyi-hist-full-fold1_time-history",
        "junyi-full-fold1_no-time",
        "junyi-hist-full-fold1_t-zero-history",
    ),
    (
        "Junyi (full) history-only fold 2",
        2,
        "junyi-hist-full-fold2_time-history",
        "junyi-full-fold2_no-time",
        "junyi-hist-full-fold2_t-zero-history",
    ),
)


def _setting_fold(corpus: str) -> tuple[str, int]:
    m = FOLD_RE.search(corpus)
    if not m:
        return corpus.strip(), 0
    return corpus[: m.start()].strip().rstrip("\\").strip(), int(m.group(2))


def family(corpus: str, contrast: str) -> str:
    if corpus.startswith("Junyi") and "history-only" not in corpus:
        return "D"
    if "1{,}000" in corpus:
        return "E"
    return "C" if contrast in CONFIRMATORY_CONTRASTS else "E"


def one_sided_p(deltas: list[float], null: float) -> float:
    n = len(deltas)
    mean = statistics.mean(deltas)
    sd = statistics.stdev(deltas)
    if sd == 0.0:
        return 0.0 if mean > null else 1.0
    return float(stats.t.sf((mean - null) / (sd / math.sqrt(n)), df=n - 1))


def holm(p: list[float]) -> list[float]:
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def bh(p: list[float]) -> list[float]:
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i], reverse=True)
    adj = [0.0] * m
    running = 1.0
    for rank_from_top, i in enumerate(order):
        rank = m - rank_from_top
        running = min(running, p[i] * m / rank)
        adj[i] = running
    return adj


def outcome(p_tau_adj: float, p0_adj: float) -> str:
    if p_tau_adj <= ALPHA:
        return "Credited"
    if p0_adj <= ALPHA:
        return "Positive, not credited"
    return "Unsupported"


def _adjust(rows: list[dict], fam: str, null: float) -> list[float]:
    idx = [i for i, r in enumerate(rows) if r["family"] == fam]
    raw = [one_sided_p(rows[i]["deltas"], null) for i in idx]
    adj = holm(raw) if fam == "C" else bh(raw)
    out = [math.nan] * len(rows)
    for i, a in zip(idx, adj):
        out[i] = a
    return out


def _decide(rows: list[dict], tau: float) -> list[str]:
    p_tau = {f: _adjust(rows, f, tau) for f in ("C", "E")}
    p_0 = {f: _adjust(rows, f, 0.0) for f in ("C", "E")}
    out = []
    for i, r in enumerate(rows):
        if r["family"] == "D":
            out.append("Diagnostic")
            continue
        f = r["family"]
        out.append(outcome(p_tau[f][i], p_0[f][i]))
    return out


def _letter(label: str) -> str:
    return {"Credited": "C", "Consistent-positive": "P", "Positive": "P", "Unsupported": "U"}[
        label.split(",")[0]
    ]


def _read_auc(arm: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for seed in SEEDS:
        path = TABLES / f"dh2_kt_ladder_{arm}_s{seed}_vs_p0.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8", newline="") as fh:
            row = next(r for r in csv.DictReader(fh) if r.get("fold", "0") != "mean")
        out[seed] = float(row["dh2_kt_auc"])
    return out


def _extra_row(
    corpus: str,
    setting: str,
    fold: int,
    contrast: str,
    rung: str,
    fam: str,
    on_arm: str,
    ctrl_arm: str,
) -> dict:
    on, ctrl = _read_auc(on_arm), _read_auc(ctrl_arm)
    deltas = [on[s] - ctrl[s] for s in SEEDS]
    # Placeholder old-rule labels (not used for primary decisions).
    mean = statistics.mean(deltas)
    lcb = mean - stats.t.ppf(0.975, df=len(deltas) - 1) * statistics.stdev(deltas) / math.sqrt(
        len(deltas)
    )
    old = "Credited" if lcb >= TAU else (
        "Consistent-positive, not credited" if all(d > 0 for d in deltas) else "Unsupported"
    )
    return {
        "corpus": corpus,
        "setting": setting,
        "fold": fold,
        "contrast": contrast,
        "rung": rung,
        "family": fam,
        "deltas": deltas,
        "sd_control": None,
        "old_test_interval": old,
        "old_test_seed": (
            "Credited" if all(d >= TAU for d in deltas) else (
                "Consistent-positive, not credited"
                if all(d > 0 for d in deltas)
                else "Unsupported"
            )
        ),
        "old_val_interval": old,
        "old_val_seed": old,
        "val_mean": None,
        "source_arms": {"on": on_arm, "control": ctrl_arm},
    }


def variance_components(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], dict[int, list[float]]] = {}
    for r in rows:
        if r["family"] != "C":
            continue
        groups.setdefault((r["setting"], r["contrast"]), {})[r["fold"]] = r["deltas"]
    out = []
    for (setting, contrast), by_fold in groups.items():
        if sorted(by_fold) != [0, 1, 2]:
            continue
        folds = [by_fold[f] for f in (0, 1, 2)]
        n = len(folds[0])
        k = 3
        fold_means = [statistics.mean(d) for d in folds]
        grand = statistics.mean(fold_means)
        ms_between = n * sum((m - grand) ** 2 for m in fold_means) / (k - 1)
        ms_within = sum(
            sum((x - fm) ** 2 for x in d) for d, fm in zip(folds, fold_means)
        ) / (k * (n - 1))
        var_fold = max(0.0, (ms_between - ms_within) / n)
        half = stats.t.ppf(0.975, df=k - 1) * statistics.stdev(fold_means) / math.sqrt(k)
        out.append(
            {
                "setting": setting,
                "contrast": contrast,
                "fold_means": fold_means,
                "pooled_mean": grand,
                "pooled_ci95_df2": [grand - half, grand + half],
                "sd_seed_within_fold": math.sqrt(ms_within),
                "sd_between_fold": math.sqrt(var_fold),
                "ms_between": ms_between,
                "ms_within": ms_within,
            }
        )
    return out


def main() -> int:
    test = json.loads(TEST_JSON.read_text(encoding="utf-8"))
    val = {
        (r["corpus"], r["contrast"]): r
        for r in json.loads(VAL_JSON.read_text(encoding="utf-8"))["rows"]
    }
    rows: list[dict] = []
    for r in test["rows"]:
        setting, fold = _setting_fold(r["corpus"])
        v = val[(r["corpus"], r["contrast"])]
        rows.append(
            {
                "corpus": r["corpus"],
                "setting": setting,
                "fold": fold,
                "contrast": r["contrast"],
                "rung": r["rung"],
                "family": family(r["corpus"], r["contrast"]),
                "deltas": r["delta_val"],
                "sd_control": r["sd_control_val"],
                "old_test_interval": r["outcome_ci"],
                "old_test_seed": r["outcome_abs"],
                "old_val_interval": v["outcome_ci"],
                "old_val_seed": v["outcome_abs"],
                "val_mean": v["mean_delta_val"],
            }
        )

    # 50k-subsample Junyi history-only (and query-and-history diagnostics).
    history = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
    for h in history["rows"]:
        rows.append(
            {
                "corpus": "Junyi (50k) history-only",
                "setting": "Junyi (50k) history-only",
                "fold": 0,
                "contrast": h["contrast"],
                "rung": h["rung"],
                "family": "D" if h["contrast"].startswith("Query") else "E",
                "deltas": h["test"]["deltas"],
                "sd_control": None,
                "old_test_interval": h["test"]["outcome_ci"],
                "old_test_seed": h["test"]["outcome_abs"],
                "old_val_interval": h["val"]["outcome_ci"],
                "old_val_seed": h["val"]["outcome_abs"],
                "val_mean": h["val"]["mean"],
            }
        )

    # All-learner Junyi history-only folds 0–2.
    for corpus, fold, on, no_time, t_zero in JUNYI_FULL_HIST:
        rows.append(
            _extra_row(
                corpus,
                corpus,
                fold,
                "History-only package",
                "Package",
                "E",
                on,
                no_time,
            )
        )
        rows.append(
            _extra_row(
                corpus,
                corpus,
                fold,
                "History-only gap vs.\\ T-zero",
                "Architecture",
                "E",
                on,
                t_zero,
            )
        )

    # Attention encoder on ASSISTments one-second export (family E).
    for fold in (0, 1, 2):
        group = f"attn-assist-fold{fold}"
        for contrast, rung, on_name, ctrl_name in ATTN_CONTRASTS:
            rows.append(
                _extra_row(
                    f"ASSIST2012 sec attn fold {fold}",
                    "ASSIST2012 sec attn",
                    fold,
                    contrast,
                    rung,
                    "E",
                    f"{group}_{on_name}",
                    f"{group}_{ctrl_name}",
                )
            )

    p_tau = {f: _adjust(rows, f, TAU) for f in ("C", "E")}
    p_0 = {f: _adjust(rows, f, 0.0) for f in ("C", "E")}
    decided = _decide(rows, TAU)
    curve = {f"{t * 1e3:.1f}": _decide(rows, t) for t in TAU_GRID}
    for i, r in enumerate(rows):
        d = r["deltas"]
        n = len(d)
        mean, sd = statistics.mean(d), statistics.stdev(d)
        f = r["family"]
        r.update(
            {
                "mean": mean,
                "sd": sd,
                "lcb95_two_sided": mean - stats.t.ppf(0.975, df=n - 1) * sd / math.sqrt(n),
                "p_tau_raw": one_sided_p(d, TAU),
                "p_0_raw": one_sided_p(d, 0.0),
                "p_tau_adj": None if f == "D" else p_tau[f][i],
                "p_0_adj": None if f == "D" else p_0[f][i],
                "seed_k": sum(x >= TAU for x in d),
                "outcome": decided[i],
                "seed_variable": decided[i] == "Credited" and sum(x >= TAU for x in d) < n,
                "tau_curve": {t: curve[t][i] for t in curve},
            }
        )

    sd_controls = [r["sd_control"] for r in rows if r["sd_control"] is not None and r["family"] != "D"]
    sd_deltas = [r["sd"] for r in rows if r["family"] != "D"]
    summary = {
        "family_sizes": {f: sum(r["family"] == f for r in rows) for f in ("C", "E", "D")},
        "outcomes_by_family": {
            f: {
                o: sum(r["family"] == f and r["outcome"] == o for r in rows)
                for o in sorted({r["outcome"] for r in rows})
            }
            for f in ("C", "E", "D")
        },
        **{
            f"changed_vs_{old}": [
                {
                    "corpus": r["corpus"],
                    "contrast": r["contrast"],
                    "family": r["family"],
                    "old": r[old],
                    "new": r["outcome"],
                }
                for r in rows
                if r["family"] != "D" and r.get(old) and _letter(r[old]) != _letter(r["outcome"])
            ]
            for old in ("old_test_interval", "old_test_seed", "old_val_interval", "old_val_seed")
        },
        "seed_sd_control_eval_auc": {
            "median": statistics.median(sd_controls),
            "max": max(sd_controls),
        },
        "seed_sd_delta_eval_auc": {
            "median": statistics.median(sd_deltas),
            "max": max(sd_deltas),
        },
    }
    out = {
        "preregistration": "docs/preregistration_primary_rule.md",
        "metric": (
            "pooled held-out evaluation AUC (dh2_kt_auc = original valid ∪ original "
            "test learners of the fold); checkpoint chosen on the internal selection "
            "split carved from training learners only"
        ),
        "tau": TAU,
        "alpha_one_sided": ALPHA,
        "adjustment": {
            "C": "Holm (FWER within family C)",
            "E": "Benjamini-Hochberg (FDR within family E)",
            "D": "none (diagnostic, no credit)",
        },
        "summary": summary,
        "variance_components": variance_components(rows),
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
