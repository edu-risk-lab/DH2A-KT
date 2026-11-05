#!/usr/bin/env python3
"""Matched-budget warnings and LSTM-minus-attention learner bootstrap.

docs/preregistration_2026-11-05.md Section 4. Family E. No new training.

Needs prediction npz from scripts/61_learner_bootstrap.py --stage dump.
If dumps are missing, tries dump for XES3G5M (LSTM) and XES3G5M attn.;
tables stay pending when files are absent.

    python scripts/66_nov5_dump_analyses.py
    python scripts/66_nov5_dump_analyses.py --dump-first
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PRED = REPO / "results" / "predictions"
TABLES = REPO / "results" / "tables"
OUT_JSON = TABLES / "nov5_dump_analyses.json"
SEEDS = (0, 17, 42, 1234, 2024, 7, 99, 314, 2718, 3407)
BOOT = 2000
BOOT_SEED = 20261023
K_FRAC = (0.01, 0.05, 0.10)

LSTM = {
    "group": "xes-ref",
    "pkg": "time-package",
    "ctrl": "no-time",
    "setting": "XES3G5M",
}
ATTN = {
    "group": "backbone-attn",
    "pkg": "time-package",
    "ctrl": "no-time",
    "setting": "XES3G5M attn.",
}


def _npz(group: str, arm: str, seed: int) -> Path:
    return PRED / f"xes3g5m_fold0_{group}_{arm}_s{seed}.npz"


def _load(group: str, arm: str, seed: int) -> dict | None:
    path = _npz(group, arm, seed)
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in ("user_ids", "targets", "probs")}


def _alert_one(y: np.ndarray, p: np.ndarray, users: np.ndarray, frac: float) -> dict:
    n = len(y)
    k = max(1, int(round(frac * n)))
    order = np.argsort(p, kind="mergesort")[:k]
    flagged = np.zeros(n, dtype=bool)
    flagged[order] = True
    incorrect = y == 0
    return {
        "k": k,
        "true_alerts": int((flagged & incorrect).sum()),
        "false_alerts": int((flagged & ~incorrect).sum()),
        "n_learners": int(np.unique(users[flagged]).size),
    }


def _mean_alert(spec: dict) -> dict | None:
    per_k: dict[float, list[dict]] = {f: [] for f in K_FRAC}
    used = []
    for seed in SEEDS:
        pkg = _load(spec["group"], spec["pkg"], seed)
        ctrl = _load(spec["group"], spec["ctrl"], seed)
        if pkg is None or ctrl is None:
            continue
        y, users = pkg["targets"].astype(np.int8), pkg["user_ids"]
        if not (np.array_equal(y, ctrl["targets"].astype(np.int8)) and np.array_equal(users, ctrl["user_ids"])):
            raise SystemExit(f"positions differ for {spec['group']} s{seed}")
        used.append(seed)
        for frac in K_FRAC:
            per_k[frac].append(
                {
                    "pkg": _alert_one(y, pkg["probs"], users, frac),
                    "ctrl": _alert_one(y, ctrl["probs"], users, frac),
                }
            )
    if not used:
        return None

    def avg(side: str, key: str, frac: float) -> float:
        return float(np.mean([row[side][key] for row in per_k[frac]]))

    return {
        "seeds_used": used,
        "by_k": {
            f"{int(100 * frac)}%": {
                "true_pkg": avg("pkg", "true_alerts", frac),
                "false_pkg": avg("pkg", "false_alerts", frac),
                "learners_pkg": avg("pkg", "n_learners", frac),
                "true_ctrl": avg("ctrl", "true_alerts", frac),
                "false_ctrl": avg("ctrl", "false_alerts", frac),
                "learners_ctrl": avg("ctrl", "n_learners", frac),
            }
            for frac in K_FRAC
        },
    }


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def _delta_of_deltas() -> dict | None:
    data = {}
    for spec, tag in ((LSTM, "lstm"), (ATTN, "attn")):
        for arm in (spec["pkg"], spec["ctrl"]):
            for seed in SEEDS:
                loaded = _load(spec["group"], arm, seed)
                if loaded is None:
                    return None
                data[(tag, arm, seed)] = loaded
    # Intersection of learners across encoders, evaluated per seed on those ids.
    lstm_ctrl = data[("lstm", LSTM["ctrl"], SEEDS[0])]
    attn_ctrl = data[("attn", ATTN["ctrl"], SEEDS[0])]
    shared = np.intersect1d(np.unique(lstm_ctrl["user_ids"]), np.unique(attn_ctrl["user_ids"]))
    d_exact = []
    d_boot = []
    rng = np.random.default_rng(BOOT_SEED)
    for seed in SEEDS:
        parts = {}
        for tag, spec in (("lstm", LSTM), ("attn", ATTN)):
            for arm, name in ((spec["pkg"], "pkg"), (spec["ctrl"], "ctrl")):
                d = data[(tag, arm, seed)]
                mask = np.isin(d["user_ids"], shared)
                parts[(tag, name)] = (d["targets"].astype(np.int8)[mask], d["probs"][mask], d["user_ids"][mask])
        y_l, p_lp, u_l = parts[("lstm", "pkg")]
        _, p_lc, _ = parts[("lstm", "ctrl")]
        y_a, p_ap, u_a = parts[("attn", "pkg")]
        _, p_ac, _ = parts[("attn", "ctrl")]
        # Align by requiring identical user sequences on the LSTM ctrl vs pkg (already true).
        delta_l = _auc(y_l, p_lp) - _auc(y_l, p_lc)
        delta_a = _auc(y_a, p_ap) - _auc(y_a, p_ac)
        d_exact.append(delta_l - delta_a)
        # Learner bootstrap on LSTM positions (shared ids).
        codes, idx = np.unique(u_l, return_inverse=True)
        n_u = len(codes)
        weights = rng.multinomial(n_u, np.full(n_u, 1.0 / n_u), size=BOOT).astype(np.float32)
        # Cheap AUC via sklearn is too slow inside B=2000; use Mann-Whitney on resampled learners
        # by aggregating per-learner pos/neg in 500 quantile bins.
        def binned_auc(y_arr, p_arr, user_i, w):
            edges = np.unique(np.quantile(p_arr, np.linspace(0.0, 1.0, 501)[1:-1]))
            b = np.searchsorted(edges, p_arr, side="right")
            nb = len(edges) + 1
            pos = np.zeros((n_u, nb), dtype=np.float32)
            neg = np.zeros((n_u, nb), dtype=np.float32)
            np.add.at(pos, (user_i[y_arr == 1], b[y_arr == 1]), 1.0)
            np.add.at(neg, (user_i[y_arr == 0], b[y_arr == 0]), 1.0)
            pos_r, neg_r = w @ pos, w @ neg
            neg_below = np.cumsum(neg_r, axis=1) - neg_r
            num = (pos_r * (neg_below + 0.5 * neg_r)).sum(axis=1)
            den = pos_r.sum(axis=1) * neg_r.sum(axis=1)
            return num / np.maximum(den, 1e-12)

        codes_a, idx_a = np.unique(u_a, return_inverse=True)
        # Independent multinomial on attention learners with same seed stream offset.
        n_ua = len(codes_a)
        w_a = rng.multinomial(n_ua, np.full(n_ua, 1.0 / n_ua), size=BOOT).astype(np.float32)
        boot_l = binned_auc(y_l, p_lp, idx, weights) - binned_auc(y_l, p_lc, idx, weights)
        boot_a = binned_auc(y_a, p_ap, idx_a, w_a) - binned_auc(y_a, p_ac, idx_a, w_a)
        d_boot.append(boot_l - boot_a)
    mean_boot = np.stack(d_boot).mean(axis=0)
    return {
        "n_shared_learners": int(len(shared)),
        "seeds_used": list(SEEDS),
        "mean_delta_of_deltas": float(np.mean(d_exact)),
        "boot_ci95": [float(np.quantile(mean_boot, 0.025)), float(np.quantile(mean_boot, 0.975))],
        "per_seed": [float(x) for x in d_exact],
    }


def write_alert_tex(alert: dict | None) -> None:
    path = REPO / "paper" / "tables" / "table_alert_budget.tex"
    if alert is None:
        return
    lines = [
        r"\begin{table}[tbp]",
        r"\caption{Matched-budget warning on XES3G5M fold~0 held-out positions",
        r"(docs/preregistration\_2026-11-05.md Section~4). Flag the $k\%$ of",
        r"positions with the lowest predicted correctness after the scored item",
        r"has opened. True alerts~=~flagged and incorrect; false alerts~=~flagged",
        r"and correct; learners~=~distinct held-out learners with at least one",
        r"flag. Mean over "
        + str(len(alert["seeds_used"]))
        + r" seeds of the LSTM time package versus no-time.",
        r"Family~E; does not decide credit.}",
        r"\label{tab:alert-budget}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"$k$ & True (pkg) & False (pkg) & Learners (pkg) & True (no time) & False (no time) & Learners (no time) \\",
        r"\midrule",
    ]
    for lab, row in alert["by_k"].items():
        lines.append(
            f"{lab} & {row['true_pkg']:.0f} & {row['false_pkg']:.0f} & {row['learners_pkg']:.0f} & "
            f"{row['true_ctrl']:.0f} & {row['false_ctrl']:.0f} & {row['learners_ctrl']:.0f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_delta_tex(delta: dict | None) -> None:
    path = REPO / "paper" / "tables" / "table_lstm_attn_delta.tex"
    if delta is None:
        return
    lo, hi = delta["boot_ci95"]
    lines = [
        r"\begin{table}[tbp]",
        r"\caption{Learner bootstrap of the difference of XES3G5M fold~0 package",
        r"effects, $(\mathrm{AUC}_{\mathrm{pkg}}-\mathrm{AUC}_{\mathrm{no\text{-}time}})_{\mathrm{LSTM}}",
        r"-(\mathrm{AUC}_{\mathrm{pkg}}-\mathrm{AUC}_{\mathrm{no\text{-}time}})_{\mathrm{attn}}$,",
        r"on the intersection of held-out learners ($B{=}2000$, seed 20261023,",
        f"{delta['n_shared_learners']} shared learners). "
        r"Family~E. Not a family~C credit.}",
        r"\label{tab:lstm-attn-delta}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"Statistic & Point & Learner 95\% CI \\",
        r"\midrule",
        f"$\\Delta_{{\\mathrm{{LSTM}}}}-\\Delta_{{\\mathrm{{attn}}}}$ ($10^{{-3}}$ AUC) & "
        f"{1e3 * delta['mean_delta_of_deltas']:+.2f} & "
        f"[{1e3 * lo:+.2f},{1e3 * hi:+.2f}] \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-first", action="store_true")
    args = parser.parse_args()
    if args.dump_first:
        import subprocess

        for only in ("XES3G5M", "XES3G5M attn."):
            print("dump", only)
            subprocess.run(
                [sys.executable, "scripts/61_learner_bootstrap.py", "--stage", "dump", "--only", only],
                cwd=REPO,
                check=False,
            )
    alert = _mean_alert(LSTM)
    try:
        delta = _delta_of_deltas()
    except SystemExit as exc:
        print("delta skipped:", exc)
        delta = None
    payload = {"preregistration": "docs/preregistration_2026-11-05.md", "alert": alert, "lstm_minus_attn": delta}
    TABLES.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_alert_tex(alert)
    write_delta_tex(delta)
    print("alert", "ok" if alert else "pending")
    print("delta", "ok" if delta else "pending")
    print("wrote", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
