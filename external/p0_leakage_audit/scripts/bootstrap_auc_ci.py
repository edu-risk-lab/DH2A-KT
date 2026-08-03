"""Learner-cluster bootstrap CI for Delta-AUC (Package B2).

Primary path: pooled valid+test predictions in
  results/predictions/xes3g5m/fold_{f}/{model}.parquet
Fallback (when parquets are absent): paired-$t$ 95% intervals over three
learner-disjoint CV folds from baseline_fold_results.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(message)s")

PRETTY = {"gkt": "GKT", "gikt": "GIKT", "simplekt": r"\textit{simpleKT}"}
ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_MAX_ROWS = 100_000
BOOTSTRAP_N_RESAMPLES = 400


def _cap_for_bootstrap(
    df_a: pd.DataFrame, df_b: pd.DataFrame, max_rows: int = BOOTSTRAP_MAX_ROWS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df_a) <= max_rows:
        return df_a.reset_index(drop=True), df_b.reset_index(drop=True)
    n_users = df_a["user_id"].nunique()
    per_user = max(8, int(np.ceil(max_rows / n_users)))
    keep_idx: list[int] = []
    for _, grp in df_a.groupby("user_id", sort=False):
        idx = grp.index.to_numpy()
        if len(idx) > per_user:
            pick = grp.sample(n=per_user, random_state=0).index.to_numpy()
            idx = pick
        keep_idx.extend(int(i) for i in idx)
    keep_idx = sorted(keep_idx)
    logging.info(
        "Bootstrap row cap: %d rows from %d (%d learners; ~%d rows/learner).",
        len(keep_idx),
        len(df_a),
        n_users,
        per_user,
    )
    return (
        df_a.loc[keep_idx].reset_index(drop=True),
        df_b.loc[keep_idx].reset_index(drop=True),
    )


def bootstrap_pair(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    pair_name: str,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
) -> dict:
    df_a, df_b = _cap_for_bootstrap(df_a, df_b)
    users_int, _ = pd.factorize(df_a["user_id"])
    users_int = np.asarray(users_int, dtype=np.intp)
    y_true = df_a["y_true"].to_numpy()
    y_prob_a = df_a["y_prob"].to_numpy()
    y_prob_b = df_b["y_prob"].to_numpy()
    num_users = int(users_int.max()) + 1

    rng = np.random.default_rng(42)
    deltas: list[float] = []
    logging.info(
        "Bootstrap %s: %d resamples, %d learners, %d rows.",
        pair_name,
        n_resamples,
        num_users,
        len(y_true),
    )

    for _ in range(n_resamples):
        sampled = rng.integers(0, num_users, size=num_users)
        multiplicity = np.bincount(sampled, minlength=num_users).astype(np.float64)
        weights = multiplicity[users_int]
        if np.dot(weights, y_true) == 0 or np.dot(weights, 1.0 - y_true) == 0:
            continue
        auc_a = roc_auc_score(y_true, y_prob_a, sample_weight=weights)
        auc_b = roc_auc_score(y_true, y_prob_b, sample_weight=weights)
        deltas.append(float(auc_a - auc_b))

    arr = np.asarray(deltas, dtype=float)
    delta_mean = float(np.mean(arr))
    ci_lower = float(np.percentile(arr, 2.5))
    ci_upper = float(np.percentile(arr, 97.5))
    logging.info("%s: Delta=%.4f 95%% CI=[%.4f, %.4f]", pair_name, delta_mean, ci_lower, ci_upper)

    return {
        "model_pair": pair_name,
        "delta_auc": delta_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_learners": num_users,
        "n_rows": len(y_true),
        "method": "learner_bootstrap",
    }


def paired_t_fallback(dataset: str = "xes3g5m") -> list[dict]:
    df = pd.read_csv(ROOT / "results/tables/baseline_fold_results.csv")
    df = df[(df["dataset"] == dataset) & (df["graph_construction"] == "train_only")]
    simple = df[df["model"] == "simplekt"].sort_values("fold")["auc"].to_numpy()
    t_val = float(stats.t.ppf(0.975, df=2))
    rows = []
    for model in ("gkt", "gikt"):
        aucs = df[df["model"] == model].sort_values("fold")["auc"].to_numpy()
        if len(aucs) != 3 or len(simple) != 3:
            continue
        diff = aucs - simple
        mean_diff = float(np.mean(diff))
        se = float(np.std(diff, ddof=1) / np.sqrt(3))
        margin = t_val * se
        rows.append(
            {
                "model_pair": f"{PRETTY[model]} vs {PRETTY['simplekt']}",
                "delta_auc": mean_diff,
                "ci_lower": mean_diff - margin,
                "ci_upper": mean_diff + margin,
                "n_learners": np.nan,
                "n_rows": np.nan,
                "method": "paired_t_fold",
            }
        )
    return rows


def predictions_available(dataset: str = "xes3g5m", folds: list[int] | None = None) -> bool:
    folds = folds or [0, 1, 2]
    for fold in folds:
        d = ROOT / "results/predictions" / dataset / f"fold_{fold}"
        for model in ("simplekt", "gkt", "gikt"):
            if not (d / f"{model}.parquet").exists():
                return False
    return True


def load_pooled_predictions(dataset: str = "xes3g5m", folds: list[int] | None = None):
    folds = folds or [0, 1, 2]
    simple_parts, gkt_parts, gikt_parts = [], [], []
    for fold in folds:
        d = ROOT / "results/predictions" / dataset / f"fold_{fold}"
        simple_parts.append(pd.read_parquet(d / "simplekt.parquet"))
        gkt_parts.append(pd.read_parquet(d / "gkt.parquet"))
        gikt_parts.append(pd.read_parquet(d / "gikt.parquet"))
    return (
        pd.concat(simple_parts, ignore_index=True),
        pd.concat(gkt_parts, ignore_index=True),
        pd.concat(gikt_parts, ignore_index=True),
    )


def write_outputs(rows: list[dict], ci_label: str, method_key: str) -> None:
    out = ROOT / "results/tables"
    out.mkdir(parents=True, exist_ok=True)

    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(out / "bootstrap_auc_ci.csv", index=False)

    tex_lines = [
        r"% Auto-generated by scripts/bootstrap_auc_ci.py",
        rf"% method: {method_key}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        (
            r"\textbf{Model Pair} & $\Delta$\textbf{AUC} & "
            rf"\textbf{{95\% CI ({ci_label})}} & "
            r"\textbf{$n_{\mathrm{learners}}$} & \textbf{$n_{\mathrm{rows}}$} \\"
        ),
        r"\midrule",
    ]
    for r in rows:
        n_learners = "---" if pd.isna(r["n_learners"]) else str(int(r["n_learners"]))
        n_rows = "---" if pd.isna(r["n_rows"]) else f"{int(r['n_rows']):,}".replace(",", "{,}")
        tex_lines.append(
            f"{r['model_pair']} & ${r['delta_auc']:+.3f}$ & "
            f"$[{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}]$ & "
            f"{n_learners} & {n_rows} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}", ""]
    (out / "bootstrap_auc_ci.tex").write_text("\n".join(tex_lines), encoding="utf-8")

    method_note = {
        "learner_bootstrap": (
            r"learner-cluster bootstrap ($B{=}400$; pooled valid+test positions "
            r"across folds~0--2; per-learner row cap when pooled rows exceed "
            rf"{BOOTSTRAP_MAX_ROWS:,}; all learners retained)".replace(",", "{,}")
        ),
        "paired_t_fold": (
            r"paired-$t$ 95\% intervals over three learner-disjoint CV folds "
            r"(default for Table~S16; matches fold-level $\Delta$AUC). "
            r"Optional learner-cluster bootstrap: "
            r"\path{scripts/bootstrap_auc_ci.py --learner-bootstrap}."
        ),
    }
    (out / "bootstrap_method_note.tex").write_text(method_note[method_key], encoding="utf-8")

    macro_lines = []
    for r in rows:
        pair = r["model_pair"]
        d = f"{r['delta_auc']:+.3f}"
        ci = f"$[{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}]$"
        if "GKT" in pair:
            macro_lines.append(rf"\renewcommand{{\GKTdeltavec}}{{{d}}}")
            macro_lines.append(rf"\renewcommand{{\GKTdeltaci}}{{{ci}}}")
        if "GIKT" in pair:
            macro_lines.append(rf"\renewcommand{{\GIKTdeltavec}}{{{d}}}")
            macro_lines.append(rf"\renewcommand{{\GIKTdeltaci}}{{{ci}}}")
    if macro_lines:
        (out / "bootstrap_ci_macros.tex").write_text("\n".join(macro_lines) + "\n", encoding="utf-8")

    logging.info("Wrote %s (method=%s)", out / "bootstrap_auc_ci.tex", method_key)


def synthesize_table(use_learner_bootstrap: bool = False) -> str:
    if use_learner_bootstrap and predictions_available():
        simple, gkt, gikt = load_pooled_predictions()
        rows = [
            bootstrap_pair(gkt, simple, "GKT vs \\textit{simpleKT}"),
            bootstrap_pair(gikt, simple, "GIKT vs \\textit{simpleKT}"),
        ]
        write_outputs(rows, ci_label="learner bootstrap", method_key="learner_bootstrap")
        return "learner_bootstrap"

    if predictions_available():
        logging.info(
            "Using paired-$t$ over three folds for Table S16 (matches fold-level "
            "$\\Delta$AUC; pass --learner-bootstrap for row-level resampling on parquets)."
        )
    else:
        logging.warning(
            "Prediction parquets not found; writing paired-$t$ fallback "
            "from baseline_fold_results.csv."
        )
    rows = paired_t_fallback()
    write_outputs(rows, ci_label=r"paired $t$, 3 folds", method_key="paired_t_fold")
    return "paired_t_fold"


if __name__ == "__main__":
    import sys

    synthesize_table(use_learner_bootstrap="--learner-bootstrap" in sys.argv)
