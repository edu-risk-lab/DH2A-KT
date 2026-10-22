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

``--n-seeds 10`` (docs/preregistration.md) adds the seeds
7, 99, 314, 2718, 3407 and the fold 1-2 rows of the attention encoder,
ASSIST2012 corpus configuration, and Junyi, with t at df = 9. XES3G5M
fold-0 rows take the new seeds from the ``xes-ref`` ladder group. Its
outputs carry a ``_10seed`` suffix; the five-seed outputs are untouched.

``--metric test`` scores the same pairs on test-only AUC instead of the
validation AUC used for early stopping. It writes ``..._test.json`` and
an appendix table of the rows whose outcome differs from the validation
map; the validation outputs are untouched.

Every row also carries ``outcome_ci_lcb_middle``: the interval rule with
its middle outcome taken from the interval (consistent-positive if
0 < lower bound < tau, unsupported if the lower bound <= 0) instead of
the preregistered "every delta_s > 0". It is a sensitivity field and is
not used for any outcome in the paper tables.

    python scripts/51_transfer_credit_tables.py
    python scripts/51_transfer_credit_tables.py --n-seeds 10
    python scripts/51_transfer_credit_tables.py --n-seeds 10 --metric test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results" / "tables"
SEEDS_5 = (0, 17, 42, 1234, 2024)
NEW_SEEDS = (7, 99, 314, 2718, 3407)
SEEDS = SEEDS_5
TAU = 0.002
# Two-sided 95% Student t quantiles, df = 4 and df = 9.
T_975_DF = {5: 2.7764451051977987, 10: 2.2621571627409915}
T_975 = T_975_DF[5]
SEED_WORD = {5: "five", 10: "ten"}
LABEL_SUFFIX = ""
SPLIT_BEFORE = "ASSIST2012 fold 1"
METRIC = "val_auc"
OUT_JSON = TABLES / "credit_ladder_transfer.json"
OUT_TEX = REPO / "paper" / "tables" / "table_credit_transfer.tex"

# Frozen XES3G5M fold-0 arms and the xes-ref ladder arm that repeats each
# one on the new seeds.
FROZEN_TO_XES_REF = {
    "qkc_cm_observed": "incidence-on",
    "qkc_cm_zero": "no-time",
    "qkc_cm_zero_time": "time-package",
    "a1_t_zero": "t-zero",
    "a1_t_misaligned": "t-misaligned",
}


def _ladder(arm: str) -> str:
    return f"dh2_kt_ladder_{arm}_s{{seed}}_vs_p0.csv"


def _frozen(prefix: str) -> str:
    return f"{prefix}_s{{seed}}.csv"


ASSIST_ARMS = (
    ("Q$\\leftarrow$KC incidence", "Capacity", "incidence-on", "incidence-off"),
    ("Package, observed incidence", "Package", "time-on-observed", "incidence-on"),
    ("Package, zero incidence", "Package", "time-on-zero", "incidence-off"),
    ("Aligned gap vs.\\ T-zero", "Architecture", "time-on-zero", "t-zero"),
    ("Aligned gap vs.\\ T-misaligned", "Architecture", "time-on-zero", "t-misaligned"),
)
ASSIST_1000S = "ASSIST2012 1{,}000\\,s"
ASSIST_GROUPS = (
    ("assist-sec", "ASSIST2012"),
    ("assist-sec-ref", "ASSIST2012 ref."),
    ("assist", ASSIST_1000S),
    ("assist-ref", ASSIST_1000S + " ref."),
)
OUT_TEX_RES = REPO / "paper" / "tables" / "table_credit_assist_resolution.tex"
OUT_HEAT = REPO / "paper" / "figures" / "fig_credit_heatmap.tex"
HISTORY_JSON = TABLES / "junyi_history_check.json"
HISTORY_TO_MAP = {
    "History-only package": "Package, zero incidence",
    "History-only gap vs.\\ T-zero": "Aligned gap vs.\\ T-zero",
}
DDAGGER = "$^{\\ddagger}$"
JUNYI_NOTE = [
    f"{DDAGGER}\\,Junyi stores completion times, so the query-side gap",
    "includes the scored item's duration; these credits are not",
    "attempt-onset results (Table~\\ref{tab:junyi-history}).",
]
MAP_LEGEND = [
    "Q$\\leftarrow$KC incidence, question--knowledge-component incidence;",
    "package, the gap input with the maps that read it; T-zero,",
    "T-misaligned, T-shuffled, and T-boundary, the gap map fed zeros, gaps",
    "permuted within each learner, gaps permuted across learners, and only",
    "$1[\\Delta t{=}0]$; param-padded, a no-time control with matched",
    "parameters; attn., attention encoder (otherwise the LSTM tracer); ref.,",
    "reference training recipe (otherwise the corpus configuration); 50k",
    "and all, the first $50{,}000$ or all Junyi learners; Cons.-pos.,",
    "consistent-positive (not credited); Unsupp., unsupported.",
]
MAP_LEGEND_B = [
    "Q$\\leftarrow$KC incidence, question--knowledge-component incidence;",
    "package, the gap input with the maps that read it; T-zero and",
    "T-misaligned, the gap map fed zeros or gaps permuted within each",
    "learner; ref., reference training recipe; 50k and all, the first",
    "$50{,}000$ or all Junyi learners; Cons.-pos., consistent-positive (not credited);",
    "Unsupp., unsupported.",
]


def _show_corpus(corpus: str) -> str:
    return corpus + DDAGGER if corpus.startswith("Junyi") else corpus

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
    # One-second export first (main map); the original 1,000 s export is the
    # resolution sensitivity table.
    *[
        (corpus, label, rung, _ladder(f"{group}_{on}"), _ladder(f"{group}_{ctrl}"))
        for group, corpus in ASSIST_GROUPS
        for label, rung, on, ctrl in ASSIST_ARMS
    ],
    *[
        (f"ASSIST2012 ref.\\ fold {fold}", label, rung,
         _ladder(f"assist-sec-fold{fold}_{on}"), _ladder(f"assist-sec-fold{fold}_{ctrl}"))
        for fold in (1, 2)
        for label, rung, on, ctrl in ASSIST_ARMS
        if ctrl in ("incidence-off", "t-zero") and on == "time-on-zero"
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

# Ten-seed map only: (row after which to insert, corpus, label, rung, on, control).
_PKG = ("Package, zero incidence", "Package", "time-package", "no-time")
_GAP0 = ("Aligned gap vs.\\ T-zero", "Architecture", "time-package", "t-zero")
_GAPM = ("Aligned gap vs.\\ T-misaligned", "Architecture", "time-package", "t-misaligned")
EXTRA_CONTRASTS_10 = [
    *[
        ("XES3G5M attn.", f"XES3G5M attn.\\ fold {fold}",
         "Package (backbone)" if label == _PKG[0] else rung,
         label, _ladder(f"backbone-attn-fold{fold}_{on}"), _ladder(f"backbone-attn-fold{fold}_{ctrl}"))
        for fold in (1, 2)
        for label, rung, on, ctrl in (_PKG, _GAP0, _GAPM)
    ],
    *[
        ("ASSIST2012", f"ASSIST2012 fold {fold}", rung, label,
         _ladder(f"assist-sec-cfg-fold{fold}_{on}"), _ladder(f"assist-sec-cfg-fold{fold}_{ctrl}"))
        for fold in (1, 2)
        for label, rung, on, ctrl in ASSIST_ARMS
    ],
    *[
        (base, f"{base} fold {fold}", rung, label,
         _ladder(f"{group}-fold{fold}_{on}"), _ladder(f"{group}-fold{fold}_{ctrl}"))
        for base, group in (("Junyi (50k)", "junyi-partial"), ("Junyi (all)", "junyi-full"))
        for fold in (1, 2)
        for label, rung, on, ctrl in (_PKG, _GAP0)
    ],
]


def _contrasts() -> list[tuple]:
    if len(SEEDS) == 5:
        return CONTRASTS
    out = list(CONTRASTS)
    block = {}
    for after, corpus, rung, label, on, ctrl in EXTRA_CONTRASTS_10:
        block.setdefault(after, {after}).add(corpus)
        last = max(i for i, c in enumerate(out) if c[0] in block[after])
        out.insert(last + 1, (corpus, label, rung, on, ctrl))
    return out


def _path(pattern: str, seed: int) -> Path:
    if seed in NEW_SEEDS and not pattern.startswith("dh2_kt_ladder_"):
        prefix = pattern[: -len("_s{seed}.csv")]
        pattern = _ladder(f"xes-ref_{FROZEN_TO_XES_REF[prefix]}")
    return TABLES / pattern.format(seed=seed)


def _available(pattern: str | None) -> bool:
    if pattern is None:
        # Observed-incidence package: a7 file on the first five seeds,
        # xes-ref time-on-observed vs incidence-on on the new ones.
        if len(SEEDS) == 5:
            return True
        return all(
            _path(_ladder(f"xes-ref_{arm}"), s).exists()
            for arm in ("time-on-observed", "incidence-on")
            for s in NEW_SEEDS
        )
    return all(_path(pattern, s).exists() for s in SEEDS)


def _read(pattern: str) -> dict[int, dict]:
    return {seed: _read_one(_path(pattern, seed)) for seed in SEEDS}


def _read_one(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    return {
        "val_auc": float(row["val_auc"]),
        "auc": float(row["dh2_kt_auc"]),
        "n": int(row["n_predictions"]),
        "batch": int(row["batch_size"]),
        "epochs": int(row["epochs"]),
        "file": path.name,
    }


def _outcome(credited: bool, deltas: list[float]) -> str:
    if credited:
        return "Credited"
    if all(d > 0 for d in deltas):
        return "Consistent-positive, not credited"
    return "Unsupported"


def _outcome_lcb_middle(lcb: float) -> str:
    if lcb >= TAU:
        return "Credited"
    if lcb > 0:
        return "Consistent-positive, not credited"
    return "Unsupported"


def _a7_both() -> tuple[dict[int, float], dict[int, dict]]:
    # QKC-T (observed incidence, both-mode gap) and its no-time control, as
    # printed in table_a7_query_dt_seeds.tex. The a7 control is the
    # qkc_cm_observed run of the same seed (identical validation AUC).
    path = TABLES / "a7_query_dt_ablation.csv"
    on: dict[int, float] = {}
    ctrl: dict[int, dict] = {}
    on_col = "val_auc" if METRIC == "val_auc" else "test_auc"
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["time_gap_mode"] != "both":
                continue
            seed = int(row["seed"])
            on[seed] = float(row[on_col])
            ctrl[seed] = _read_one(_path(_frozen("qkc_cm_observed"), seed))
            ctrl[seed]["val_auc"] = float(row["twin_val_auc"])
    for seed in SEEDS:
        if seed in NEW_SEEDS:
            on[seed] = _read_one(_path(_ladder("xes-ref_time-on-observed"), seed))[METRIC]
            ctrl[seed] = _read_one(_path(_ladder("xes-ref_incidence-on"), seed))
    return on, ctrl


_HEAT_LABEL = {
    "Q$\\leftarrow$KC incidence": "Incidence",
    "Package, observed incidence": "Pkg, obs.",
    "Package, zero incidence": "Pkg, zero",
    "Aligned gap vs.\\ T-zero": "Gap vs 0",
    "Aligned gap vs.\\ T-misaligned": "Gap vs mis.",
    "Package vs.\\ param-padded no-time": "Pkg vs pad",
    "Aligned gap vs.\\ T-shuffled": "Gap vs shuf.",
    "Aligned gap vs.\\ T-boundary": "Gap vs bnd.",
    "T-boundary vs.\\ T-zero": "Bnd vs 0",
}
_HEAT_FILL = {
    "Credited": "cbBlue!35",
    "Consistent-positive, not credited": "cbOrange!50",
    "Unsupported": "black!16",
}
_HEAT_MARK = {
    "Credited": r"$\checkmark$",
    "Consistent-positive, not credited": r"$\circ$",
    "Unsupported": r"$\times$",
}


def _write_heatmap(rows: list[dict]) -> None:
    keep = [r for r in rows if not r["corpus"].startswith(ASSIST_1000S)]
    contrasts: list[str] = []
    corpora: list[str] = []
    for row in keep:
        if row["contrast"] not in contrasts:
            contrasts.append(row["contrast"])
        if row["corpus"] not in corpora:
            corpora.append(row["corpus"])
    lookup = {(r["corpus"], r["contrast"]): r for r in keep}
    history = "Junyi (50k) history-only"
    with_history = bool(LABEL_SUFFIX) and HISTORY_JSON.exists()
    if with_history:
        for h in json.loads(HISTORY_JSON.read_text(encoding="utf-8"))["rows"]:
            if h["contrast"] in HISTORY_TO_MAP:
                v = h["val"]
                lookup[(history, HISTORY_TO_MAP[h["contrast"]])] = {
                    "mean_delta_val": v["mean"],
                    "outcome_abs": v["outcome_abs"],
                    "outcome_ci": v["outcome_ci"],
                }
        corpora.insert(corpora.index("Junyi (50k)") + 1, history)
    col_w, row_h = 1.38, 0.46
    label_x = 0.0
    x0 = 0.55 + col_w / 2
    lines = [
        "% Generated by scripts/51_transfer_credit_tables.py",
        "\\begin{figure}[H]",
        "\\centering",
        "\\definecolor{cbBlue}{HTML}{0072B2}",
        "\\definecolor{cbOrange}{HTML}{E69F00}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tikzpicture}[font=\\scriptsize\\sffamily, every node/.style={inner sep=1.5pt}]",
    ]
    header_y = 0.28 + len(corpora) * row_h
    for j, contrast in enumerate(contrasts):
        label = _HEAT_LABEL.get(contrast, contrast)
        x = x0 + j * col_w
        lines.append(
            f"\\node[rotate=50, anchor=south west, inner sep=0pt] at ({x-0.28},{header_y}) {{{label}}};"
        )
    for i, corpus in enumerate(corpora):
        y = (len(corpora) - 1 - i) * row_h
        lines.append(
            f"\\node[anchor=east, font=\\footnotesize\\sffamily] at ({label_x},{y}) "
            f"{{{corpus if corpus == history else _show_corpus(corpus)}}};"
        )
        for j, contrast in enumerate(contrasts):
            row = lookup.get((corpus, contrast))
            x = x0 + j * col_w
            if row is None:
                lines.append(
                    f"\\node[minimum width={col_w-0.08}cm, "
                    f"minimum height={row_h-0.06}cm] at ({x},{y}) {{}};"
                )
                continue
            fill = _HEAT_FILL[row["outcome_ci"]]
            mark = _HEAT_MARK[row["outcome_ci"]]
            text = _milli(row["mean_delta_val"])
            if row["outcome_abs"] != row["outcome_ci"]:
                text += "$^{\\dagger}$"
            lines.append(
                f"\\node[draw, fill={fill}, minimum width={col_w-0.08}cm, "
                f"minimum height={row_h-0.06}cm] at ({x},{y}) {{{mark}\\,{text}}};"
            )
    lines += [
        "\\end{tikzpicture}}",
        (
            f"\\caption{{Credit outcomes; row names match Tables~\\ref{{tab:credit-transfer{LABEL_SUFFIX}}},"
            f" \\ref{{tab:credit-transfer{LABEL_SUFFIX}-b}}, and~\\ref{{tab:junyi-history}}."
            if with_history
            else f"\\caption{{Credit outcomes; row names match Tables~\\ref{{tab:credit-transfer{LABEL_SUFFIX}}}"
            f" and~\\ref{{tab:credit-transfer{LABEL_SUFFIX}-b}}."
        ),
        "A check on blue is credited, a circle on orange is consistent-positive",
        "(not credited), and a cross on grey is unsupported. Column labels:",
        "Incidence; Pkg, obs.\\ and Pkg, zero, the package with observed and with",
        "zeroed incidence; Gap vs 0, mis., shuf., and bnd., the aligned gap",
        "against T-zero, T-misaligned, T-shuffled, and T-boundary; Pkg vs pad,",
        "the package against the parameter-padded control; Bnd vs 0, T-boundary",
        "against T-zero. The number is the mean paired difference in units of",
        "$10^{-3}$ validation AUC. An empty cell was not run.",
        *(
            []
            if all(r["outcome_abs"] == r["outcome_ci"] for r in keep)
            else [
                "Colours follow the interval rule; a dagger marks a cell where",
                "the seed rule gives consistent-positive instead.",
            ]
        ),
        *JUNYI_NOTE,
        *(
            ["The history-only row feeds the Junyi gap to the recurrent input only."]
            if with_history
            else []
        ),
        f"The $1{{,}}000$\\,s export is Table~\\ref{{tab:credit-assist-resolution{LABEL_SUFFIX}}}.}}",
        f"\\label{{fig:credit-heatmap{LABEL_SUFFIX}}}",
        "\\end{figure}",
        "",
    ]
    OUT_HEAT.write_text("\n".join(lines), encoding="utf-8")


def _milli(value: float) -> str:
    return f"{1000 * value:+.2f}"


def _one_outcome(row: dict, short: dict[str, str]) -> str:
    n = len(SEEDS)
    if row["outcome_abs"] == row["outcome_ci"]:
        return f"{short[row['outcome_abs']]} ${row['pass_abs']}/{n}$"
    return (
        f"{short[row['outcome_abs']]}/{short[row['outcome_ci']]} "
        f"${row['pass_abs']}/{n}$"
    )


def _rules_sentence(rows: list[dict]) -> tuple[str, str]:
    """The agreement sentence split across two caption source lines."""
    if all(r["outcome_abs"] == r["outcome_ci"] for r in rows):
        return "The outcome is the same under both", "rules."
    return "Where the rules differ, the outcome", "is given as seed rule/interval rule."


def _write_resolution_table(rows: list[dict], short: dict[str, str]) -> None:
    by_key = {(r["corpus"], r["contrast"]): r for r in rows}
    lines = [
        "\\begin{table}[H]",
        "\\caption{ASSISTments~2012 on the original export, whose start times",
        "are stored at 1{,}000\\,s resolution. The one-second re-export of the",
        f"same rows is in Table~\\ref{{tab:credit-transfer{LABEL_SUFFIX}}}. Fold~0, {SEED_WORD[len(SEEDS)]} paired",
        "seeds. Differences are in units of $10^{-3}$ validation AUC; the",
        f"bracket is the lower 95\\% bound. {_rules_sentence(rows)[0]}",
        f"{_rules_sentence(rows)[1]} Corpus, the corpus configuration;",
        "Reference, the reference training recipe; Q$\\leftarrow$KC incidence,",
        "question--knowledge-component incidence; package, the gap input with",
        "the maps that read it; T-zero and T-misaligned, the gap map fed zeros",
        "or gaps permuted within each learner; Cons.-pos., consistent-positive",
        "(not credited).}",
        f"\\label{{tab:credit-assist-resolution{LABEL_SUFFIX}}}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}llrl@{}}",
        "\\toprule",
        "Recipe & Contrast & Mean [lower] & Outcome \\\\",
        "\\midrule",
    ]
    for recipe, _new, old in (
        ("Corpus", "ASSIST2012", ASSIST_1000S),
        ("Reference", "ASSIST2012 ref.", ASSIST_1000S + " ref."),
    ):
        first = True
        for label, _rung, _on, _ctrl in ASSIST_ARMS:
            row = by_key.get((old, label))
            if row is None:
                continue
            lines.append(
                f"{recipe if first else ''} & {label} & "
                f"${_milli(row['mean_delta_val'])}$ [${_milli(row['lcb95_delta_val'])}$] & "
                f"{_one_outcome(row, short)} \\\\"
            )
            first = False
        if recipe == "Corpus":
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    OUT_TEX_RES.write_text("\n".join(lines), encoding="utf-8")


def _write_test_table(rows: list[dict], short: dict[str, str]) -> None:
    val = {
        (r["corpus"], r["contrast"]): r
        for r in json.loads(OUT_JSON.read_text(encoding="utf-8"))["rows"]
    }
    changed = [
        r for r in rows
        if (r["outcome_abs"], r["outcome_ci"])
        != (val[(r["corpus"], r["contrast"])]["outcome_abs"], val[(r["corpus"], r["contrast"])]["outcome_ci"])
    ]
    n = len(SEEDS)
    n_ci = sum(r["outcome_ci"] != val[(r["corpus"], r["contrast"])]["outcome_ci"] for r in rows)
    n_abs = sum(r["outcome_abs"] != val[(r["corpus"], r["contrast"])]["outcome_abs"] for r in rows)
    lines = [
        f"% Generated by scripts/51_transfer_credit_tables.py --n-seeds {n} --metric test",
        "\\begin{table}[tbp]",
        f"\\caption{{Credit map recomputed on test-only AUC, which plays no part",
        "in early stopping. Same pairs, seeds, threshold, and rules as",
        f"Table~\\ref{{tab:credit-transfer{LABEL_SUFFIX}}}. Of {len(rows)} contrasts, {n_ci} change",
        f"their interval-rule outcome and {n_abs} their seed-rule outcome; only",
        "the rows that change are listed, with the validation outcome first.",
        "Differences are $10^{-3}$ test-only AUC; the bracket is the lower 95\\%",
        "bound; outcomes are seed rule/interval rule, with C credited,",
        "P consistent-positive (not credited), and U unsupported. ref., the",
        "reference training recipe (otherwise the corpus configuration).}",
        f"\\label{{tab:credit-test{LABEL_SUFFIX}}}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}llrll@{}}",
        "\\toprule",
        "Setting & Contrast & Test mean [lower] & Validation & Test \\\\",
        "\\midrule",
    ]
    letter = {"Credited": "C", "Consistent-positive, not credited": "P", "Unsupported": "U"}
    for r in changed:
        v = val[(r["corpus"], r["contrast"])]
        lines.append(
            f"{r['corpus']} & {r['contrast']} & ${_milli(r['mean_delta_val'])}$ "
            f"[${_milli(r['lcb95_delta_val'])}$] & "
            f"{letter[v['outcome_abs']]}/{letter[v['outcome_ci']]} & "
            f"{letter[r['outcome_abs']]}/{letter[r['outcome_ci']]} \\\\"
        )
    if not changed:
        lines.append("\\multicolumn{5}{@{}l}{No outcome changes.} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    out = OUT_TEX.with_name(f"table_credit_test{LABEL_SUFFIX.replace('-', '_')}.tex")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(changed)} changed rows)")


def _configure(n_seeds: int) -> None:
    global SEEDS, T_975, LABEL_SUFFIX, OUT_JSON, OUT_TEX, OUT_TEX_RES, OUT_HEAT
    if n_seeds == 5:
        return
    SEEDS = SEEDS_5 + NEW_SEEDS
    T_975 = T_975_DF[n_seeds]
    LABEL_SUFFIX = "-10seed"
    OUT_JSON = OUT_JSON.with_name(OUT_JSON.stem + "_10seed.json")
    OUT_TEX = OUT_TEX.with_name(OUT_TEX.stem + "_10seed.tex")
    OUT_TEX_RES = OUT_TEX_RES.with_name(OUT_TEX_RES.stem + "_10seed.tex")
    OUT_HEAT = OUT_HEAT.with_name(OUT_HEAT.stem + "_10seed.tex")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, choices=(5, 10), default=5)
    parser.add_argument("--metric", choices=("val", "test"), default="val")
    args = parser.parse_args()
    _configure(args.n_seeds)
    global METRIC
    METRIC = "val_auc" if args.metric == "val" else "auc"
    n = len(SEEDS)
    rows = []
    pending = []
    for corpus, label, rung, on_pat, ctrl_pat in _contrasts():
        if not (_available(on_pat) and _available(ctrl_pat)):
            pending.append(f"{corpus}: {label}")
            continue
        if on_pat is None:
            on_val, ctrl = _a7_both()
        else:
            ctrl = _read(ctrl_pat)
            on_val = {s: v[METRIC] for s, v in _read(on_pat).items()}
        deltas = [on_val[s] - ctrl[s][METRIC] for s in SEEDS]
        sd_ctrl = statistics.stdev(ctrl[s][METRIC] for s in SEEDS)
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
                "outcome_ci_lcb_middle": _outcome_lcb_middle(lcb),
                "control_batch": ctrl[SEEDS[0]]["batch"],
                "control_epochs": ctrl[SEEDS[0]]["epochs"],
                "n_scored": ctrl[SEEDS[0]]["n"],
            }
        )

    payload = {
        "metric": METRIC,
        "tau_abs": TAU,
        f"t_975_df{n - 1}": T_975,
        "interval_rule": f"mean - t_975 * SD(delta) / sqrt({n}) >= tau",
        "rows": rows,
        "pending": pending,
    }
    short = {"Credited": "Credited", "Consistent-positive, not credited": "Cons.-pos.", "Unsupported": "Unsupp."}
    if METRIC != "val_auc":
        out_json = OUT_JSON.with_name(OUT_JSON.stem + "_test.json")
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_test_table(rows, short)
        print(f"wrote {out_json}")
        return 0
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    map_rows = [r for r in rows if not r["corpus"].startswith(ASSIST_1000S)]
    # The map with its legend is too tall for one page: split into two floats.
    split = SPLIT_BEFORE if LABEL_SUFFIX else "ASSIST2012"
    b_desc = (
        "ASSISTments~2012 folds~1--2 and reference recipe, and Junyi."
        if LABEL_SUFFIX
        else "ASSISTments~2012 and Junyi."
    )
    cut = next((i for i, r in enumerate(map_rows) if r["corpus"] == split), len(map_rows))
    parts = [map_rows[:cut], map_rows[cut:]] if cut < len(map_rows) else [map_rows]
    label = f"tab:credit-transfer{LABEL_SUFFIX}"
    lines: list[str] = []
    for k, part in enumerate(parts):
        if k == 0:
            caption = [
                "\\caption{Credit map. Differences are $10^{-3}$ validation AUC; the",
                f"bracket is the lower 95\\% bound. {_rules_sentence(rows)[0]}",
                f"{_rules_sentence(rows)[1]} Abbreviations are defined below the table."
                + (f" Continued in Table~\\ref{{{label}-b}}.}}" if len(parts) > 1 else "}"),
                f"\\label{{{label}}}",
            ]
        else:
            caption = [
                f"\\caption{{Credit map, continued from Table~\\ref{{{label}}}:",
                f"{b_desc} Units, bracket, and outcome format as there;",
                "abbreviations below.}",
                f"\\label{{{label}-b}}",
            ]
        lines += [
            "\\begin{table}[tbp]",
            *caption,
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\renewcommand{\\arraystretch}{0.84}",
            "\\begin{tabular}{@{}llrl@{}}",
            "\\toprule",
            "Setting & Contrast & Mean [lower] & Outcome \\\\",
            "\\midrule",
        ]
        last = None
        for r in part:
            corpus = _show_corpus(r["corpus"]) if r["corpus"] != last else ""
            if last is not None and r["corpus"] != last:
                lines.append("\\addlinespace[1pt]")
            last = r["corpus"]
            lines.append(
                f"{corpus} & {r['contrast']} & ${_milli(r['mean_delta_val'])}$ "
                f"[${_milli(r['lcb95_delta_val'])}$] & {_one_outcome(r, short)} \\\\"
            )
        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            "\\smallskip",
            "\\parbox{0.92\\linewidth}{\\scriptsize",
            *(MAP_LEGEND if k == 0 else MAP_LEGEND_B),
            *(JUNYI_NOTE if any(r["corpus"].startswith("Junyi") for r in part) else []),
            "}",
            "\\end{table}",
            "",
        ]
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
    _write_resolution_table(rows, short)
    _write_heatmap(rows)

    for r in rows:
        print(
            f"{r['corpus']:15} {r['contrast']:40} mean={r['mean_delta_val']:+.6f} "
            f"min={r['min_delta_val']:+.6f} sd_delta={r['sd_delta_val']:.6f} "
            f"lcb95={r['lcb95_delta_val']:+.6f} seed={r['pass_abs']}/{n} {r['outcome_abs']} | "
            f"interval {r['outcome_ci']}"
        )
    for p in pending:
        print(f"PENDING {p}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
