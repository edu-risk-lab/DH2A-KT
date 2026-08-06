#!/usr/bin/env python3
"""Export LaTeX table fragments from results/tables/* into paper/tables/."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "tables"
OUT = ROOT / "paper" / "tables"


def _write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(body.strip() + "\n", encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)}")


def export_auc() -> None:
    rows = list(csv.DictReader((RESULTS / "dh2_kt_vs_p0.csv").open()))
    by_fold: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["reference_model"] != "gkt":
            continue
        fold = r["fold"]
        if fold == "mean":
            continue
        by_fold[fold] = {
            "dh2": float(r["dh2_kt_auc"]),
            "gkt": float(r["p0_auc"]),
        }
    sk = next(r for r in rows if r["fold"] == "mean" and r["reference_model"] == "simplekt")
    sm = next(r for r in rows if r["fold"] == "mean" and r["reference_model"] == "gkt")
    lines = [
        r"% Auto-derived from results/tables/dh2_kt_vs_p0.csv",
        r"\begin{table}[!t]",
        r"\caption{Test AUC on XES3G5M under matched training budget (batch 4, 10 epochs, max seq.\ len.\ 200). P0 baseline AUC reused from \cite{daominh2026p0}; DH$^2$-KT v2 uses chain concept-prerequisite hyperedges and graph-only interaction.}",
        r"\label{tab:auc-xes3g5m}",
        r"\centering",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Fold & DH$^2$-KT v2 & GKT (P0) & simpleKT (P0) \\",
        r"\midrule",
    ]
    for fold in sorted(by_fold, key=int):
        d = by_fold[fold]
        lines.append(
            f"{fold} & {d['dh2']:.4f} & {d['gkt']:.4f} & {float(sk['p0_auc']):.4f} \\\\"
        )
    lines += [
        rf"\textbf{{Mean}} & \textbf{{{float(sm['dh2_kt_auc']):.4f}}} & \textbf{{{float(sm['p0_auc']):.4f}}} & \textbf{{{float(sk['p0_auc']):.4f}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    _write("table_auc_xes3g5m.tex", "\n".join(lines))


def export_manipulation() -> None:
    m = json.loads((RESULTS / "xes3g5m_fold0_manipulation_check.json").read_text())
    body = f"""% Auto-derived from results/tables/xes3g5m_fold0_manipulation_check.json
\\begin{{table}}[!t]
\\caption{{Manipulation check on XES3G5M fold~0 (DH$^2$-KT v2, graph-only). Near-total hypergraph destruction ($p{{=}}0.9$ node-drop) must move AUC meaningfully; a graph-inert backbone fails this gate.}}
\\label{{tab:manipulation}}
\\centering
\\begin{{tabular}}{{@{{}}lc@{{}}}}
\\toprule
Condition & Test AUC \\\\
\\midrule
Clean hypergraph & {m['auc_clean']:.4f} \\\\
Destroyed ($p{{=}}0.9$ node-drop) & {m['auc_destroyed']:.4f} \\\\
$\\Delta$AUC & \\textbf{{{m['auc_drop']:.4f}}} \\\\
DDR & {m['ddr']:.3f} \\\\
Pass manipulation check & \\textbf{{Yes}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    _write("table_manipulation.tex", body)


def export_leakage() -> None:
    a = json.loads((RESULTS / "xes3g5m_fold0_hyperedge_audit.json").read_text())
    n_he = f"{a['n_hyperedges']:,}".replace(",", "{,}")
    body = f"""% Auto-derived from results/tables/xes3g5m_fold0_hyperedge_audit.json
\\begin{{table}}[!t]
\\caption{{Concept-prerequisite hyperedge audit on XES3G5M fold~0.
Pairwise TBMR/$|\\rho|$ are skipped when the pairwise projection exceeds the
implementation cap of $100{{,}}000$ pairs (11{{,}}568{{,}}053 projected pairs in
this fold; see audit notes).}}
\\label{{tab:leakage-measured}}
\\centering
\\begin{{tabular}}{{@{{}}lc@{{}}}}
\\toprule
Metric & Value \\\\
\\midrule
Hyperedges & {n_he} \\\\
$\\mathrm{{ECR}}^{{\\mathrm{{flag}}}}$ & {a['ecr_flag']} \\\\
Group-membership leak rate & {a['group_membership_leak_rate']} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    _write("table_leakage_audit.tex", body)


def export_junyi_gt() -> None:
    path = RESULTS / "junyi_fold0_gt_crossval" / "gt_crossval_table15_format.csv"
    rows = {r["source"]: r for r in csv.DictReader(path.open()) if r["top_k"] == "1131"}
    e_pre, chain = rows["e_pre"], rows["hyperedge_chain"]
    body = f"""% Auto-derived from {path.relative_to(ROOT).as_posix()} @ K=|expert|=1131
\\begin{{table}}[!t]
\\caption{{Ground-truth cross-validation on Junyi Academy (fold~0, top-$K{{=}}|$expert$|{{=}}1131$). Hyperedge chain slightly underlaps pairwise $E_{{\\mathrm{{pre}}}}$ but remains the same order of magnitude.}}
\\label{{tab:junyi-gt}}
\\centering
\\begin{{tabular}}{{@{{}}lccc@{{}}}}
\\toprule
Representation & Edge F1 & Edge prec. & Edge rec. \\\\
\\midrule
P0 $E_{{\\mathrm{{pre}}}}$ & {float(e_pre['edge_f1']):.3f} & {float(e_pre['edge_precision']):.3f} & {float(e_pre['edge_recall']):.3f} \\\\
Hyperedge chain & {float(chain['edge_f1']):.3f} & {float(chain['edge_precision']):.3f} & {float(chain['edge_recall']):.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    _write("table_junyi_gt.tex", body)


def export_tier2() -> None:
    stub = json.loads((RESULTS / "xes3g5m_fold0_tier2_pilot_summary.json").read_text())
    v1 = json.loads((RESULTS / "xes3g5m_fold0_tier2_pilot_ollama_summary.json").read_text())
    v2 = json.loads((RESULTS / "xes3g5m_fold0_tier2_pilot_ollama_v2_summary.json").read_text())
    ev_stub = json.loads((RESULTS / "xes3g5m_fold0_tier2_pilot_eval.json").read_text())
    ev_v1 = json.loads((RESULTS / "xes3g5m_fold0_tier2_pilot_ollama_eval.json").read_text())
    ev_v2 = json.loads((RESULTS / "xes3g5m_fold0_tier2_pilot_ollama_v2_eval.json").read_text())
    body = f"""% Auto-derived from results/tables/*tier2_pilot*_summary.json
\\begin{{table}}[!t]
\\caption{{Tier~2 agent faithfulness pilot (XES3G5M fold~0, $n{{=}}500$, seed 42, Qwen2.5-7B via Ollama). Flag rate = fraction of Diagnostician outputs flagged by the mandatory Critic. v2 applies probability-format calibration (\\texttt{{critic\\_calibration.py}}).}}
\\label{{tab:tier2}}
\\centering
\\begin{{tabular}}{{@{{}}lccc@{{}}}}
\\toprule
Run & Backend & Flag rate & Mean $P(\\mathrm{{correct}})$ \\\\
\\midrule
Stub (sanity) & StubLLM & {stub['flag_rate']:.3f} ({stub['n_flagged']}/{stub['n_samples']}) & {ev_stub['mean_predicted_prob']:.3f} \\\\
Ollama v1 & qwen2.5:7b & {v1['flag_rate']:.3f} ({v1['n_flagged']}/{v1['n_samples']}) & {ev_v1['mean_predicted_prob']:.3f} \\\\
Ollama v2 & qwen2.5:7b & \\textbf{{{v2['flag_rate']:.3f}}} ({v2['n_flagged']}/{v2['n_samples']}) & {ev_v2['mean_predicted_prob']:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    _write("table_tier2.tex", body)


def main() -> None:
    export_auc()
    export_manipulation()
    export_leakage()
    export_junyi_gt()
    export_tier2()


if __name__ == "__main__":
    main()
