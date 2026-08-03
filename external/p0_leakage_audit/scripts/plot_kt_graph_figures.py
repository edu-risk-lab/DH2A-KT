"""Generate KT graph and cold-start figures from released fold exports.

Uses ``data/processed/<dataset>/fold_0/e_pre_train_only.csv`` when present;
falls back to schematic topology only if exports are missing.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd

from src.cold_start_report import bin_kcs_by_frequency
from src.io_utils import load_interactions, load_yaml
from src.split_checker import learner_based_folds

logger = logging.getLogger(__name__)

OUT = Path("results/figures")
DEFAULT_DATASETS = ("junyi", "assist2012", "xes3g5m")
DEFAULT_CONFIGS = {
    "junyi": "configs/junyi.yaml",
    "assist2012": "configs/assist2012.yaml",
    "xes3g5m": "configs/xes3g5m.yaml",
}


def _kc_label(kc_id: object) -> str:
    s = str(kc_id)
    return f"c{s[-4:]}" if len(s) > 4 else s


def _load_pre_edges(dataset: str, fold: int = 0) -> pd.DataFrame | None:
    path = Path("data/processed") / dataset / f"fold_{fold}" / "e_pre_train_only.csv"
    if not path.exists():
        logger.warning("Missing %s; skipping data-driven graph for %s", path, dataset)
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def _sample_pre_subgraph(edges: pd.DataFrame, *, max_edges: int = 6, seed: int = 42) -> list[tuple[str, str]]:
    """Return a readable directed chain/sample from the heaviest train-only edges."""
    ranked = edges.sort_values("weight", ascending=False).head(max_edges * 3)
    g = nx.DiGraph()
    for row in ranked.itertuples(index=False):
        g.add_edge(_kc_label(row.src_kc), _kc_label(row.dst_kc), weight=float(row.weight))
    if g.number_of_edges() == 0:
        return []
    # Prefer a short directed path for layout stability.
    nodes = list(g.nodes)
    rng = pd.Series(range(len(nodes))).sample(frac=1.0, random_state=seed).index.tolist()
    for i in rng:
        for j in rng:
            if i == j:
                continue
            try:
                path = nx.shortest_path(g, nodes[i], nodes[j])
                if 3 <= len(path) <= max_edges + 1:
                    return [(path[k], path[k + 1]) for k in range(len(path) - 1)]
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    top = ranked.head(max_edges)
    return [(_kc_label(r.src_kc), _kc_label(r.dst_kc)) for r in top.itertuples(index=False)]


def _horizontal_chain_pos(g: nx.DiGraph) -> dict[str, tuple[float, float]] | None:
    """Left-to-right positions when ``g`` is a single directed path."""
    if g.number_of_nodes() == 0:
        return {}
    sources = [n for n in g.nodes if g.in_degree(n) == 0]
    if len(sources) != 1:
        return None
    path = [sources[0]]
    seen = {sources[0]}
    while True:
        outs = list(g.successors(path[-1]))
        if not outs:
            break
        if len(outs) != 1 or outs[0] in seen:
            return None
        path.append(outs[0])
        seen.add(outs[0])
    if set(path) != set(g.nodes):
        return None
    return {node: (float(i), 0.0) for i, node in enumerate(path)}


def _draw(edges: list[tuple[str, str]], title: str, path: Path) -> None:
    g = nx.DiGraph()
    g.add_edges_from(edges)
    pos = _horizontal_chain_pos(g)
    if pos is None:
        pos = nx.spring_layout(g, seed=42, k=1.4)
        figsize = (4.2, 3.2)
        rad = 0.08
    else:
        # Compact schematic: wide and short to cut float height in the paper.
        figsize = (5.4, 1.35)
        rad = 0.0
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(g, pos, node_color="#4C72B0", node_size=620, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(
        g,
        pos,
        edge_color="#555555",
        arrows=True,
        arrowsize=14,
        width=1.4,
        connectionstyle=f"arc3,rad={rad}",
        ax=ax,
        min_source_margin=8,
        min_target_margin=8,
    )
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    if rad == 0.0:
        ys = [y for _, y in pos.values()]
        ax.set_ylim(min(ys) - 0.55, max(ys) + 0.55)
    fig.tight_layout(pad=0.15)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _temporal_neighbor_edges(train: pd.DataFrame, center_kc: object, *, top_k: int = 5) -> list[tuple[str, str, float]]:
    """Train-only temporal predecessor/successor counts around ``center_kc``."""
    counts: Counter[tuple[object, object]] = Counter()
    for _, g in train.sort_values(["user_id", "timestamp"]).groupby("user_id"):
        seq = g["kc_id"].tolist()
        for i, kc in enumerate(seq):
            if kc != center_kc:
                continue
            if i > 0:
                counts[(seq[i - 1], center_kc)] += 1
            if i + 1 < len(seq):
                counts[(center_kc, seq[i + 1])] += 1
    ranked = counts.most_common(top_k * 2)
    return [(a, b, float(w)) for (a, b), w in ranked[: top_k * 2]]


def _pick_cold_kc_and_learner(
    train: pd.DataFrame,
    test: pd.DataFrame,
    bins: tuple[tuple[int, int], ...],
    sim_edges: pd.DataFrame | None,
) -> tuple[object, int, pd.DataFrame]:
    strata = bin_kcs_by_frequency(train, bins)
    very_cold = strata.loc[strata["stratum"] == "very_cold", "kc_id"].tolist()
    sim_touch = set()
    if sim_edges is not None and not sim_edges.empty:
        sim_touch = set(sim_edges["src_kc"]) | set(sim_edges["dst_kc"])

    best: tuple[float, object, int, pd.DataFrame] | None = None
    for kc in very_cold:
        if kc not in set(test["kc_id"]):
            continue
        learner_counts = test.loc[test["kc_id"] == kc].groupby("user_id").size()
        if learner_counts.empty:
            continue
        uid = int(learner_counts.idxmax())
        n_hits = int(learner_counts.max())
        seq = test.loc[test["user_id"] == uid].sort_values("timestamp")
        nbr = len(_temporal_neighbor_edges(train, kc))
        sim_bonus = 5.0 if kc in sim_touch else 0.0
        score = n_hits * 100.0 + nbr * 5.0 + sim_bonus - len(seq) * 0.01
        if best is None or score > best[0]:
            best = (score, kc, uid, seq)

    if best is None:
        raise RuntimeError("Could not find a very-cold KC with test-fold coverage.")

    _, kc, uid, seq = best
    logger.info("Cold-start panel: kc=%s learner=%s seq_len=%d cold_hits=%d", kc, uid, len(seq), int((seq["kc_id"] == kc).sum()))
    return kc, uid, seq


def _ego_edges_for_kc(
    center_kc: object,
    train: pd.DataFrame,
    pre_edges: pd.DataFrame | None,
    sim_edges: pd.DataFrame | None,
    *,
    max_neighbors: int = 6,
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Build a fold-0 ego graph: E_pre, E_sim (if present), plus train transitions."""
    labels: dict[object, str] = {center_kc: _kc_label(center_kc)}
    weighted: Counter[tuple[str, str]] = Counter()

    def _add(src: object, dst: object, w: float = 1.0) -> None:
        labels.setdefault(src, _kc_label(src))
        labels.setdefault(dst, _kc_label(dst))
        weighted[(labels[src], labels[dst])] += w

    if pre_edges is not None:
        touch = pre_edges[(pre_edges["src_kc"] == center_kc) | (pre_edges["dst_kc"] == center_kc)]
        for row in touch.nlargest(max_neighbors, "weight").itertuples(index=False):
            _add(row.src_kc, row.dst_kc, float(row.weight))

    if sim_edges is not None:
        touch = sim_edges[(sim_edges["src_kc"] == center_kc) | (sim_edges["dst_kc"] == center_kc)]
        for row in touch.nlargest(max_neighbors, "weight").itertuples(index=False):
            _add(row.src_kc, row.dst_kc, float(row.weight))

    for src, dst, w in _temporal_neighbor_edges(train, center_kc, top_k=max_neighbors):
        _add(src, dst, w)

    if not weighted:
        _add(center_kc, center_kc, 1.0)  # degenerate fallback; should not happen on XES3G5M

    edges = list(weighted.keys())
    center_label = labels[center_kc]
    return edges, {"center": center_label, "labels": labels}


def _draw_ego_cold(
    edges: list[tuple[str, str]],
    center_label: str,
    *,
    kc_id: object,
    n_train: int,
    path: Path,
) -> None:
    g = nx.DiGraph()
    g.add_edges_from(edges)
    pos = nx.spring_layout(g, seed=7, k=1.6)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    node_colors = ["#D62728" if n == center_label else "#4C72B0" for n in g.nodes]
    node_sizes = [900 if n == center_label else 640 for n in g.nodes]
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(
        g,
        pos,
        edge_color="#555555",
        arrows=True,
        arrowsize=14,
        width=1.4,
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )
    ax.set_title(
        f"Ego-graph around very-cold KC {_kc_label(kc_id)} "
        f"({n_train} train hits; fold-0 E_pre + E_sim + transitions)",
        fontsize=11,
    )
    ax.legend(
        handles=[
            mpatches.Patch(color="#D62728", label="Very-cold KC (centre)"),
            mpatches.Patch(color="#4C72B0", label="Train-only neighbours"),
        ],
        loc="lower left",
        fontsize=9,
        frameon=False,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _window_around_cold(seq: pd.DataFrame, cold_kc: object, *, max_steps: int = 14) -> pd.DataFrame:
    idx = seq.index[seq["kc_id"] == cold_kc].tolist()
    if not idx:
        return seq.head(max_steps)
    center = idx[len(idx) // 2]
    loc = seq.index.get_loc(center)
    if isinstance(loc, slice):
        loc = loc.start or 0
    start = max(0, int(loc) - max_steps // 2)
    return seq.iloc[start : start + max_steps]


def _draw_learner_timeline(seq: pd.DataFrame, cold_kc: object, learner_id: int, path: Path) -> None:
    window = _window_around_cold(seq, cold_kc, max_steps=14)
    kcs = window["kc_id"].tolist()
    labels = [_kc_label(k) for k in kcs]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    xs = list(range(n))
    colors = ["#D62728" if k == cold_kc else "#4C72B0" for k in kcs]
    ax.bar(xs, [1] * n, color=colors, width=0.72, edgecolor="white", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$t_{i}$" for i in xs], fontsize=10)
    ax.set_yticks([])
    for i, lab in enumerate(labels):
        ax.text(i, 0.5, lab, ha="center", va="center", fontsize=9, color="white")
    ax.set_xlabel(
        f"Test-sequence window (learner {str(learner_id)[-6:]}, fold-0 XES3G5M)",
        fontsize=10,
    )
    ax.set_title("KC timeline with very-cold steps highlighted", fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _draw_dataset_graphs(datasets: tuple[str, ...]) -> None:
    for name in datasets:
        edges_df = _load_pre_edges(name, fold=0)
        if edges_df is None:
            logger.warning("Skipping %s overview graph (no export).", name)
            continue
        edges = _sample_pre_subgraph(edges_df)
        title = f"Train-only E_pre sample ({name}, fold 0)"
        _draw(edges, title, OUT / f"fig_kt_graph_{name}.pdf")


def _draw_xes3g5m_cold_start() -> None:
    cfg = load_yaml(Path(DEFAULT_CONFIGS["xes3g5m"]))
    df = load_interactions(Path(cfg["processed_path"]))
    ratios = tuple(cfg["split"]["ratios"])
    fold, _, splits = next(learner_based_folds(df, ratios, cfg["split"], default_seed=42))
    train, test = splits["train"], splits["test"]
    bins = tuple(tuple(b) for b in cfg["cold_start"]["bins"])
    pre = _load_pre_edges("xes3g5m", fold=fold)
    sim_path = Path("data/processed/xes3g5m") / f"fold_{fold}" / "e_sim_train_only.csv"
    sim = pd.read_csv(sim_path) if sim_path.exists() else None

    cold_kc, learner_id, seq = _pick_cold_kc_and_learner(train, test, bins, sim)
    n_train = int(train.loc[train["kc_id"] == cold_kc].shape[0])
    ego_edges, meta = _ego_edges_for_kc(cold_kc, train, pre, sim)
    _draw_ego_cold(
        ego_edges,
        meta["center"],
        kc_id=cold_kc,
        n_train=n_train,
        path=OUT / "fig_cold_kc_ego_xes3g5m.pdf",
    )
    _draw_learner_timeline(seq, cold_kc, learner_id, OUT / "fig_learner_timeline_xes3g5m.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KT graph and cold-start figures.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    OUT.mkdir(parents=True, exist_ok=True)
    _draw_dataset_graphs(DEFAULT_DATASETS)
    _draw_xes3g5m_cold_start()
    print(f"Wrote KT graph and cold-start figures to {OUT}")


if __name__ == "__main__":
    main()
