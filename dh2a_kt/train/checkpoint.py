"""Save/load trained Tier-1 folds for reuse in M6/M8 without retraining."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dh2a_kt.train.tier1 import TrainedFold


def save_trained_fold(path: Path, trained: TrainedFold) -> None:
    import torch

    from dh2a_kt.hyperedge.construction import Hyperedge

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = trained.model.config
    payload = {
        "model_state": trained.model.state_dict(),
        "kc_to_idx": trained.kc_to_idx,
        "item_to_idx": trained.item_to_idx,
        "clean_hyperedges": [asdict(he) for he in trained.clean_hyperedges],
        "config": {
            "n_concepts": cfg.n_concepts,
            "n_exercises": cfg.n_exercises,
            "hidden_dim": cfg.hidden_dim,
            "embed_dim": cfg.embed_dim,
            "n_hypergraph_layers": cfg.n_hypergraph_layers,
            "dropout": cfg.dropout,
            "hyperedge_kinds": list(cfg.hyperedge_kinds),
            "architecture": getattr(cfg, "architecture", "v2"),
            "diffusion_alpha": float(getattr(cfg, "diffusion_alpha", 0.5)),
            "use_questions": bool(getattr(cfg, "use_questions", False)),
            "n_lstm_layers": int(getattr(cfg, "n_lstm_layers", 1)),
            "memory_dim": int(getattr(cfg, "memory_dim", 16)),
            "max_degree": int(getattr(cfg, "max_degree", 16)),
            "graph_transport": float(getattr(cfg, "graph_transport", 0.5)),
            "event_pool": str(getattr(cfg, "event_pool", "mean")),
            "transport": str(getattr(cfg, "transport", "clique")),
            "use_hyperedge_embed": bool(getattr(cfg, "use_hyperedge_embed", False)),
            "kind_conditioned": bool(getattr(cfg, "kind_conditioned", False)),
            "recap_attention": bool(getattr(cfg, "recap_attention", False)),
            "question_kc_agg": bool(getattr(cfg, "question_kc_agg", False)),
            "question_graph": bool(getattr(cfg, "question_graph", False)),
            "question_hypergraph": bool(getattr(cfg, "question_hypergraph", False)),
        },
    }
    torch.save(payload, path)


def load_trained_fold(path: Path, *, device: str = "cpu") -> TrainedFold:
    import torch

    from dh2a_kt.hyperedge.construction import Hyperedge
    from dh2a_kt.hyperedge.indexing import hyperedge_index_from_list
    from dh2a_kt.models.dh2_kt import DH2KTConfig
    from dh2a_kt.train.tier1 import TrainedFold, build_model

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = payload["config"]
    kinds = tuple(cfg_dict.get("hyperedge_kinds", ("concept_prerequisite",)))
    architecture = str(cfg_dict.get("architecture", "v2"))
    config = DH2KTConfig(
        n_concepts=int(cfg_dict["n_concepts"]),
        n_exercises=int(cfg_dict["n_exercises"]),
        hidden_dim=int(cfg_dict["hidden_dim"]),
        embed_dim=int(cfg_dict.get("embed_dim", cfg_dict["hidden_dim"])),
        n_hypergraph_layers=int(cfg_dict["n_hypergraph_layers"]),
        dropout=float(cfg_dict.get("dropout", 0.2)),
        hyperedge_kinds=kinds,
        architecture=architecture,
        diffusion_alpha=float(cfg_dict.get("diffusion_alpha", 0.5)),
        use_questions=bool(cfg_dict.get("use_questions", False)),
        n_lstm_layers=int(cfg_dict.get("n_lstm_layers", 1)),
        memory_dim=int(cfg_dict.get("memory_dim", 16)),
        max_degree=int(cfg_dict.get("max_degree", 16)),
        graph_transport=float(cfg_dict.get("graph_transport", 0.5)),
        event_pool=str(cfg_dict.get("event_pool", "mean")),
        transport=str(cfg_dict.get("transport", "clique")),
        use_hyperedge_embed=bool(cfg_dict.get("use_hyperedge_embed", False)),
        kind_conditioned=bool(cfg_dict.get("kind_conditioned", False)),
        recap_attention=bool(cfg_dict.get("recap_attention", False)),
        question_kc_agg=bool(cfg_dict.get("question_kc_agg", False)),
        question_graph=bool(cfg_dict.get("question_graph", False)),
        question_hypergraph=bool(cfg_dict.get("question_hypergraph", False)),
    )
    model = build_model(
        config.n_concepts,
        config.n_exercises,
        hidden_dim=config.hidden_dim,
        n_hypergraph_layers=config.n_hypergraph_layers,
        dropout=config.dropout,
        hyperedge_kinds=kinds,
        architecture=architecture,
        diffusion_alpha=float(cfg_dict.get("diffusion_alpha", 0.5)),
        use_questions=config.use_questions,
        n_lstm_layers=config.n_lstm_layers,
        memory_dim=config.memory_dim,
        max_degree=config.max_degree,
        graph_transport=config.graph_transport,
        event_pool=config.event_pool,
        transport=config.transport,
        use_hyperedge_embed=config.use_hyperedge_embed,
        kind_conditioned=config.kind_conditioned,
        recap_attention=config.recap_attention,
        question_kc_agg=config.question_kc_agg,
        question_graph=config.question_graph,
        question_hypergraph=config.question_hypergraph,
    )
    model.load_state_dict(payload["model_state"])
    dev = torch.device(device)
    model = model.to(dev)
    kc_to_idx = {int(k): int(v) for k, v in payload["kc_to_idx"].items()}
    item_to_idx = {int(k): int(v) for k, v in payload["item_to_idx"].items()}
    clean_hyperedges = [Hyperedge(**he) for he in payload["clean_hyperedges"]]
    hyperedge_index = hyperedge_index_from_list(clean_hyperedges, kc_to_idx)
    return TrainedFold(
        model=model,
        eval_loader=None,  # type: ignore[arg-type] — rebuilt by pilot driver when needed
        kc_to_idx=kc_to_idx,
        item_to_idx=item_to_idx,
        clean_hyperedge_index={k: v.to(dev) for k, v in hyperedge_index.items()},
        clean_hyperedges=clean_hyperedges,
        device=dev,
    )
