"""Tier 1 core model: DH2-KT (Dynamic Heterogeneous Hypergraph Knowledge
Tracing). docs/idea-D-plan.md Pha 2 ("Encoder + causal layer").

Interface-complete stub — the encoder/attention math is intentionally not
implemented yet (that is real multi-week model-development work per the
plan's own timeline, not something to fabricate). What IS real here: the
module boundary, the expected tensor contracts, and the dependency-optional
import guard so the rest of the package stays importable without PyTorch
Geometric installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

try:
    from torch_geometric.nn import HypergraphConv, HeteroConv  # noqa: F401
    _PYG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYG_AVAILABLE = False


@dataclass
class DH2KTConfig:
    n_concepts: int
    n_exercises: int
    embed_dim: int = 128
    hidden_dim: int = 128
    n_hypergraph_layers: int = 2
    dropout: float = 0.2
    # Relation-specific HypergraphConv, combined via HeteroConv-style merge —
    # see docs/idea-D-plan.md section 7 (tech stack) for rationale.
    hyperedge_kinds: tuple[str, ...] = (
        "concept_prerequisite", "session", "discussion_thread", "teacher_intervention",
    )


if _TORCH_AVAILABLE:

    class DH2KT(nn.Module):
        """Tier 1 model. ``forward`` raises NotImplementedError — this class
        exists to fix the public interface (config, expected inputs/outputs)
        before Pha 2 implementation work starts, per docs/idea-D-plan.md.
        """

        def __init__(self, config: DH2KTConfig):
            super().__init__()
            if not _PYG_AVAILABLE:
                raise ImportError(
                    "DH2KT requires torch_geometric. Install with: "
                    "pip install torch_geometric (see docs/idea-D-plan.md section 7)"
                )
            self.config = config
            self.concept_embed = nn.Embedding(config.n_concepts, config.embed_dim)
            self.exercise_embed = nn.Embedding(config.n_exercises, config.embed_dim)
            # TODO (Pha 2): one HypergraphConv per relation type in
            # config.hyperedge_kinds, combined via a HeteroConv-style merge.
            # TODO (Pha 2): dual-gated temporal update (per HGKT, cited in
            # Phien ban A section 3.3) to weight old vs new evidence.

        def forward(self, batch):  # noqa: ANN001
            raise NotImplementedError(
                "DH2KT.forward is not implemented — Pha 2 of docs/idea-D-plan.md. "
                "This class currently only fixes the module's public config/interface."
            )

else:  # pragma: no cover

    class DH2KT:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            raise ImportError(
                "DH2KT requires PyTorch. Install with: pip install torch (see requirements.txt)"
            )
