"""Bridge module — imports P0's leakage-controlled graph/audit code directly
from the vendored copy at ``external/p0_leakage_audit`` instead of
reimplementing it.

Why a bridge instead of copy-pasting P0's functions into this package:
  - Guarantees DH2A-KT always calls the *exact* code P0 used to produce the
    numbers in ``baselines/p0_reused/`` (no silent drift between the two).
  - Any bugfix/update in P0 propagates here by re-vendoring, not by manual sync.

Provenance / environment note:
  P0 is vendored as a **static, pinned snapshot** (see
  ``external/p0_leakage_audit/.p0_vendored_commit.txt`` for the exact commit
  hash) rather than a live ``git submodule``. A live submodule was attempted
  first but the mounted project filesystem in the authoring environment does
  not support the unlink/relink operations git submodule machinery needs.
  On a normal local machine you can convert this to a live submodule:

      git submodule add https://github.com/edu-risk-lab/leakage-controlled-kt-audit.git \\
          external/p0_leakage_audit_submodule
      # then point this module's _P0_ROOT at the submodule checkout instead.

Naming collision note:
  P0's internal imports use the top-level package name ``src`` (e.g.
  ``from src.io_utils import ...``). This package is named ``dh2a_kt``
  (deliberately, not ``src``) precisely so both can sit on ``sys.path``
  simultaneously without shadowing each other.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_P0_ROOT = _REPO_ROOT / "external" / "p0_leakage_audit"

if not (_P0_ROOT / "src" / "__init__.py").exists():
    raise ImportError(
        "P0 vendored source not found.\n"
        f"Expected: {_P0_ROOT / 'src'}\n"
        "If external/p0_leakage_audit is empty, re-vendor it: see "
        "docs/idea-D-plan.md section 7, or clone "
        "https://github.com/edu-risk-lab/leakage-controlled-kt-audit "
        "into external/p0_leakage_audit."
    )

if str(_P0_ROOT) not in sys.path:
    sys.path.insert(0, str(_P0_ROOT))

# --- Re-export P0's original, unmodified public functions ------------------
# Graph construction (P0 Section 3.3-3.4 / Algorithm 2)
from src.graph_builder import (  # noqa: E402
    build_q_matrix_from_train,
    infer_similarity_edges_from_q_matrix,
    infer_similarity_edges_from_train,
    infer_prerequisites_from_train,
)

# Leakage diagnostics (P0 Section 3.2 / Algorithm 1: ECR_flag, ECR_overlap, |rho|, TBMR)
from src.leakage_metrics import (  # noqa: E402
    compute_ecr_flag,
    compute_ecr_overlap,
    compute_rho_edge_outcome,
    compute_tbvr,
    compute_leakage_row,
    compute_edge_heldout_shares,
    summarize_edge_heldout_shares,
    merge_leakage_metrics_csv,
)

# DAG audit (P0 Section 3.5 / Algorithm 2, cycle pruning via Johnson's algorithm)
from src.dag_audit import (  # noqa: E402
    DAGReport,
    topological_sort,
    detect_cycles,
    prune_cycles,
    audit_dag,
)

# DDR + augmentation operators (P0 Section 3.6 / Algorithm 3)
from src.dag_disruption import (  # noqa: E402
    apply_node_drop,
    apply_edge_drop,
    apply_attribute_mask,
    apply_subgraph_sampling,
    apply_prereq_preserve,
    compute_dag_disruption_rate,
    reachability_f1,
)

# Cold-start KC stratification (P0 Section 3.7)
from src.cold_start_report import (  # noqa: E402
    MIN_DISCORDANT_PAIRS,
    bin_kcs_by_frequency,
    assign_stratum_for_kc,
    per_stratum_metrics,
)

# Ground-truth cross-validation against expert DAGs (P0 Section 3.8, Junyi)
from src.gt_cross_validation import (  # noqa: E402
    DEFAULT_K_LIST,
    load_expert_dag,
    align_kc_ids,
    compute_overlap_metrics,
    precision_recall_sweep,
    diagnose_disagreement,
)

# Learner-based temporal split protocol (P0 Section 4.1: 0.7/0.1/0.2, seeds 42-44)
from src.split_checker import (  # noqa: E402
    learner_based_split,
    learner_based_folds,
    fold_seeds,
    assert_no_user_overlap,
    assert_temporal_ordering,
)

__all__ = [
    "build_q_matrix_from_train",
    "infer_similarity_edges_from_q_matrix",
    "infer_similarity_edges_from_train",
    "infer_prerequisites_from_train",
    "compute_ecr_flag",
    "compute_ecr_overlap",
    "compute_rho_edge_outcome",
    "compute_tbvr",
    "compute_leakage_row",
    "compute_edge_heldout_shares",
    "summarize_edge_heldout_shares",
    "merge_leakage_metrics_csv",
    "DAGReport",
    "topological_sort",
    "detect_cycles",
    "prune_cycles",
    "audit_dag",
    "apply_node_drop",
    "apply_edge_drop",
    "apply_attribute_mask",
    "apply_subgraph_sampling",
    "apply_prereq_preserve",
    "compute_dag_disruption_rate",
    "reachability_f1",
    "MIN_DISCORDANT_PAIRS",
    "bin_kcs_by_frequency",
    "assign_stratum_for_kc",
    "per_stratum_metrics",
    "DEFAULT_K_LIST",
    "load_expert_dag",
    "align_kc_ids",
    "compute_overlap_metrics",
    "precision_recall_sweep",
    "diagnose_disagreement",
    "learner_based_split",
    "learner_based_folds",
    "fold_seeds",
    "assert_no_user_overlap",
    "assert_temporal_ordering",
]
