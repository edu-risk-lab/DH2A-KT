"""GreyKT-v3: Reliability-Aware Hybrid Knowledge Tracing.

Hybrid black-box + white-box knowledge tracing. DH2-KT
(``dh2a_kt/models/dh2_kt.py``) is a pure black-box hypergraph neural
predictor -- the audited, leakage-controlled prerequisite graph (E_pre,
from the companion P0 paper) is consumed only as *structure* feeding a
learned encoder. GreyKT adds a second, fully rule-based ("white-box")
branch that reads the same audited E_pre directly as an interpretable
prerequisite-mastery estimator, and fuses it with the black-box branch
through a deterministic, non-learned *reliability gate* -- the gate
compares an explicit confidence for each branch and weights accordingly,
rather than trusting one branch by default.

Revision history (each fixes a concrete weakness found in review; see
docs/greykt-design-note.md for the full discussion):

v1 -> v2: replaced a raw empirical-mean/binary-coverage white-box
estimator with Beta-Binomial shrinkage, multi-hop traversal, recency
decay, and an optional secondary gate signal.

v2 -> v3 (this version): two changes to the gate and fusion mechanics.

1. **Relative-reliability gate.** v2's gate combined a white-box
   confidence signal with an optional black-box confidence signal via
   ``min()`` -- a floor, not a comparison. v3 makes both confidences
   first-class, explicitly computed quantities and derives the gate from
   their *ratio*: ``g_B = C_B / (C_B + C_W + eps)``. This is the standard
   precision-weighted (inverse-variance-style) fusion rule: when both
   confidences are low the gate lands near 0.5 (genuine shared
   uncertainty) instead of collapsing to whichever branch happens to be
   weaker, which is what a ``min()``-based gate does.
2. **Probability-space fusion, not logit-space.** v2 fused
   ``gate * black_logit + (1 - gate) * white_logit``, a geometric mixture
   of the two branches' probabilities -- at ``gate=0.5`` this is *not* a
   50/50 blend of probabilities, which undercuts the interpretability the
   gate is supposed to provide. Since the gate is deliberately
   non-learned (see below), there is no gradient-flow argument for
   logit-space combination. v3 fuses
   ``p_GreyKT = g_B * p_B + (1 - g_B) * p_W``, an arithmetic mixture
   (standard mixture-of-experts form) where the gate value *is* the
   probability weight, directly. ``GreyKTOutput.logits`` is still
   provided (via a numerically-clamped inverse-sigmoid of the fused
   probability) purely for compatibility with training code written
   against ``BCEWithLogitsLoss``; new training code should prefer
   ``BCELoss`` (or an equivalent) directly on ``GreyKTOutput.probs``.

White-box branch, per timestep, for target concept ``c``:

    S_{u,p,t}^eff = sum_{tau < t} lambda^{t-tau} y_tau      (recency-weighted correct count on prereq p)
    N_{u,p,t}^eff = sum_{tau < t} lambda^{t-tau}             (recency-weighted attempt count on prereq p)
    M_{u,p,t}     = (alpha0 + S_eff) / (alpha0 + beta0 + N_eff)   (Beta-Binomial posterior mastery of prereq p)
    C_{u,p,t}     = N_eff / (N_eff + kappa_w)                     (per-prerequisite confidence)
    w_{pc}        = rho^(d(p,c) - 1)                              (hop-distance decay, d = hop distance, bounded by max_hops)

    p_W = sum_p w_pc * C_{u,p,t} * M_{u,p,t}  /  sum_p w_pc * C_{u,p,t}   (falls back to prior_mean if the denominator is ~0)
    C_W = sum_p w_pc * N_{u,p,t}^eff  /  (sum_p w_pc * N_{u,p,t}^eff + kappa_w)   (aggregate white-box confidence)

Black-box branch confidence, per concept, precomputed once from the
training corpus (a fixed statistic, not learned via gradient descent):

    C_B(c) = N_c^train / (N_c^train + kappa_b)

Reliability gate and fusion:

    g_B         = C_B / (C_B + C_W + eps)
    p_GreyKT    = g_B * p_B + (1 - g_B) * p_W

where ``p_B = sigmoid(black_logits)``.

Both branches, both confidences, the gate, and the fused probability are
all exposed on :class:`GreyKTOutput` -- the point of a grey-box model is
that a reader can decompose any single prediction into "how much
black-box, how much white-box, and why" after the fact.

v3 -> v4 (optional, config-gated, both OFF by default)
------------------------------------------------------

Two further refinements are implemented but disabled by default, so v3
stays exactly reproducible and each increment can be measured on its own
in an ablation ladder (v3 / v4a / v4b / v4a+b). Enable one only if it
earns its complexity on real validation data.

**v4a -- black-box probability calibration** (``temperature``).
Probability-space fusion assumes p_B and p_W sit on a comparable scale.
By default they do not: neural predictors are typically overconfident,
while the Beta-Binomial white-box estimator is conservative by
construction, so a p_B of 0.9 and a p_W of 0.9 do not carry equal
evidential weight. Temperature scaling (``p_B = sigmoid(logits / T)``,
fit post-hoc on validation via ``fit_temperature()``) corrects this with
a single monotone parameter, leaving AUC exactly unchanged.

**v4b -- absolute-reliability backoff** (``use_absolute_backoff``).
The relative gate is scale-free, so ``C_B = C_W = 0.9`` (both branches
reliable) and ``C_B = C_W = 0.05`` (neither branch informed) both yield
``g_B = 0.5``. v4b adds an orthogonal absolute weight
``alpha = R/(R + kappa_r)``, ``R = C_B + C_W``, and mixes the branch
blend against a base rate: ``p = alpha*mixture + (1-alpha)*backoff_prob``.
The gate still decides *how* to split between branches; alpha decides
*how much* to trust that split at all.

**Deliberately not implemented.** Hierarchical per-concept priors,
graph-stability edge weighting, and a posterior-variance C_W are all
plausible next increments, but each targets a weakness whose real
severity is unknown until GreyKT has been run on actual data even once.
Building them now would be a fourth consecutive design round with zero
empirical results. ``dh2a_kt/diagnostics/black_confidence.py`` tests the
gate's core premise (that C_B predicts black-box error) and should be run
before any of this is worth extending.

Status: implemented and logic-checked with a torch-free numpy simulation
(scripts/_greykt_logic_check.py) because this authoring sandbox could not
install torch/torch_geometric (network-restricted; large CUDA wheel).
``tests/test_greykt.py`` mirrors ``tests/test_dh2_kt.py`` conventions and
must be run on a machine with torch + torch_geometric (e.g. the RTX 3090
host) before any numbers from this module are trusted or reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from dh2a_kt.models.dh2_kt import DH2KT, DH2KTBatch, DH2KTConfig

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


@dataclass
class GreyKTConfig:
    n_concepts: int
    n_exercises: int
    embed_dim: int = 128
    hidden_dim: int = 128
    n_hypergraph_layers: int = 2
    dropout: float = 0.2
    hyperedge_kinds: tuple[str, ...] = (
        "concept_prerequisite",
        "session",
        "discussion_thread",
        "teacher_intervention",
    )

    # --- white-box mastery estimator ---
    prior_mean: float = 0.5
    """Beta prior mean for prerequisite mastery (alpha0 / (alpha0+beta0)).
    Should be set from the training set's actual global correct-rate, not
    left at the 0.5 default, before any number from this module is
    trusted."""
    prior_strength: float = 4.0
    """Beta prior pseudo-count, alpha0 + beta0. Controls how much a
    prerequisite's own evidence must accumulate before M_{u,p,t} departs
    from prior_mean."""
    kappa_w: float | None = None
    """Pseudo-count for white-box confidence (both per-prerequisite C_p
    and the aggregate C_W). Decoupled from prior_strength by design --
    shrinkage strength and confidence smoothing need not share a value.
    Defaults to prior_strength when left unset."""
    max_hops: int = 3
    """Prerequisite walk depth (d(p,c) in the hop-decay weight). Set
    equal to the chain hyperedges' ell_max (Section 4.1 / Algorithm 1 in
    main.tex) so the white-box and black-box branches see the same
    horizon."""
    hop_decay: float = 0.5
    """rho in w_pc = rho^(d(p,c)-1): per-hop discount for ancestors beyond
    direct prerequisites."""
    recency_decay: float = 0.98
    """lambda in the EWMA sums S_eff/N_eff: per-timestep multiplicative
    decay applied to running per-concept evidence. 1.0 disables recency
    weighting."""

    # --- black-box confidence / reliability gate ---
    kappa_b: float = 4.0
    """Pseudo-count for black-box confidence, C_B(c) = N_c^train /
    (N_c^train + kappa_b). N_c^train is supplied per batch via
    GreyKTBatch.concept_train_freq (a fixed statistic computed once from
    the training corpus, not learned)."""
    black_confidence_mode: str = "frequency"
    """How C_B is computed: ``frequency`` (N_c^train lookup; NOT_SUPPORTED
    on XES3G5M fold 0), ``mc_dropout`` (batch tensor; NOT_SUPPORTED), or
    ``predictive`` (|2 p_B - 1|, detached; SUPPORTED). Predictive C_B is
    output-confidence, not training reliability."""
    gate_eps: float = 1e-6
    """epsilon in g_B = C_B / (C_B + C_W + eps), guards the case where
    both confidences are exactly zero (no white-box evidence and no
    black-box training-frequency data)."""

    # --- v4a: black-box probability calibration (off by default) ---
    temperature: float = 1.0
    """Temperature scaling applied to the black-box logits before the
    sigmoid: p_B = sigmoid(black_logits / T). Probability-space fusion
    assumes both branches' probabilities are on a comparable scale, but a
    neural predictor is typically overconfident while the white-box
    Beta-Binomial estimator is, by construction, conservative -- so a raw
    p_B of 0.9 and a white-box p_W of 0.9 do not carry the same evidential
    weight, and averaging them is not meaningful. Temperature scaling is
    the standard single-parameter post-hoc fix; it is monotone, so it
    leaves AUC exactly unchanged and isolates the calibration effect from
    any change in ranking. Fit on a held-out split with
    ``fit_temperature()`` AFTER the black-box branch is trained. T=1.0 is
    the identity (v3 behavior)."""

    # --- v4b: absolute-reliability backoff (off by default) ---
    use_absolute_backoff: bool = False
    """Enable the three-way mixture that backs off to a base rate when
    BOTH branches are un-evidenced. The relative gate g_B is scale-free:
    C_B = C_W = 0.9 (both branches highly reliable) and C_B = C_W = 0.05
    (neither branch knows anything) both yield g_B = 0.5, so v3 cannot
    distinguish confident agreement from shared ignorance. This adds an
    orthogonal absolute-reliability weight so the two cases diverge:

        R     = C_B + C_W                                  (total evidence)
        alpha = R / (R + kappa_r)                           (in [0, 1))
        p     = alpha * [g_B*p_B + (1-g_B)*p_W] + (1-alpha) * backoff_prob

    g_B still decides HOW to split between branches; alpha decides HOW
    MUCH to trust that split at all versus falling back to the base rate.
    Off by default so v3 remains exactly reproducible for ablation."""
    kappa_r: float = 1.0
    """Pseudo-count in alpha = R / (R + kappa_r). Since R = C_B + C_W is
    bounded in [0, 2), kappa_r is on a different scale from kappa_b /
    kappa_w (which divide raw counts) -- a value near 1.0 puts the
    half-trust point at R = kappa_r, i.e. both confidences around 0.5.
    Only used when ``use_absolute_backoff`` is True."""
    backoff_prob: float | None = None
    """Probability to fall back to when both branches are un-evidenced.
    Defaults to ``prior_mean`` (the training base rate) when None. Only
    used when ``use_absolute_backoff`` is True."""

    logit_eps: float = 1e-4
    """Clamp for the fused probability before an inverse-sigmoid,
    when exposing GreyKTOutput.logits for BCEWithLogitsLoss-style
    training code."""

    def to_dh2kt_config(self) -> DH2KTConfig:
        return DH2KTConfig(
            n_concepts=self.n_concepts,
            n_exercises=self.n_exercises,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            n_hypergraph_layers=self.n_hypergraph_layers,
            dropout=self.dropout,
            hyperedge_kinds=self.hyperedge_kinds,
        )

    def resolved_kappa_w(self) -> float:
        return self.kappa_w if self.kappa_w is not None else self.prior_strength

    def resolved_backoff_prob(self) -> float:
        return self.backoff_prob if self.backoff_prob is not None else self.prior_mean

    def alpha0_beta0(self) -> tuple[float, float]:
        alpha0 = self.prior_mean * self.prior_strength
        beta0 = (1.0 - self.prior_mean) * self.prior_strength
        return alpha0, beta0


@dataclass
class GreyKTBatch:
    """DH2KTBatch plus the raw, directed prerequisite edges the white-box
    branch reads directly (P0's audited E_pre -- pairs, *not* the
    bounded-length chain hyperedges the black-box branch consumes), and
    an optional per-concept training-frequency table for the black-box
    confidence C_B.

    ``prereq_edge_index[0, k]`` -> ``prereq_edge_index[1, k]`` means
    concept ``prereq_edge_index[0, k]`` is a direct prerequisite of
    concept ``prereq_edge_index[1, k]``.
    """

    concept_ids: "torch.Tensor"
    exercise_ids: "torch.Tensor"
    responses: "torch.Tensor"
    hyperedge_index: dict[str, "torch.Tensor"] = field(default_factory=dict)
    concept_states: "torch.Tensor | None" = None
    prereq_edge_index: "torch.Tensor | None" = None
    concept_train_freq: "torch.Tensor | None" = None
    """Optional, shape (n_concepts,), raw non-negative counts N_c^train.
    Used only when ``black_confidence`` is omitted."""
    black_confidence: "torch.Tensor | None" = None
    """Optional per-prediction C_B, shape (B, T). Overrides the frequency
    table. Used for MC-dropout (or any other per-timestep signal). Detached
    by the caller -- this is not a learned parameter."""

    def as_dh2kt_batch(self) -> DH2KTBatch:
        return DH2KTBatch(
            concept_ids=self.concept_ids,
            exercise_ids=self.exercise_ids,
            responses=self.responses,
            hyperedge_index=self.hyperedge_index,
            concept_states=self.concept_states,
        )


if _TORCH_AVAILABLE:

    @dataclass
    class GreyKTOutput:
        """Every branch and every confidence exposed for audit."""

        probs: torch.Tensor
        """Fused probability p_GreyKT, shape (B, T) -- the primary output."""
        logits: torch.Tensor
        """Fused logits, shape (B, T, 1): a numerically-clamped inverse
        sigmoid of ``probs``, provided only for compatibility with
        BCEWithLogitsLoss-based training code. Prefer ``probs`` with
        BCELoss for new code (see module docstring, fix 2)."""
        black_probs: torch.Tensor
        """Black-box (DH2KT) branch probability p_B = sigmoid(black_logits), shape (B, T)."""
        black_logits: torch.Tensor
        """Raw black-box branch logits, shape (B, T)."""
        white_prob: torch.Tensor
        """White-box branch probability p_W, shape (B, T), in [0, 1]."""
        gate: torch.Tensor
        """Reliability gate g_B, shape (B, T), in [0, 1).
        g_B -> 1 means fully black-box; g_B -> 0 means fully white-box."""
        black_confidence: torch.Tensor
        """C_B for the current timestep's target concept, shape (B, T), in [0, 1)."""
        white_confidence: torch.Tensor
        """Aggregate C_W, shape (B, T), in [0, 1)."""
        branch_mixture: torch.Tensor
        """The two-branch blend g_B*p_B + (1-g_B)*p_W BEFORE any
        absolute-reliability backoff, shape (B, T). Equals ``probs`` when
        ``use_absolute_backoff`` is False. Kept separate so the backoff's
        contribution is measurable on its own in an ablation."""
        absolute_reliability: torch.Tensor
        """alpha = R / (R + kappa_r) with R = C_B + C_W, shape (B, T), in
        [0, 1). All-ones when ``use_absolute_backoff`` is False. Low alpha
        marks predictions where NEITHER branch had evidence -- the case a
        purely relative gate cannot express."""

    def _build_multihop_prereq_lookup(
        prereq_edge_index: torch.Tensor | None,
        n_concepts: int,
        max_hops: int,
        hop_decay: float,
    ) -> list[list[tuple[int, float]]]:
        """Weighted ancestor list per concept, via breadth-first walk over
        the *reversed* direct-prerequisite edges, bounded to ``max_hops``.

        Returns, for each concept ``c``, a list of ``(ancestor_id, w_pc)``
        pairs with ``w_pc = hop_decay ** (hop - 1)`` for the ancestor's
        shortest hop distance to ``c``. Structural (non-differentiable) by
        design -- built once per forward call, not per timestep.
        """
        direct_parents: list[list[int]] = [[] for _ in range(n_concepts)]
        if prereq_edge_index is not None and prereq_edge_index.numel() > 0:
            src = prereq_edge_index[0].tolist()
            dst = prereq_edge_index[1].tolist()
            for p, c in zip(src, dst):
                if 0 <= c < n_concepts and 0 <= p < n_concepts:
                    direct_parents[c].append(p)

        lookup: list[list[tuple[int, float]]] = [[] for _ in range(n_concepts)]
        for c in range(n_concepts):
            if not direct_parents[c]:
                continue
            visited: dict[int, int] = {}  # ancestor -> shortest hop distance
            frontier = list(direct_parents[c])
            hop = 1
            while frontier and hop <= max_hops:
                next_frontier: list[int] = []
                for node in frontier:
                    if node in visited:
                        continue
                    visited[node] = hop
                    next_frontier.extend(direct_parents[node])
                frontier = next_frontier
                hop += 1
            lookup[c] = [(anc, hop_decay ** (h - 1)) for anc, h in visited.items()]
        return lookup

    def whitebox_branch(
        concept_ids: torch.Tensor,
        responses: torch.Tensor,
        prereq_edge_index: torch.Tensor | None,
        n_concepts: int,
        prior_mean: float = 0.5,
        prior_strength: float = 4.0,
        max_hops: int = 3,
        hop_decay: float = 0.5,
        recency_decay: float = 0.98,
        kappa_w: float = 4.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-prerequisite Beta-Binomial mastery, aggregated into a
        confidence-weighted white-box probability and an aggregate white
        confidence. See the module docstring's formulas.

        For each (sample, timestep): walks up to ``max_hops`` ancestors of
        the timestep's target concept in the audited prerequisite DAG.
        For each ancestor ``p``, pools the student's own recency-weighted
        evidence on ``p`` (strictly before this timestep -- no leakage)
        into a per-prerequisite Beta-Binomial posterior mastery ``M_p``
        and confidence ``C_p``. Combines across prerequisites via a
        ``C_p``-weighted (and hop-decay-weighted) average -- each
        prerequisite's own contribution is a directly inspectable
        intermediate, which a single pooled-sufficient-statistics
        estimate would not expose as cleanly.

        Falls back to ``prior_mean`` / confidence 0 when the target
        concept has no reachable ancestors within ``max_hops``, or when
        every reachable ancestor still has zero evidence (avoids 0/0).

        Returns ``(white_prob, white_confidence)``, each shape ``(B, T)``.
        """
        device = concept_ids.device
        batch_size, seq_len = concept_ids.shape
        lookup = _build_multihop_prereq_lookup(prereq_edge_index, n_concepts, max_hops, hop_decay)
        alpha0 = prior_mean * prior_strength
        beta0 = (1.0 - prior_mean) * prior_strength

        # Running, recency-decayed evidence per (sample, concept): S_eff, N_eff.
        running_correct = torch.zeros(batch_size, n_concepts, device=device)
        running_count = torch.zeros(batch_size, n_concepts, device=device)

        white_prob = torch.full((batch_size, seq_len), float(prior_mean), device=device)
        white_confidence = torch.zeros(batch_size, seq_len, device=device)

        for t in range(seq_len):
            concept_t = concept_ids[:, t]  # (B,)
            for b in range(batch_size):
                c = int(concept_t[b].item())
                ancestors = lookup[c] if 0 <= c < n_concepts else []
                if not ancestors:
                    continue  # stays at (prior_mean, 0.0 confidence)
                anc_ids = torch.tensor([a for a, _ in ancestors], device=device, dtype=torch.long)
                w_pc = torch.tensor([w for _, w in ancestors], device=device)

                n_eff_p = running_count[b, anc_ids]
                s_eff_p = running_correct[b, anc_ids]
                m_p = (alpha0 + s_eff_p) / (alpha0 + beta0 + n_eff_p)
                c_p = n_eff_p / (n_eff_p + kappa_w)

                numerator = (w_pc * c_p * m_p).sum()
                denominator = (w_pc * c_p).sum()
                pooled_n_eff = (w_pc * n_eff_p).sum()

                white_prob[b, t] = numerator / denominator if denominator > 0 else prior_mean
                white_confidence[b, t] = pooled_n_eff / (pooled_n_eff + kappa_w)

            # Recency decay, then fold in this step's own observation --
            # vectorized across the batch dimension.
            running_count *= recency_decay
            running_correct *= recency_decay
            batch_idx = torch.arange(batch_size, device=device)
            running_count.index_put_(
                (batch_idx, concept_t), running_count[batch_idx, concept_t] + 1.0
            )
            running_correct.index_put_(
                (batch_idx, concept_t), running_correct[batch_idx, concept_t] + responses[:, t]
            )

        return white_prob, white_confidence

    def black_confidence_table(
        concept_train_freq: torch.Tensor | None,
        n_concepts: int,
        kappa_b: float,
        device: "torch.device",
    ) -> torch.Tensor:
        """C_B(c) = N_c^train / (N_c^train + kappa_b) for every concept,
        shape (n_concepts,). Returns all zeros if concept_train_freq is
        not supplied (conservative default -- see GreyKTBatch docstring)."""
        if concept_train_freq is None:
            return torch.zeros(n_concepts, device=device)
        return concept_train_freq / (concept_train_freq + kappa_b)

    def reliability_gate(
        black_confidence_t: torch.Tensor, white_confidence: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """g_B = C_B / (C_B + C_W + eps): the fraction of combined
        confidence attributable to the black-box branch. Both inputs
        shape (B, T)."""
        return black_confidence_t / (black_confidence_t + white_confidence + eps)

    class GreyKT(nn.Module):
        """Reliability-aware hybrid KT: black-box DH2KT + white-box
        prerequisite rule, fused by a non-learned relative-reliability
        gate in probability space.

        ``self.black_box`` is a plain ``DH2KT`` instance, so a Tier-1
        checkpoint trained with ``dh2a_kt.models.dh2_kt.DH2KT`` loads
        directly via ``model.black_box.load_state_dict(...)``.
        """

        def __init__(self, config: GreyKTConfig):
            super().__init__()
            self.config = config
            self.black_box = DH2KT(config.to_dh2kt_config())

        def forward(self, batch: GreyKTBatch) -> GreyKTOutput:
            black_logits = self.black_box(batch.as_dh2kt_batch()).squeeze(-1)  # (B, T)
            # v4a: temperature scaling (identity at T=1.0). Monotone, so the
            # black-box branch's ranking -- and its AUC -- are unchanged.
            black_probs = torch.sigmoid(black_logits / self.config.temperature)

            white_prob, white_confidence = whitebox_branch(
                concept_ids=batch.concept_ids,
                responses=batch.responses,
                prereq_edge_index=batch.prereq_edge_index,
                n_concepts=self.config.n_concepts,
                prior_mean=self.config.prior_mean,
                prior_strength=self.config.prior_strength,
                max_hops=self.config.max_hops,
                hop_decay=self.config.hop_decay,
                recency_decay=self.config.recency_decay,
                kappa_w=self.config.resolved_kappa_w(),
            )

            if self.config.black_confidence_mode == "predictive":
                # Stop-gradient: C_B is a diagnostic function of p_B, not a
                # learned knob. Leaving it in the graph would let training
                # push p toward 0.5 (or the extremes) solely to retune the
                # gate rather than to predict better.
                black_confidence_t = (2.0 * black_probs - 1.0).abs().detach()
            elif batch.black_confidence is not None:
                black_confidence_t = batch.black_confidence
            else:
                black_conf_lookup = black_confidence_table(
                    batch.concept_train_freq,
                    self.config.n_concepts,
                    self.config.kappa_b,
                    black_logits.device,
                )
                black_confidence_t = black_conf_lookup[
                    batch.concept_ids.clamp(0, self.config.n_concepts - 1)
                ]

            gate = reliability_gate(black_confidence_t, white_confidence, self.config.gate_eps)
            branch_mixture = gate * black_probs + (1.0 - gate) * white_prob

            # v4b: back off toward the base rate when NEITHER branch has
            # evidence. Without this, g_B=0.5 is emitted both when the two
            # branches confidently agree and when both are ignorant.
            if self.config.use_absolute_backoff:
                total_reliability = black_confidence_t + white_confidence
                alpha = total_reliability / (total_reliability + self.config.kappa_r)
                fused_probs = alpha * branch_mixture + (1.0 - alpha) * self.config.resolved_backoff_prob()
            else:
                alpha = torch.ones_like(branch_mixture)
                fused_probs = branch_mixture

            eps = self.config.logit_eps
            fused_clamped = fused_probs.clamp(eps, 1.0 - eps)
            fused_logits = torch.log(fused_clamped / (1.0 - fused_clamped))

            return GreyKTOutput(
                probs=fused_probs,
                logits=fused_logits.unsqueeze(-1),
                black_probs=black_probs,
                black_logits=black_logits,
                white_prob=white_prob,
                gate=gate,
                black_confidence=black_confidence_t,
                white_confidence=white_confidence,
                branch_mixture=branch_mixture,
                absolute_reliability=alpha,
            )

    def fit_temperature(
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        max_iter: int = 200,
        lr: float = 0.01,
    ) -> float:
        """Fit the v4a temperature by minimizing NLL on a held-out split.

        Standard post-hoc temperature scaling: optimize a single scalar T
        so that ``sigmoid(logits / T)`` is calibrated. Because the map is
        monotone, AUC is provably unchanged -- only NLL/Brier/ECE move --
        which is what makes this safe to apply without re-validating
        discrimination.

        Fit this on a VALIDATION split, never on training data (the model
        is overconfident precisely on data it has memorized, so a
        train-fitted T would be far too small) and never on test.

        Returns the fitted T; assign it to ``GreyKTConfig.temperature``.
        """
        logits = logits.detach().reshape(-1).float()
        labels = labels.detach().reshape(-1).float()
        log_t = torch.zeros(1, requires_grad=True)  # T = exp(log_t) keeps T > 0
        optimizer = torch.optim.Adam([log_t], lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(max_iter):
            optimizer.zero_grad()
            loss = loss_fn(logits / torch.exp(log_t), labels)
            loss.backward()
            optimizer.step()
        return float(torch.exp(log_t).item())

    def expected_calibration_error(
        probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10
    ) -> float:
        """Standard binned ECE. Run this on black-only, white-only, and
        fused predictions separately on a held-out split before trusting
        the fusion -- a well-calibrated black-box branch and a
        well-calibrated white-box branch do not guarantee a well-calibrated
        *fused* prediction."""
        probs = probs.detach().reshape(-1)
        labels = labels.detach().reshape(-1).float()
        bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
        ece = torch.zeros(1, device=probs.device)
        n = probs.numel()
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            in_bin = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
            if in_bin.sum() == 0:
                continue
            bin_conf = probs[in_bin].mean()
            bin_acc = labels[in_bin].mean()
            ece += (in_bin.sum().float() / n) * (bin_conf - bin_acc).abs()
        return float(ece.item())

    def negative_log_likelihood(probs: torch.Tensor, labels: torch.Tensor, eps: float = 1e-7) -> float:
        """Mean binary NLL (log loss). Complements ECE: NLL penalizes both
        miscalibration and poor discrimination in one number, so it can
        move even when AUC (a ranking-only metric) does not."""
        p = probs.detach().reshape(-1).clamp(eps, 1.0 - eps)
        y = labels.detach().reshape(-1).float()
        nll = -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))
        return float(nll.mean().item())

    def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
        """Mean squared error between predicted probability and the
        binary label -- a proper scoring rule combining calibration and
        sharpness."""
        p = probs.detach().reshape(-1)
        y = labels.detach().reshape(-1).float()
        return float(((p - y) ** 2).mean().item())

    def stratified_metrics_2d(
        fused_probs: torch.Tensor,
        black_probs: torch.Tensor,
        black_confidence: torch.Tensor,
        white_confidence: torch.Tensor,
        labels: torch.Tensor,
        n_buckets: int = 3,
    ) -> dict[str, dict[str, float]]:
        """The direct test of GreyKT's central hypothesis, stratified
        jointly along BOTH confidence axes -- concept_train_support (C_B)
        and prerequisite_evidence (C_W) -- rather than a single collapsed
        gate value. A 1D stratification by gate alone cannot distinguish
        "both confidences low" (genuine shared uncertainty) from "one low,
        one high" (exactly the asymmetric cold-start case the design is
        meant to handle); the 2D grid keeps that distinction visible.

        Bucket boundaries are equal-width over [0, 1] on each axis
        (``n_buckets`` per axis, so ``n_buckets**2`` cells total). Reports
        AUC, NLL, and Brier score for both the fused model and the
        black-box-alone baseline in every cell, plus cell size. Cells with
        fewer than 2 samples or a single label class report NaN metrics.

        Requires sklearn (not a project dependency elsewhere; import is
        local so the rest of this module has no hard sklearn requirement).
        If the fused model does not win (lower NLL/Brier, higher AUC)
        specifically in the low-C_B/high-C_W cell, the design's central
        claim does not hold on that run -- see docs/greykt-design-note.md
        "Next steps".
        """
        from sklearn.metrics import roc_auc_score

        cb = black_confidence.detach().reshape(-1)
        cw = white_confidence.detach().reshape(-1)
        fp = fused_probs.detach().reshape(-1)
        bp = black_probs.detach().reshape(-1)
        y = labels.detach().reshape(-1)

        edges = torch.linspace(0.0, 1.0, n_buckets + 1)
        results: dict[str, dict[str, float]] = {}
        for i in range(n_buckets):
            lo_b, hi_b = edges[i], edges[i + 1]
            mask_b = (cb >= lo_b) & (cb <= hi_b if i == n_buckets - 1 else cb < hi_b)
            for j in range(n_buckets):
                lo_w, hi_w = edges[j], edges[j + 1]
                mask_w = (cw >= lo_w) & (cw <= hi_w if j == n_buckets - 1 else cw < hi_w)
                mask = mask_b & mask_w
                key = f"C_B[{lo_b:.2f},{hi_b:.2f}]_C_W[{lo_w:.2f},{hi_w:.2f}]"
                n = int(mask.sum().item())
                if n < 2 or y[mask].unique().numel() < 2:
                    results[key] = {
                        "n": float(n),
                        "fused_auc": float("nan"),
                        "black_auc": float("nan"),
                        "fused_nll": float("nan"),
                        "black_nll": float("nan"),
                        "fused_brier": float("nan"),
                        "black_brier": float("nan"),
                    }
                    continue
                y_np = y[mask].numpy()
                results[key] = {
                    "n": float(n),
                    "fused_auc": float(roc_auc_score(y_np, fp[mask].numpy())),
                    "black_auc": float(roc_auc_score(y_np, bp[mask].numpy())),
                    "fused_nll": negative_log_likelihood(fp[mask], y[mask]),
                    "black_nll": negative_log_likelihood(bp[mask], y[mask]),
                    "fused_brier": brier_score(fp[mask], y[mask]),
                    "black_brier": brier_score(bp[mask], y[mask]),
                }
        return results

else:  # pragma: no cover

    class GreyKT:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            raise ImportError(
                "GreyKT requires PyTorch (+ torch_geometric via DH2KT). "
                "Install with: pip install torch torch_geometric"
            )
