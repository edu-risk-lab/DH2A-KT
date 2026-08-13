"""Tests for dh2a_kt/models/greykt.py::GreyKT (v3, "Reliability-Aware
Hybrid KT").

Mirrors tests/test_dh2_kt.py conventions. NOTE: could not be executed in
the authoring sandbox (torch/torch_geometric not installable there --
network-restricted). Run this on a machine with both installed (e.g. the
RTX 3090 host) before trusting GreyKT numbers in any table.

v3 note: the gate changed from a min()-combined confidence floor (v2) to
an explicit relative-reliability ratio g_B = C_B/(C_B+C_W+eps), and
fusion changed from logit-space to probability-space
(p_GreyKT = g_B*p_B + (1-g_B)*p_W). See the module docstring in
dh2a_kt/models/greykt.py for the full rationale.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from dh2a_kt.models.greykt import (
    GreyKT,
    GreyKTBatch,
    GreyKTConfig,
    black_confidence_table,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    reliability_gate,
    stratified_metrics_2d,
    whitebox_branch,
)


def _toy_hyperedge_index() -> dict[str, torch.Tensor]:
    return {
        "concept_prerequisite": torch.tensor(
            [[0, 1, 2, 1, 2, 3], [0, 0, 0, 1, 1, 1]],
            dtype=torch.long,
        ),
    }


def _toy_prereq_edges() -> torch.Tensor:
    # Direct prerequisite pairs (prereq -> target): 0->1, 1->2, 2->3.
    return torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)


def _make_toy_batch(
    batch_size: int = 4,
    seq_len: int = 5,
    *,
    n_concepts: int = 8,
    n_exercises: int = 6,
    with_prereqs: bool = True,
    with_train_freq: bool = False,
) -> GreyKTBatch:
    return GreyKTBatch(
        concept_ids=torch.randint(0, n_concepts, (batch_size, seq_len)),
        exercise_ids=torch.randint(0, n_exercises, (batch_size, seq_len)),
        responses=torch.randint(0, 2, (batch_size, seq_len)).float(),
        hyperedge_index=_toy_hyperedge_index(),
        prereq_edge_index=_toy_prereq_edges() if with_prereqs else None,
        concept_train_freq=torch.randint(0, 100, (n_concepts,)).float() if with_train_freq else None,
    )


def test_greykt_forward_output_shapes():
    config = GreyKTConfig(n_concepts=8, n_exercises=6, hidden_dim=32, embed_dim=32)
    model = GreyKT(config)
    batch = _make_toy_batch(with_train_freq=True)
    out = model(batch)
    assert out.probs.shape == (4, 5)
    assert out.logits.shape == (4, 5, 1)
    assert out.black_probs.shape == (4, 5)
    assert out.black_logits.shape == (4, 5)
    assert out.white_prob.shape == (4, 5)
    assert out.gate.shape == (4, 5)
    assert out.black_confidence.shape == (4, 5)
    assert out.white_confidence.shape == (4, 5)


def test_greykt_probs_gate_and_confidences_in_unit_interval():
    config = GreyKTConfig(n_concepts=8, n_exercises=6, hidden_dim=16, embed_dim=16)
    model = GreyKT(config)
    batch = _make_toy_batch(with_train_freq=True)
    out = model(batch)
    assert (out.probs >= 0.0).all() and (out.probs <= 1.0).all()
    assert (out.gate >= 0.0).all() and (out.gate <= 1.0).all()
    assert (out.white_prob >= 0.0).all() and (out.white_prob <= 1.0).all()
    assert (out.black_confidence >= 0.0).all() and (out.black_confidence < 1.0).all()
    assert (out.white_confidence >= 0.0).all() and (out.white_confidence < 1.0).all()


def test_per_prediction_black_confidence_overrides_frequency_table():
    """MC-dropout (or any per-timestep C_B) must ignore N_c^train lookup."""
    config = GreyKTConfig(n_concepts=8, n_exercises=6, hidden_dim=16, embed_dim=16)
    model = GreyKT(config)
    model.eval()
    batch = _make_toy_batch(with_train_freq=True)
    batch.black_confidence = torch.zeros(batch.concept_ids.shape, dtype=torch.float32)
    out = model(batch)
    assert torch.allclose(out.black_confidence, torch.zeros_like(out.black_confidence))


def test_predictive_mode_uses_abs_2p_minus_1():
    config = GreyKTConfig(
        n_concepts=8,
        n_exercises=6,
        hidden_dim=16,
        embed_dim=16,
        black_confidence_mode="predictive",
    )
    model = GreyKT(config)
    model.eval()
    batch = _make_toy_batch(with_train_freq=True)
    out = model(batch)
    expected = (2.0 * out.black_probs - 1.0).abs()
    assert torch.allclose(out.black_confidence, expected)


def test_predictive_confidence_is_detached_from_logits():
    """Training must not game the gate by moving p_B to change C_B."""
    config = GreyKTConfig(
        n_concepts=8,
        n_exercises=6,
        hidden_dim=16,
        embed_dim=16,
        black_confidence_mode="predictive",
    )
    model = GreyKT(config)
    model.train()
    batch = _make_toy_batch(with_train_freq=True)
    out = model(batch)
    assert out.black_probs.requires_grad
    assert not out.black_confidence.requires_grad


def test_greykt_no_prereqs_and_no_train_freq_falls_back_to_prior_prob():
    """With prereq_edge_index=None (no white-box evidence anywhere) and
    concept_train_freq=None (no black-box confidence data), both C_B and
    C_W are zero everywhere, so the gate is the eps-guarded 0/0 case
    (lands near 0, favoring white-box) and the white-box probability is
    exactly the configured prior everywhere -- the fused probability must
    equal prior_mean everywhere."""
    prior_mean = 0.37
    config = GreyKTConfig(
        n_concepts=8, n_exercises=6, hidden_dim=16, embed_dim=16, prior_mean=prior_mean
    )
    model = GreyKT(config)
    batch = _make_toy_batch(with_prereqs=False, with_train_freq=False)
    out = model(batch)

    assert torch.allclose(out.white_confidence, torch.zeros_like(out.white_confidence))
    assert torch.allclose(out.black_confidence, torch.zeros_like(out.black_confidence))
    assert torch.allclose(out.white_prob, torch.full_like(out.white_prob, prior_mean))
    assert torch.allclose(out.probs, torch.full_like(out.probs, prior_mean), atol=1e-4)


def test_whitebox_branch_shrinks_toward_prior_with_sparse_evidence():
    """Single-prerequisite, single-observation sanity check computed by
    hand. Prereq graph: 0 -> 1. t=0: concept 0 (no prereqs of its own,
    falls back to prior). t=1: concept 1 -- concept 0 has been observed
    once, correct=1. With a single prerequisite, the confidence-weighted
    average over prerequisites reduces to that prerequisite's own
    Beta-Binomial posterior mastery, which must land strictly between the
    prior and the raw empirical rate (1.0) -- not jump straight to 1.0.
    """
    prereq_edge_index = torch.tensor([[0], [1]], dtype=torch.long)  # 0 -> 1
    concept_ids = torch.tensor([[0, 1]])
    responses = torch.tensor([[1.0, 0.0]])
    prior_mean, prior_strength, kappa_w = 0.5, 4.0, 4.0

    white_prob, white_confidence = whitebox_branch(
        concept_ids=concept_ids,
        responses=responses,
        prereq_edge_index=prereq_edge_index,
        n_concepts=4,
        prior_mean=prior_mean,
        prior_strength=prior_strength,
        kappa_w=kappa_w,
    )

    assert white_prob[0, 0].item() == pytest.approx(prior_mean)  # concept 0 has no prereqs
    assert white_confidence[0, 0].item() == pytest.approx(0.0)

    expected_m_p = (prior_mean * prior_strength + 1.0) / (prior_strength + 1.0)  # (2+1)/(4+1)=0.6
    assert white_prob[0, 1].item() == pytest.approx(expected_m_p)
    assert 0.0 < white_prob[0, 1].item() < 1.0  # strictly shrunk, not raw 1.0

    expected_confidence_t1 = 1.0 / (1.0 + kappa_w)  # pooled N_eff=1 -> 0.2
    assert white_confidence[0, 1].item() == pytest.approx(expected_confidence_t1)
    assert white_confidence[0, 1].item() < 1.0


def test_whitebox_multihop_reaches_beyond_direct_prerequisites():
    """Prereq chain 0 -> 1 -> 2. With max_hops=1, observing only concept 0
    must NOT affect concept 2's confidence (its only 1-hop ancestor, 1,
    was never observed). With max_hops=2, concept 0 becomes reachable and
    contributes."""
    prereq_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # 0->1, 1->2
    concept_ids = torch.tensor([[0, 2]])
    responses = torch.tensor([[1.0, 0.0]])

    _, white_confidence_1hop = whitebox_branch(
        concept_ids=concept_ids,
        responses=responses,
        prereq_edge_index=prereq_edge_index,
        n_concepts=4,
        max_hops=1,
    )
    assert white_confidence_1hop[0, 1].item() == pytest.approx(0.0)

    _, white_confidence_2hop = whitebox_branch(
        concept_ids=concept_ids,
        responses=responses,
        prereq_edge_index=prereq_edge_index,
        n_concepts=4,
        max_hops=2,
        hop_decay=0.5,
    )
    assert white_confidence_2hop[0, 1].item() > 0.0


def test_black_confidence_table_matches_formula():
    concept_train_freq = torch.tensor([100.0, 1.0, 0.0, 50.0])
    kappa_b = 4.0
    table = black_confidence_table(concept_train_freq, n_concepts=4, kappa_b=kappa_b, device=torch.device("cpu"))
    assert table[0].item() == pytest.approx(100.0 / 104.0)
    assert table[2].item() == pytest.approx(0.0)
    assert table[3].item() == pytest.approx(50.0 / 54.0)


def test_black_confidence_table_defaults_to_zero_when_train_freq_missing():
    table = black_confidence_table(None, n_concepts=4, kappa_b=4.0, device=torch.device("cpu"))
    assert torch.allclose(table, torch.zeros(4))


def test_reliability_gate_is_relative_not_a_floor():
    """The whole point of switching from min() to a ratio: when both
    confidences are equal, the gate must be exactly 0.5 (a min()-based
    gate would just return that shared value, not 0.5)."""
    cb = torch.tensor([0.4])
    cw = torch.tensor([0.4])
    gate = reliability_gate(cb, cw, eps=1e-6)
    assert gate[0].item() == pytest.approx(0.5, abs=1e-5)

    # asymmetric cases land on the correct side.
    gate_black_favored = reliability_gate(torch.tensor([0.9]), torch.tensor([0.05]), eps=1e-6)
    assert gate_black_favored[0].item() > 0.9
    gate_white_favored = reliability_gate(torch.tensor([0.05]), torch.tensor([0.9]), eps=1e-6)
    assert gate_white_favored[0].item() < 0.1

    # both-zero case must not NaN.
    gate_both_zero = reliability_gate(torch.tensor([0.0]), torch.tensor([0.0]), eps=1e-6)
    assert torch.isfinite(gate_both_zero).all()


def test_greykt_black_box_loads_pretrained_dh2kt_state_dict():
    """A checkpoint trained as a plain DH2KT must load into GreyKT.black_box
    without shape mismatches -- this is the whole point of composing
    rather than reimplementing the black-box branch."""
    from dh2a_kt.models.dh2_kt import DH2KT

    dh2kt_config = GreyKTConfig(n_concepts=8, n_exercises=6, hidden_dim=16, embed_dim=16).to_dh2kt_config()
    standalone = DH2KT(dh2kt_config)

    grey_config = GreyKTConfig(n_concepts=8, n_exercises=6, hidden_dim=16, embed_dim=16)
    grey = GreyKT(grey_config)
    grey.black_box.load_state_dict(standalone.state_dict())  # must not raise


def test_greykt_sanity_overfit_tiny_batch():
    """Gradients should flow through the black-box branch and let the
    fused loss decrease. Uses BCELoss on the fused probability directly
    (v3's preferred training path -- see module docstring, fix 2) rather
    than BCEWithLogitsLoss on the derived logits."""
    torch.manual_seed(0)
    config = GreyKTConfig(
        n_concepts=4,
        n_exercises=4,
        hidden_dim=32,
        embed_dim=32,
        n_hypergraph_layers=1,
        dropout=0.0,
    )
    model = GreyKT(config)
    optimizer = torch.optim.Adam(model.black_box.parameters(), lr=1e-2)

    concept_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]] * 4)
    exercise_ids = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 0]] * 4)
    labels = (concept_ids % 2).float()
    responses = torch.zeros_like(labels)
    responses[:, 1:] = labels[:, :-1]

    batch = GreyKTBatch(
        concept_ids=concept_ids,
        exercise_ids=exercise_ids,
        responses=responses,
        hyperedge_index={
            "concept_prerequisite": torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]], dtype=torch.long),
        },
        prereq_edge_index=None,  # isolate the black-box branch for this check
        concept_train_freq=None,  # -> C_B=0 everywhere -> gate=0 -> fused == white == prior;
        # this test only exercises gradient flow through black_logits/black_probs,
        # so we read black_probs directly rather than the (gate-degenerate) fused probs.
    )

    loss_fn = torch.nn.BCELoss()
    targets = labels[:, 1:]
    for _ in range(400):
        optimizer.zero_grad()
        black_probs = model(batch).black_probs[:, :-1]
        loss = loss_fn(black_probs, targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = loss_fn(model(batch).black_probs[:, :-1], targets).item()
    assert final_loss < 0.05


def test_expected_calibration_error_perfect_calibration_is_zero():
    probs = torch.tensor([0.1, 0.1, 0.9, 0.9])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.1, abs=1e-6)


def test_negative_log_likelihood_and_brier_score_basic_properties():
    probs = torch.tensor([0.9, 0.9, 0.1, 0.1])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])  # perfectly confident and correct
    nll_good = negative_log_likelihood(probs, labels)
    brier_good = brier_score(probs, labels)

    labels_wrong = torch.tensor([0.0, 0.0, 1.0, 1.0])  # confidently wrong
    nll_bad = negative_log_likelihood(probs, labels_wrong)
    brier_bad = brier_score(probs, labels_wrong)

    assert nll_good < nll_bad
    assert brier_good < brier_bad
    assert brier_good == pytest.approx(0.01, abs=1e-6)


def test_v4a_temperature_preserves_ranking_and_is_identity_at_one():
    """Temperature scaling is monotone, so it must leave the black-box
    branch's ordering (and therefore its AUC) exactly unchanged -- that is
    what makes it safe to apply post-hoc without re-validating
    discrimination. T=1.0 must be a no-op."""
    config_t1 = GreyKTConfig(n_concepts=8, n_exercises=6, hidden_dim=16, embed_dim=16)
    config_t3 = GreyKTConfig(
        n_concepts=8, n_exercises=6, hidden_dim=16, embed_dim=16, temperature=3.0
    )
    torch.manual_seed(0)
    model = GreyKT(config_t1)
    model.eval()  # dropout would reshuffle logits across the two forwards
    batch = _make_toy_batch(with_train_freq=True)

    out_t1 = model(batch)
    model.config = config_t3
    out_t3 = model(batch)

    assert torch.allclose(out_t1.black_probs, torch.sigmoid(out_t1.black_logits))

    order_t1 = torch.argsort(out_t1.black_probs.reshape(-1))
    order_t3 = torch.argsort(out_t3.black_probs.reshape(-1))
    assert torch.equal(order_t1, order_t3)

    # T > 1 must shrink probabilities toward 0.5 (the overconfidence fix).
    assert ((out_t3.black_probs - 0.5).abs() <= (out_t1.black_probs - 0.5).abs() + 1e-6).all()


def test_v4b_backoff_separates_both_strong_from_both_weak():
    """The case v3 structurally cannot express. With C_B == C_W the gate is
    0.5 whether both branches are highly reliable or both are ignorant;
    the absolute-reliability weight must tell them apart."""
    cb_strong, cw_strong = torch.tensor([0.9]), torch.tensor([0.9])
    cb_weak, cw_weak = torch.tensor([0.05]), torch.tensor([0.05])

    gate_strong = reliability_gate(cb_strong, cw_strong, eps=1e-6)
    gate_weak = reliability_gate(cb_weak, cw_weak, eps=1e-6)
    assert gate_strong.item() == pytest.approx(0.5, abs=1e-4)
    assert gate_weak.item() == pytest.approx(0.5, abs=1e-4)

    kappa_r = 1.0
    alpha_strong = (cb_strong + cw_strong) / (cb_strong + cw_strong + kappa_r)
    alpha_weak = (cb_weak + cw_weak) / (cb_weak + cw_weak + kappa_r)
    assert alpha_strong.item() > 0.6
    assert alpha_weak.item() < 0.15


def test_v4b_disabled_by_default_reproduces_v3_exactly():
    """The ablation ladder depends on v3 being bit-identical when the v4
    flags are off."""
    torch.manual_seed(0)
    config = GreyKTConfig(n_concepts=8, n_exercises=6, hidden_dim=16, embed_dim=16)
    assert config.use_absolute_backoff is False
    assert config.temperature == 1.0

    model = GreyKT(config)
    out = model(_make_toy_batch(with_train_freq=True))
    assert torch.allclose(out.probs, out.branch_mixture)
    assert torch.allclose(out.absolute_reliability, torch.ones_like(out.absolute_reliability))


def test_v4b_enabled_pulls_unevidenced_predictions_toward_base_rate():
    torch.manual_seed(0)
    base_rate = 0.62
    config = GreyKTConfig(
        n_concepts=8,
        n_exercises=6,
        hidden_dim=16,
        embed_dim=16,
        prior_mean=base_rate,
        use_absolute_backoff=True,
        kappa_r=1.0,
    )
    model = GreyKT(config)
    # No prerequisite edges and no training frequency -> C_B = C_W = 0
    # everywhere, so alpha = 0 and every prediction must collapse to the
    # base rate regardless of what the black-box branch says.
    batch = _make_toy_batch(with_prereqs=False, with_train_freq=False)
    out = model(batch)

    assert torch.allclose(out.absolute_reliability, torch.zeros_like(out.absolute_reliability))
    assert torch.allclose(out.probs, torch.full_like(out.probs, base_rate), atol=1e-5)


def test_fit_temperature_recovers_a_known_scaling():
    """Generate labels from deliberately over-sharp logits; the fitted
    temperature must be > 1 (i.e. it detects and corrects overconfidence)."""
    torch.manual_seed(0)
    true_logits = torch.randn(4000) * 1.5
    # Labels drawn from the WELL-calibrated probabilities...
    probs = torch.sigmoid(true_logits)
    labels = torch.bernoulli(probs)
    # ...but the model reports logits scaled up by 2.5x (overconfident).
    overconfident_logits = true_logits * 2.5

    fitted = fit_temperature(overconfident_logits, labels, max_iter=400, lr=0.05)
    assert fitted > 1.5, f"expected T > 1.5 to undo 2.5x overconfidence, got {fitted}"

    # And the correction must improve calibration without touching ranking.
    ece_before = expected_calibration_error(torch.sigmoid(overconfident_logits), labels)
    ece_after = expected_calibration_error(torch.sigmoid(overconfident_logits / fitted), labels)
    assert ece_after < ece_before


def test_stratified_metrics_2d_returns_grid_with_expected_cell_count():
    torch.manual_seed(0)
    n = 200
    fused_probs = torch.rand(n)
    black_probs = torch.rand(n)
    black_confidence = torch.rand(n)
    white_confidence = torch.rand(n)
    labels = torch.randint(0, 2, (n,)).float()

    n_buckets = 3
    results = stratified_metrics_2d(
        fused_probs, black_probs, black_confidence, white_confidence, labels, n_buckets=n_buckets
    )
    assert len(results) == n_buckets * n_buckets
    for cell_stats in results.values():
        for key in ("n", "fused_auc", "black_auc", "fused_nll", "black_nll", "fused_brier", "black_brier"):
            assert key in cell_stats
