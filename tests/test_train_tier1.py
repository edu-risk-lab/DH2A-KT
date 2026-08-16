"""M5 gate: Tier 1 train/eval loop and P0 comparison table."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from dh2a_kt.hyperedge.p0_inputs import REPO_ROOT, load_configs, load_e_pre, load_interactions_with_ids, get_fold_splits
from dh2a_kt.train.tier1 import (
    TrainingBudget,
    next_step_loss,
    resolve_training_budget,
    train_and_evaluate_fold,
    write_comparison_table,
)

XES3G5M_CONFIG = REPO_ROOT / "configs" / "xes3g5m.yaml"
P0_PARQUET = REPO_ROOT / "external" / "p0_leakage_audit" / "data" / "processed" / "xes3g5m.parquet"


def _synthetic_logs(n_users: int = 16, seq_len: int = 12) -> pd.DataFrame:
    rows = []
    i = 0
    for u in range(n_users):
        for t in range(seq_len):
            rows.append(
                {
                    "user_id": u,
                    "item_id": (u + t) % 5,
                    "kc_id": t % 4,
                    "timestamp": u * 1000 + t,
                    "correct": (u + t) % 2,
                }
            )
            i += 1
    return pd.DataFrame(rows)


def test_resolve_training_budget_matches_p0_gkt():
    _dh2, p0_cfg, _ = load_configs(XES3G5M_CONFIG)
    budget = resolve_training_budget(p0_cfg, reference_model="gkt")
    assert budget.batch_size == 4
    assert budget.epochs == 10
    assert budget.matched_p0 is True


def test_next_step_loss_ignores_short_sequences():
    torch = pytest.importorskip("torch")
    logits = torch.randn(2, 4, 1)
    targets = torch.randint(0, 2, (2, 4)).float()
    lengths = torch.tensor([4, 1])
    loss = next_step_loss(logits, targets, lengths)
    assert torch.isfinite(loss)


def test_train_and_evaluate_fold_toy():
    pytest.importorskip("torch_geometric")
    torch = pytest.importorskip("torch")
    train_df = _synthetic_logs(n_users=12, seq_len=10)
    eval_df = _synthetic_logs(n_users=8, seq_len=10)
    e_pre = pd.DataFrame({"src_kc": [0, 1], "dst_kc": [1, 2], "weight": [1.0, 1.0]})
    budget = TrainingBudget(
        reference_model="gkt",
        batch_size=4,
        epochs=3,
        lr=1e-2,
        max_seq_len=10,
        matched_p0=True,
    )
    result = train_and_evaluate_fold(
        train_df,
        eval_df,
        e_pre,
        budget,
        device="cpu",
        hidden_dim=16,
    )
    assert np.isfinite(result.auc)
    assert result.n_predictions > 0


def test_train_and_evaluate_fold_toy_v3():
    pytest.importorskip("torch_geometric")
    train_df = _synthetic_logs(n_users=12, seq_len=10)
    eval_df = _synthetic_logs(n_users=8, seq_len=10)
    e_pre = pd.DataFrame({"src_kc": [0, 1], "dst_kc": [1, 2], "weight": [1.0, 1.0]})
    budget = TrainingBudget(
        reference_model="gkt",
        batch_size=4,
        epochs=2,
        lr=1e-2,
        max_seq_len=10,
        matched_p0=True,
    )
    result = train_and_evaluate_fold(
        train_df,
        eval_df,
        e_pre,
        budget,
        device="cpu",
        hidden_dim=16,
        architecture="v3",
    )
    assert np.isfinite(result.auc)
    assert result.n_predictions > 0


def test_chunked_windows_cover_the_whole_log():
    from dh2a_kt.train.tier1 import UserSequenceDataset

    logs = _synthetic_logs(n_users=5, seq_len=25)
    kc_to_idx = {kc: i for i, kc in enumerate(sorted(logs["kc_id"].unique()))}
    item_to_idx = {it: i for i, it in enumerate(sorted(logs["item_id"].unique()))}
    kwargs = dict(max_seq_len=10)
    first = UserSequenceDataset(logs, kc_to_idx, item_to_idx, window_mode="first", **kwargs)
    chunked = UserSequenceDataset(logs, kc_to_idx, item_to_idx, window_mode="chunked", **kwargs)

    def covered(dataset) -> int:
        # __getitem__ truncates each window at max_seq_len.
        return sum(
            min(e - s, dataset.max_seq_len)
            for s, e in zip(dataset._starts, dataset._ends, strict=True)
        )

    assert len(first) == 5
    assert covered(first) == 5 * 10  # only the first window per user

    assert covered(chunked) == len(logs)
    assert all(e - s <= 10 for s, e in zip(chunked._starts, chunked._ends, strict=True))


def test_last_window_matches_p0s_tail_slice():
    """P0's pyKT export keeps ``q_list[-max_seq_len:]``, so 'first' scores other rows.

    Both modes yield one window per learner and therefore the same position count,
    which is why the mismatch stayed invisible in the comparison tables.
    """
    from dh2a_kt.train.tier1 import UserSequenceDataset

    logs = _synthetic_logs(n_users=5, seq_len=25)
    kc_to_idx = {kc: i for i, kc in enumerate(sorted(logs["kc_id"].unique()))}
    item_to_idx = {it: i for i, it in enumerate(sorted(logs["item_id"].unique()))}
    kwargs = dict(max_seq_len=10)
    first = UserSequenceDataset(logs, kc_to_idx, item_to_idx, window_mode="first", **kwargs)
    last = UserSequenceDataset(logs, kc_to_idx, item_to_idx, window_mode="last", **kwargs)

    assert len(last) == len(first) == 5
    for (fs, fe), (ls, le) in zip(
        zip(first._starts, first._ends, strict=True),
        zip(last._starts, last._ends, strict=True),
        strict=True,
    ):
        assert le - ls == 10, "the tail window is full whenever the learner is long enough"
        assert le == fe, "both modes end at the learner boundary in _ends"
        assert ls == fe - 10, "'last' starts max_seq_len before the learner's final row"
        assert ls != fs, "so the two modes score disjoint halves of a 25-row log"

    short = _synthetic_logs(n_users=1, seq_len=4)
    short_ds = UserSequenceDataset(
        short,
        {kc: i for i, kc in enumerate(sorted(short["kc_id"].unique()))},
        {it: i for i, it in enumerate(sorted(short["item_id"].unique()))},
        window_mode="last",
        max_seq_len=10,
    )
    assert short_ds._starts == [0], "a learner shorter than the window is not clipped"


def _logs_with_multi_kc_questions() -> pd.DataFrame:
    """One learner whose 2nd question covers 3 concepts, as pyKT's KC-level export writes it."""
    rows = [
        {"user_id": 0, "item_id": 10, "kc_id": 0, "timestamp": 100, "correct": 1},
        {"user_id": 0, "item_id": 11, "kc_id": 1, "timestamp": 200, "correct": 0},
        {"user_id": 0, "item_id": 11, "kc_id": 2, "timestamp": 200, "correct": 0},
        {"user_id": 0, "item_id": 11, "kc_id": 3, "timestamp": 200, "correct": 0},
        {"user_id": 0, "item_id": 12, "kc_id": 0, "timestamp": 300, "correct": 1},
        # A different learner reusing the same item must not be flagged.
        {"user_id": 1, "item_id": 11, "kc_id": 1, "timestamp": 200, "correct": 1},
        {"user_id": 1, "item_id": 12, "kc_id": 2, "timestamp": 400, "correct": 0},
    ]
    return pd.DataFrame(rows)


def test_repeat_flags_mark_only_continuations_of_the_same_attempt():
    from dh2a_kt.train.tier1 import _repeat_flags

    logs = _logs_with_multi_kc_questions().sort_values(
        ["user_id", "timestamp", "item_id"]
    ).reset_index(drop=True)
    flags = _repeat_flags(logs)
    # rows 2 and 3 continue row 1's attempt; the learner boundary resets the flag.
    assert flags.tolist() == [False, False, True, True, False, False, False]


def test_dataset_exposes_repeat_flags_per_window():
    torch = pytest.importorskip("torch")
    from dh2a_kt.train.tier1 import UserSequenceDataset

    logs = _logs_with_multi_kc_questions()
    kc_to_idx = {kc: i for i, kc in enumerate(sorted(logs["kc_id"].unique()))}
    item_to_idx = {it: i for i, it in enumerate(sorted(logs["item_id"].unique()))}
    dataset = UserSequenceDataset(
        logs, kc_to_idx, item_to_idx, max_seq_len=8, window_mode="chunked"
    )
    first = dataset[0]
    assert first["is_repeat"].dtype == torch.bool
    length = int(first["length"])
    assert first["is_repeat"][:length].tolist() == [False, False, True, True, False]
    # padding must never be scored as a repeat
    assert not first["is_repeat"][length:].any()


def test_next_step_mask_drops_repeat_targets_only_when_asked():
    torch = pytest.importorskip("torch")
    from dh2a_kt.train.tier1 import next_step_mask

    lengths = torch.tensor([5])
    is_repeat = torch.tensor([[False, False, True, True, False]])
    default = next_step_mask(4, lengths)
    clean = next_step_mask(4, lengths, is_repeat=is_repeat)
    # positions t=0..3 predict rows 1..4; rows 2 and 3 are repeats.
    assert default.tolist() == [[True, True, True, True]]
    assert clean.tolist() == [[True, False, False, True]]


def test_next_step_loss_ignores_repeat_targets_under_clean_protocol():
    torch = pytest.importorskip("torch")

    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0]])
    lengths = torch.tensor([5])
    is_repeat = torch.tensor([[False, False, True, True, False]])
    logits = torch.zeros(1, 5, 1)
    clean_before = next_step_loss(logits, targets, lengths, is_repeat=is_repeat)

    # Perturbing only the masked-out positions must not move the clean loss,
    # while the default protocol does react to them.
    perturbed = logits.clone()
    perturbed[0, 1, 0] = 9.0
    perturbed[0, 2, 0] = -9.0
    clean_after = next_step_loss(perturbed, targets, lengths, is_repeat=is_repeat)
    default_after = next_step_loss(perturbed, targets, lengths)

    assert torch.isclose(clean_before, clean_after)
    assert not torch.isclose(next_step_loss(logits, targets, lengths), default_after)


def test_clean_protocol_scores_fewer_positions_than_p0_protocol():
    pytest.importorskip("torch_geometric")
    from dh2a_kt.train.tier1 import _collate, UserSequenceDataset, build_model, evaluate_auc
    from dh2a_kt.hyperedge.indexing import hyperedge_index_from_list

    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    logs = pd.concat(
        [_logs_with_multi_kc_questions().assign(user_id=lambda d: d["user_id"] + 2 * k) for k in range(6)],
        ignore_index=True,
    )
    kc_to_idx = {kc: i for i, kc in enumerate(sorted(logs["kc_id"].unique()))}
    item_to_idx = {it: i for i, it in enumerate(sorted(logs["item_id"].unique()))}
    loader = DataLoader(
        UserSequenceDataset(logs, kc_to_idx, item_to_idx, max_seq_len=8, window_mode="chunked"),
        batch_size=4,
        shuffle=False,
        collate_fn=_collate,
    )
    model = build_model(len(kc_to_idx), len(item_to_idx), hidden_dim=8, architecture="v4")
    index = hyperedge_index_from_list([], kc_to_idx)
    _p0_auc, p0_n = evaluate_auc(model, loader, index, torch.device("cpu"))
    _clean_auc, clean_n = evaluate_auc(
        model, loader, index, torch.device("cpu"), mask_repeats=True
    )
    # each 3-concept question contributes exactly 2 repeat targets per learner
    assert p0_n - clean_n == 2 * 6


def test_responses_for_model_alignment_per_architecture():
    torch = pytest.importorskip("torch")
    from dh2a_kt.train.tier1 import build_model, responses_for_model

    batch = {
        "correct": torch.tensor([[1.0, 0.0, 1.0]]),
        "responses": torch.tensor([[0.0, 1.0, 0.0]]),
    }
    for architecture, expected in (("v2", "responses"), ("v3", "responses"), ("v4", "correct")):
        model = build_model(4, 4, hidden_dim=8, architecture=architecture)
        assert torch.equal(responses_for_model(model, batch), batch[expected])


@pytest.mark.parametrize("use_questions", [False, True])
def test_train_and_evaluate_fold_toy_v4(use_questions: bool):
    pytest.importorskip("torch_geometric")
    train_df = _synthetic_logs(n_users=12, seq_len=20)
    eval_df = _synthetic_logs(n_users=8, seq_len=20)
    e_pre = pd.DataFrame({"src_kc": [0, 1], "dst_kc": [1, 2], "weight": [1.0, 1.0]})
    budget = TrainingBudget(
        reference_model="gkt",
        batch_size=4,
        epochs=2,
        lr=1e-2,
        max_seq_len=10,
        matched_p0=False,
    )
    result = train_and_evaluate_fold(
        train_df,
        eval_df,
        e_pre,
        budget,
        device="cpu",
        hidden_dim=16,
        architecture="v4",
        use_questions=use_questions,
        window_mode="chunked",
        val_frac=0.25,
        early_stop_patience=2,
    )
    assert np.isfinite(result.auc)
    # chunked windows must score more positions than one window per user would.
    assert result.n_predictions > 8 * 9


def test_write_comparison_table(tmp_path: Path):
    budget = TrainingBudget(
        reference_model="gkt",
        batch_size=4,
        epochs=10,
        lr=0.001,
        max_seq_len=200,
        matched_p0=True,
    )
    out = tmp_path / "dh2_kt_vs_p0.csv"
    df = write_comparison_table(
        [type("R", (), {"fold": 0, "auc": 0.85, "n_predictions": 1000})()],
        dataset="xes3g5m",
        budget=budget,
        comparison_models=["gkt"],
        output_path=out,
    )
    assert out.exists()
    assert "p0_auc" in df.columns
    assert "comparison_type" in df.columns
    assert df["comparison_type"].iloc[0] == "matched"


def _real_data_available() -> bool:
    if not XES3G5M_CONFIG.exists() or not P0_PARQUET.exists():
        return False
    from dh2a_kt.hyperedge.p0_inputs import e_pre_export_path

    _dh2, p0_cfg, _ = load_configs(XES3G5M_CONFIG)
    return e_pre_export_path(p0_cfg, 0).exists()


@pytest.mark.slow
@pytest.mark.skipif(not _real_data_available(), reason="P0 XES3G5M processed data not present")
def test_train_tier1_smoke_xes3g5m_fold0(tmp_path: Path):
    _dh2, p0_cfg, _ = load_configs(XES3G5M_CONFIG)
    interactions = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(interactions, p0_cfg, 0)
    eval_df = pd.concat([splits["valid"], splits["test"]], ignore_index=True)
    e_pre = load_e_pre(p0_cfg, 0)
    budget = resolve_training_budget(p0_cfg, reference_model="gkt", max_seq_len=50)
    smoke_budget = replace(budget, epochs=1)

    result = train_and_evaluate_fold(
        splits["train"],
        eval_df,
        e_pre,
        budget=smoke_budget,
        device="cpu",
        hidden_dim=32,
        max_users=32,
    )
    assert np.isfinite(result.auc)

    out = tmp_path / "dh2_kt_vs_p0.csv"
    write_comparison_table(
        [type("R", (), {"fold": 0, "auc": result.auc, "n_predictions": result.n_predictions})()],
        dataset="xes3g5m",
        budget=smoke_budget,
        comparison_models=["gkt"],
        output_path=out,
    )
    assert out.exists()
