"""Sidecar columns for giai đoạn M (Q-matrix, time-gap, saw, teacher, Junyi DAG)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dh2a_kt.data.aux_signals import (
    attach_assist_timing,
    attach_saw_answer,
    dense_group_ids,
    infer_timestamp_seconds_scale,
    load_item_kc_map,
    load_junyi_expert_edges,
    log_duration_idle,
    log_time_gaps,
)
from dh2a_kt.models.causal_integration import build_entity_maps


def test_load_item_kc_map_keeps_all_skills(tmp_path):
    path = tmp_path / "Skills.csv"
    path.write_text("problem_id,skill_id\n1,10\n1,11\n2,10\n", encoding="utf-8")
    mapping = load_item_kc_map(path)
    assert mapping[1] == [10, 11]
    assert mapping[2] == [10]


def test_log_time_gaps_resets_at_user_boundary():
    ts = np.array([0, 10, 100, 5, 15], dtype=np.int64)
    users = np.array([1, 1, 1, 2, 2], dtype=np.int64)
    gaps = log_time_gaps(ts, users)
    assert gaps[0] == 0.0
    assert gaps[1] == pytest.approx(np.log1p(10))
    assert gaps[2] == pytest.approx(np.log1p(90))
    assert gaps[3] == 0.0
    assert gaps[4] == pytest.approx(np.log1p(10))


def test_xes_start_ms_becomes_log1p_seconds_gap():
    """P0 maps XES millisecond start times to unix seconds before log1p."""
    raw_ms = np.array([1_600_000_000_000, 1_600_000_005_000], dtype=np.int64)
    unix_s = raw_ms // 1000
    gaps = log_time_gaps(unix_s, np.array([1, 1], dtype=np.int64))
    assert gaps[0] == 0.0
    assert gaps[1] == pytest.approx(np.log1p(5.0))


def test_shared_start_timestamp_has_zero_gap():
    ts = np.array([100, 100, 110], dtype=np.int64)
    users = np.array([1, 1, 1], dtype=np.int64)
    gaps = log_time_gaps(ts, users)
    assert gaps[1] == 0.0
    assert gaps[2] == pytest.approx(np.log1p(10.0))


def test_time_gap_control_zero_and_misaligned_keep_starts():
    from dh2a_kt.data.aux_signals import apply_time_gap_control

    ts = np.array([0, 10, 100, 5, 15], dtype=np.int64)
    users = np.array([1, 1, 1, 2, 2], dtype=np.int64)
    gaps = log_time_gaps(ts, users)
    zeros = apply_time_gap_control(gaps, users, "zero")
    assert np.all(zeros == 0.0)
    mis = apply_time_gap_control(gaps, users, "misaligned", seed=7)
    assert mis[0] == 0.0 and mis[3] == 0.0
    assert sorted(mis[1:3].tolist()) == sorted(gaps[1:3].tolist())
    other = apply_time_gap_control(gaps, users, "misaligned", seed=8)
    # Same multiset per user; seed 7 vs 8 may or may not differ on n=2.
    assert sorted(other[1:3].tolist()) == sorted(gaps[1:3].tolist())


def test_log_duration_idle_subtracts_previous_duration():
    ts = np.array([0, 10, 20], dtype=np.int64)
    users = np.array([1, 1, 1], dtype=np.int64)
    dur = np.array([2.0, 3.0, 1.0], dtype=np.float32)
    dur_log, idle_log = log_duration_idle(
        ts, users, dur, timestamp_unit_seconds=1.0
    )
    assert dur_log[0] == pytest.approx(np.log1p(2.0))
    assert idle_log[0] == 0.0
    assert idle_log[1] == pytest.approx(np.log1p(8.0))
    assert idle_log[2] == pytest.approx(np.log1p(7.0))


def test_infer_timestamp_seconds_scale_assist_vs_xes():
    assist = np.array([1_346_779, 1_346_780], dtype=np.int64)
    xes = np.array([1_348_845_087, 1_348_845_200], dtype=np.int64)
    assert infer_timestamp_seconds_scale(assist) == 1000.0
    assert infer_timestamp_seconds_scale(xes) == 1.0


def test_attach_assist_timing_joins_duration(tmp_path):
    from dh2a_kt.data.aux_signals import _stable_int64

    start = "2012-09-28 12:00:00"
    ts = pd.Timestamp(start, tz="UTC")
    unix_s = int(ts.timestamp())
    parquet_ts = unix_s // 1000
    raw = tmp_path / "assist.csv"
    raw.write_text(
        "user_id,start_time,ms_first_response\n"
        f"7,{start},2500\n",
        encoding="utf-8",
    )
    processed = pd.DataFrame(
        {
            "user_id": [_stable_int64(7)],
            "item_id": [1],
            "timestamp": [parquet_ts],
            "kc_id": [1],
            "correct": [1],
        }
    )
    out = attach_assist_timing(processed, raw)
    assert out.loc[0, "duration_sec"] == pytest.approx(2.5)


def test_attach_saw_answer_left_join():
    canonical = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [10, 11],
            "timestamp": [100, 200],
            "kc_id": [3, 3],
            "correct": [1, 0],
        }
    )
    extended = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [10, 11],
            "timestamp": [100, 200],
            "saw_answer": [True, False],
        }
    )
    out = attach_saw_answer(canonical, extended)
    assert out["saw_answer"].tolist() == [1, 0]


def test_dense_group_ids_reserve_zero_for_missing():
    values = pd.Series([10, 10, np.nan, 7, 10])
    codes, mapping, n_groups = dense_group_ids(values)
    assert codes[2] == 0
    assert set(codes[codes > 0].tolist()) == {mapping[10], mapping[7]}
    assert n_groups == 3
    assert 0 not in mapping.values()


def test_build_entity_maps_extra_kc_ids():
    interactions = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [5],
            "kc_id": [2],
            "timestamp": [0],
            "correct": [1],
        }
    )
    e_pre = pd.DataFrame({"src_kc": [2], "dst_kc": [2]})
    kc_to_idx, _ = build_entity_maps(interactions, e_pre, extra_kc_ids={9, 4})
    assert set(kc_to_idx) == {2, 4, 9}


def test_load_junyi_expert_edges_hashes_names(tmp_path):
    path = tmp_path / "dag.csv"
    path.write_text(
        "Exercise_A,Exercise_B,Prerequisite_avg\nfoo,bar,8.0\nfoo,bar,8.0\n",
        encoding="utf-8",
    )
    edges = load_junyi_expert_edges(path)
    assert len(edges) == 1
    assert edges.loc[0, "src_kc"] != edges.loc[0, "dst_kc"]


def test_attach_assist_teacher_joins_on_user(tmp_path):
    from dh2a_kt.data.aux_signals import _stable_int64, attach_assist_teacher

    raw = tmp_path / "assist.csv"
    raw.write_text(
        "user_id,problem_id,start_time,teacher_id,school_id\n"
        "7,1,2012-01-01 00:00:00,99,1\n"
        "7,2,2012-01-01 00:00:01,99,1\n"
        "8,1,2012-01-01 00:00:00,88,1\n",
        encoding="utf-8",
    )
    processed = pd.DataFrame(
        {
            "user_id": [_stable_int64(7), _stable_int64(8)],
            "item_id": [1, 2],
            "timestamp": [0, 1],
            "kc_id": [1, 1],
            "correct": [1, 0],
        }
    )
    out = attach_assist_teacher(processed, raw)
    assert out["teacher_id"].notna().all()
    assert out.loc[0, "teacher_id"] != out.loc[1, "teacher_id"]
