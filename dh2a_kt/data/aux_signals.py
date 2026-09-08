"""Sidecar columns that P0's 5-column schema drops or never maps.

Cheap experiments (giai đoạn M): multi-skill Q-matrix, log time-gap,
previous-step ``saw_answer``, ASSIST2012 teacher grouping, Junyi expert DAG.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def _stable_int64(value: object) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)


def load_item_kc_map(
    path: Path, *, item_col: str = "problem_id", kc_col: str = "skill_id"
) -> dict[int, list[int]]:
    """problem_id → all skill_ids (FoundationalASSIST Skills.csv)."""
    df = pd.read_csv(path)
    missing = {item_col, kc_col} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    out: dict[int, list[int]] = {}
    for item, group in df.groupby(item_col, sort=False):
        out[int(item)] = [int(k) for k in group[kc_col].tolist()]
    return out


def attach_saw_answer(canonical: pd.DataFrame, extended: pd.DataFrame) -> pd.DataFrame:
    """Left-join ``saw_answer`` onto a 5-column split frame."""
    keys = ["user_id", "item_id", "timestamp"]
    extra_cols = [c for c in ("saw_answer", "hint_used", "hint_count") if c in extended.columns]
    slim = extended[keys + extra_cols].drop_duplicates(keys)
    out = canonical.merge(slim, on=keys, how="left")
    if "saw_answer" in out.columns:
        out["saw_answer"] = out["saw_answer"].fillna(0).astype("int64")
    return out


def log_time_gaps(timestamps: np.ndarray, user_ids: np.ndarray) -> np.ndarray:
    """log1p seconds since the previous row of the same user; 0 at user starts."""
    n = len(timestamps)
    dt = np.zeros(n, dtype=np.float32)
    if n < 2:
        return dt
    raw = timestamps[1:].astype(np.int64) - timestamps[:-1].astype(np.int64)
    same = user_ids[1:] == user_ids[:-1]
    dt[1:] = np.log1p(np.maximum(raw * same.astype(np.int64), 0).astype(np.float32))
    return dt


def apply_time_gap_control(
    gaps: np.ndarray,
    user_ids: np.ndarray,
    control: str,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Keep the time *branch*; change only the gap *input*.

    ``real`` leaves aligned ``log(1+Δt)``. ``zero`` sets every gap to 0
    (raw feature zero; ``f(0)=b`` still injects bias). ``misaligned``
    permutes non-start gaps within each user so the marginal gap
    distribution is unchanged but alignment is broken.
    """
    if control not in ("real", "zero", "misaligned"):
        raise ValueError(
            f"time_gap_control must be real/zero/misaligned, got {control!r}"
        )
    out = np.asarray(gaps, dtype=np.float32)
    if control == "real":
        return out.copy()
    if control == "zero":
        return np.zeros_like(out)
    users = np.asarray(user_ids)
    rng = np.random.default_rng(int(seed))
    shuffled = out.copy()
    for uid in np.unique(users):
        idx = np.flatnonzero(users == uid)
        if idx.size < 3:
            continue
        body = idx[1:]
        shuffled[body] = rng.permutation(out[body])
    return shuffled


def infer_timestamp_seconds_scale(timestamps: np.ndarray) -> float:
    """ASSIST2012 P0 stores unix-seconds//1000 (~1e6); XES stores seconds (~1e9)."""
    if len(timestamps) == 0:
        return 1.0
    median = float(np.median(np.abs(timestamps.astype(np.float64))))
    return 1000.0 if median < 1.0e8 else 1.0


def log_duration_idle(
    timestamps: np.ndarray,
    user_ids: np.ndarray,
    duration_sec: np.ndarray,
    *,
    timestamp_unit_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """log1p duration at t and log1p idle before t (same-user consecutive rows)."""
    dur = np.maximum(duration_sec.astype(np.float32), 0.0)
    dur_log = np.log1p(dur)
    n = len(timestamps)
    idle = np.zeros(n, dtype=np.float32)
    if n < 2:
        return dur_log, idle
    scale = (
        infer_timestamp_seconds_scale(timestamps)
        if timestamp_unit_seconds is None
        else float(timestamp_unit_seconds)
    )
    same = user_ids[1:] == user_ids[:-1]
    gap_sec = (
        timestamps[1:].astype(np.float64) - timestamps[:-1].astype(np.float64)
    ) * scale
    idle[1:] = np.maximum(gap_sec - dur[:-1].astype(np.float64), 0.0).astype(
        np.float32
    ) * same.astype(np.float32)
    return dur_log, np.log1p(idle)


def attach_assist_timing(processed: pd.DataFrame, raw_csv: Path) -> pd.DataFrame:
    """Join ASSIST2012 ``ms_first_response`` as duration_sec via hashed user + ts.

    P0 timestamps are unix-seconds//1000. Duration is milliseconds of first
    response / 1000. Idle is computed later from consecutive processed rows.
    """
    raw = pd.read_csv(raw_csv, usecols=["user_id", "start_time", "ms_first_response"])
    raw["user_id"] = raw["user_id"].map(_stable_int64).astype("int64")
    ts = pd.to_datetime(raw["start_time"], utc=True, errors="coerce")
    ns = ts.astype("int64").to_numpy()
    unix_s = np.where(ts.notna().to_numpy(), ns // 1_000_000_000, 0)
    raw["timestamp"] = (unix_s // 1000).astype("int64")
    raw["duration_sec"] = (
        pd.to_numeric(raw["ms_first_response"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        / 1000.0
    )
    slim = (
        raw.groupby(["user_id", "timestamp"], sort=False)["duration_sec"]
        .median()
        .reset_index()
    )
    out = processed.merge(slim, on=["user_id", "timestamp"], how="left")
    out["duration_sec"] = out["duration_sec"].fillna(0.0)
    return out


def attach_assist_teacher(processed: pd.DataFrame, raw_csv: Path) -> pd.DataFrame:
    """Attach ASSIST2012 ``teacher_id`` by hashed ``user_id`` (P0 blake2).

    Teacher is a property of the learner, not of ``(item, timestamp)``. Row-level
    joins fail because P0 stores timestamps as unix-seconds//1000 and item hashes
    do not always round-trip. Map each user to their modal teacher instead.
    """
    raw = pd.read_csv(raw_csv, usecols=["user_id", "teacher_id"])
    raw["user_id"] = raw["user_id"].map(_stable_int64).astype("int64")

    def _hash_or_zero(value: object) -> int:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0
        try:
            if pd.isna(value):
                return 0
        except (TypeError, ValueError):
            pass
        if isinstance(value, (np.floating, float)) and float(value).is_integer():
            value = int(value)
        return _stable_int64(value)

    raw["teacher_id"] = raw["teacher_id"].map(_hash_or_zero).astype("int64")
    user_teacher = (
        raw.loc[raw["teacher_id"] != 0]
        .drop_duplicates("user_id", keep="first")
        .set_index("user_id")["teacher_id"]
    )
    out = processed.copy()
    out["teacher_id"] = processed["user_id"].map(user_teacher)
    return out


def dense_group_ids(values: pd.Series) -> tuple[np.ndarray, dict[int, int], int]:
    """Map hashed group ids to ``1..n``; missing/unknown → 0. Returns codes, map, n_groups."""
    codes = np.zeros(len(values), dtype=np.int64)
    mapping: dict[int, int] = {}
    next_id = 1
    for i, raw in enumerate(values.tolist()):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        try:
            if pd.isna(raw):
                continue
        except (TypeError, ValueError):
            pass
        key = int(raw)
        if key == 0:
            continue
        if key not in mapping:
            mapping[key] = next_id
            next_id += 1
        codes[i] = mapping[key]
    return codes, mapping, next_id


def load_junyi_expert_edges(path: Path, *, threshold: float | None = None) -> pd.DataFrame:
    """Directed expert prerequisite pairs, hashed like P0 ``exercise`` ids.

    Chang et al. scores are ~1–9; YAML ``prereq_threshold: 0.5`` keeps every
    annotated pair. Pass a higher cut to sparsify.
    """
    df = pd.read_csv(path)
    if "Prerequisite_avg" in df.columns and threshold is not None:
        df = df[df["Prerequisite_avg"] >= float(threshold)].copy()
    src = df["Exercise_A"].map(_stable_int64).astype("int64")
    dst = df["Exercise_B"].map(_stable_int64).astype("int64")
    out = pd.DataFrame({"src_kc": src, "dst_kc": dst, "weight": 1.0})
    return out.drop_duplicates(["src_kc", "dst_kc"]).reset_index(drop=True)
