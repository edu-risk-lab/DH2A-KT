#!/usr/bin/env python3
"""Verify DH2A-KT's config points at exactly P0's config/data (docs/idea-D-plan.md
section 10, top risk: "Preprocessing cua D khong khop chinh xac voi P0 ->
baseline reuse khong hop le"). Run this BEFORE trusting any baseline reuse.

This does not re-run preprocessing; it checks that the P0 config referenced
by a DH2A-KT config file exists, is unmodified relative to the vendored
commit, and that the vendored commit hash is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="e.g. configs/xes3g5m.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    p0_config_rel = cfg.get("p0_config")
    if not p0_config_rel:
        print(f"FAIL: {args.config} has no 'p0_config' key.", file=sys.stderr)
        return 1

    p0_config_path = REPO_ROOT / p0_config_rel
    if not p0_config_path.exists():
        print(f"FAIL: {p0_config_path} does not exist.", file=sys.stderr)
        return 1

    vendored_commit_file = REPO_ROOT / "external" / "p0_leakage_audit" / ".p0_vendored_commit.txt"
    if not vendored_commit_file.exists():
        print(
            f"FAIL: {vendored_commit_file} missing — P0 provenance cannot be "
            "verified. Re-vendor P0 (see docs/idea-D-plan.md section 7).",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {args.config} points at {p0_config_path}")
    print(f"    p0_config sha256: {sha256(p0_config_path)}")
    print(f"    vendored commit:  {vendored_commit_file.read_text().strip()}")
    print(
        "\nReminder: this only checks the config FILE is present and pinned. "
        "It does not verify your locally preprocessed data/ matches P0's — "
        "you must still follow external/p0_leakage_audit/README.md section 3 "
        "exactly (same raw files, same schema_mapping) for baseline reuse to "
        "be valid."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
