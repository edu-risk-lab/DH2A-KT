"""Credit-ladder transfer plan after the KBS rejection.

Prints the run list for Expert Systems follow-up experiments. Does not
invent AUC. ``--launch`` starts the next unfinished job only when the
visible CUDA device has at least ``--min-vram-mib`` free (default 20000,
the planned RTX 3090 24GB). A smaller GPU is refused so a laptop run
cannot be written into the paper tables.

Groups, in the order the plan cuts from the end:

  backbone-attn     XES3G5M fold 0, v4 with a causal attention encoder in
                    place of the LSTM, 4 arms x 5 seeds, frozen XES recipe
  assist            ASSISTments 2012 fold 0, 6 arms x 5 seeds, YAML recipe
  assist-ref        same 6 arms x 5 seeds, frozen XES recipe
  recent-baselines  two pyKT names x 3 seeds (context rows only)
  package-capacity  param-padded no-time x 5 seeds, YAML recipe
  package-refbudget param-padded no-time x 5 seeds, reference budget but
                    YAML graph flags (does not pair with the frozen ladder)
  package-matched   param-padded no-time x 5 seeds, frozen XES recipe
  junyi-partial     Junyi fold 0, 3 arms x 5 seeds, --max-users 50000, YAML recipe
  junyi-full        Junyi fold 0, 3 arms x 5 seeds, all learners, frozen XES recipe

"Frozen XES recipe" is REF_RECIPE below, the flags of scripts 34 and 38.

``--time-gap-pad`` allocates Linear(1, hidden) and adds a zero multiple,
so the parameter count matches the timed arm and the representation does
not. The padded weights get no gradient: the control matches parameter
count, not usable capacity. Outcomes are scored by
scripts/51_transfer_credit_tables.py.

The vendored evaluator (external/p0_leakage_audit/src/pykt_engine.py)
only forwards dkt, akt, gkt, simplekt, gikt, sakt, skt, dygkt, dgekt.
sparseKT and stableKT are the requested 2023-2025 context models and
are listed as blocked_engine until a forward exists. They are not
replaced by DyGKT.

Usage:
  python scripts/50_credit_ladder_plan.py
  python scripts/50_credit_ladder_plan.py --group assist-ref --launch-all
  python scripts/50_credit_ladder_plan.py --group package-matched --launch-all
  python scripts/50_credit_ladder_plan.py --group backbone-attn --launch-all
  python scripts/50_credit_ladder_plan.py --group junyi-full --launch-all
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEEDS_5 = (42, 17, 1234, 0, 2024)
SEEDS_3 = (42, 17, 1234)
STATUS = REPO / "results" / "tables" / "credit_ladder_status.json"

COMMON = [
    "--fold",
    "0",
    "--device",
    "cuda",
    "--architecture",
    "v4",
    "--use-questions",
    "--mask-repeats",
    "--window-mode",
    "chunked",
    "--max-seq-len",
    "400",
]

# Recipe of the frozen XES ladder (scripts 34 and 38). Without these flags the
# YAML turns on the concept and session hypergraphs, graph dropout and the
# auxiliary loss, and the arm no longer pairs with qkc_cm_* / a1_*.
REF_RECIPE = [
    "--no-graph",
    "--no-session",
    "--batch-size",
    "16",
    "--epochs",
    "30",
    "--val-frac",
    "0.1",
    "--early-stop-patience",
    "5",
    "--graph-dropout",
    "0",
    "--graph-sensitivity-weight",
    "0",
]


def _dh2(config: str, seed: int, extra: list[str], max_users: int | None = None) -> list[str]:
    cmd = [sys.executable, "scripts/03_train_tier1.py", config, *COMMON, "--seed", str(seed), *extra]
    if max_users is not None:
        cmd.extend(["--max-users", str(max_users)])
    return cmd


def _jobs() -> list[dict]:
    jobs: list[dict] = []

    def add(group: str, arm: str, seed: int, cmd: list[str] | None, block: str | None = None) -> None:
        if cmd is not None:
            # Keep each arm×seed off the shared v4q checkpoint and CSV.
            tag = f"ladder_{group}_{arm}_s{seed}"
            cmd = [*cmd, "--tag", tag]
        jobs.append(
            {
                "group": group,
                "arm": arm,
                "seed": seed,
                "cmd": cmd,
                "block": block,
            }
        )

    # Second backbone: the vendored simpleKT/AKT forwards take no gap tensor and
    # the pyKT clean export has placeholder timestamps, so the swap is done
    # inside v4: a causal Transformer replaces the LSTM, everything else fixed.
    # Reference budget so the arms pair with the frozen XES LSTM ladder.
    attn = ["--sequence-encoder", "attention", *REF_RECIPE]
    zero_q = ["--question-graph", "--question-incidence-control", "zero"]
    timed = [*zero_q, "--time-gap", "--time-gap-mode", "both"]
    backbone_arms = {
        "no-time": zero_q,
        "time-package": timed,
        "t-zero": [*timed, "--time-gap-control", "zero"],
        "t-misaligned": [*timed, "--time-gap-control", "misaligned"],
    }
    for seed in SEEDS_5:
        for arm, extra in backbone_arms.items():
            add("backbone-attn", arm, seed, _dh2("configs/xes3g5m.yaml", seed, [*extra, *attn]))

    assist = "configs/assist2012.yaml"
    for seed in SEEDS_5:
        add("assist", "incidence-off", seed, _dh2(assist, seed, ["--question-graph", "--question-incidence-control", "zero"]))
        add(
            "assist",
            "incidence-on",
            seed,
            _dh2(assist, seed, ["--question-graph", "--question-incidence-control", "observed"]),
        )
        add(
            "assist",
            "time-on-observed",
            seed,
            _dh2(
                assist,
                seed,
                ["--question-graph", "--question-incidence-control", "observed", "--time-gap", "--time-gap-mode", "both"],
            ),
        )
        add(
            "assist",
            "time-on-zero",
            seed,
            _dh2(
                assist,
                seed,
                ["--question-graph", "--question-incidence-control", "zero", "--time-gap", "--time-gap-mode", "both"],
            ),
        )
        add(
            "assist",
            "t-zero",
            seed,
            _dh2(
                assist,
                seed,
                [
                    "--question-graph",
                    "--question-incidence-control",
                    "zero",
                    "--time-gap",
                    "--time-gap-mode",
                    "both",
                    "--time-gap-control",
                    "zero",
                ],
            ),
        )
        add(
            "assist",
            "t-misaligned",
            seed,
            _dh2(
                assist,
                seed,
                [
                    "--question-graph",
                    "--question-incidence-control",
                    "zero",
                    "--time-gap",
                    "--time-gap-mode",
                    "both",
                    "--time-gap-control",
                    "misaligned",
                ],
            ),
        )

    # Same six arms as "assist", frozen XES recipe, so corpus is the only
    # change from the XES ladder.
    assist_ref_arms = {
        "incidence-off": zero_q,
        "incidence-on": ["--question-graph", "--question-incidence-control", "observed"],
        "time-on-observed": [
            "--question-graph", "--question-incidence-control", "observed",
            "--time-gap", "--time-gap-mode", "both",
        ],
        "time-on-zero": timed,
        "t-zero": [*timed, "--time-gap-control", "zero"],
        "t-misaligned": [*timed, "--time-gap-control", "misaligned"],
    }
    for seed in SEEDS_5:
        for arm, extra in assist_ref_arms.items():
            add("assist-ref", arm, seed, _dh2(assist, seed, [*extra, *REF_RECIPE]))

    for model in ("sparsekt", "stablekt"):
        for seed in SEEDS_3:
            add(
                "recent-baselines",
                model,
                seed,
                None,
                block=(
                    "vendored pykt_engine forwards only dkt/akt/gkt/simplekt/"
                    "gikt/sakt/skt/dygkt/dgekt; do not substitute DyGKT"
                ),
            )

    for seed in SEEDS_5:
        add(
            "package-capacity",
            "no-time-param-padded",
            seed,
            _dh2(
                "configs/xes3g5m.yaml",
                seed,
                [
                    "--question-graph",
                    "--question-incidence-control",
                    "zero",
                    "--time-gap-pad",
                ],
            ),
        )

    # Budget only: this group kept the YAML graph, session hyperedges,
    # graph dropout 0.15 and the auxiliary loss, so it is not the frozen
    # recipe. Kept for provenance; package-matched is the paired control.
    ref_budget = ["--batch-size", "16", "--epochs", "30", "--early-stop-patience", "5"]
    for seed in SEEDS_5:
        add(
            "package-refbudget",
            "no-time-param-padded",
            seed,
            _dh2(
                "configs/xes3g5m.yaml",
                seed,
                [
                    "--question-graph",
                    "--question-incidence-control",
                    "zero",
                    "--time-gap-pad",
                    *ref_budget,
                ],
            ),
        )

    junyi = "configs/junyi.yaml"
    for seed in SEEDS_5:
        add(
            "junyi-partial",
            "no-time",
            seed,
            _dh2(junyi, seed, ["--question-graph", "--question-incidence-control", "zero"], max_users=50000),
        )
        add(
            "junyi-partial",
            "time-package",
            seed,
            _dh2(
                junyi,
                seed,
                ["--question-graph", "--question-incidence-control", "zero", "--time-gap", "--time-gap-mode", "both"],
                max_users=50000,
            ),
        )
        add(
            "junyi-partial",
            "t-zero",
            seed,
            _dh2(
                junyi,
                seed,
                [
                    "--question-graph",
                    "--question-incidence-control",
                    "zero",
                    "--time-gap",
                    "--time-gap-mode",
                    "both",
                    "--time-gap-control",
                    "zero",
                ],
                max_users=50000,
            ),
        )

    for seed in SEEDS_5:
        add(
            "package-matched",
            "no-time-param-padded",
            seed,
            _dh2("configs/xes3g5m.yaml", seed, [*zero_q, "--time-gap-pad", *REF_RECIPE]),
        )

    # All learners, frozen XES recipe, so corpus is the only change from XES.
    junyi_arms = {
        "no-time": zero_q,
        "time-package": timed,
        "t-zero": [*timed, "--time-gap-control", "zero"],
    }
    for seed in SEEDS_5:
        for arm, extra in junyi_arms.items():
            add("junyi-full", arm, seed, _dh2(junyi, seed, [*extra, *REF_RECIPE]))
    return jobs


def _free_vram_mib() -> int | None:
    if shutil.which("nvidia-smi") is None:
        return None
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return None
    return int(lines[0])


def _stamp(job: dict) -> Path:
    log_dir = REPO / "logs" / "credit_ladder"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{job['group']}_{job['arm']}_s{job['seed']}"


def _run_job(job: dict, *, force: bool) -> int:
    stem = _stamp(job)
    ok = stem.with_suffix(".ok")
    if ok.exists() and not force:
        print(f"SKIP  {job['group']} {job['arm']} s{job['seed']} ({ok.name})")
        return 0
    log = stem.with_suffix(".log")
    print("launch:", " ".join(job["cmd"]))
    print(f"log: {log}")
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(job["cmd"], cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode == 0:
        ok.write_text("ok\n", encoding="utf-8")
    else:
        print(f"FAIL  {job['group']} {job['arm']} s{job['seed']} exit={proc.returncode}")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="Start the next unfinished runnable job if VRAM allows")
    parser.add_argument("--launch-all", action="store_true", help="Run every unfinished runnable job in order")
    parser.add_argument("--force", action="store_true", help="Ignore existing .ok stamps")
    parser.add_argument("--min-vram-mib", type=int, default=20000)
    parser.add_argument("--group", default=None, help="Restrict to one group name")
    args = parser.parse_args()
    if args.launch and args.launch_all:
        raise SystemExit("use either --launch or --launch-all")

    jobs = _jobs()
    if args.group:
        jobs = [j for j in jobs if j["group"] == args.group]
    runnable = [j for j in jobs if j["cmd"] is not None]
    blocked = [j for j in jobs if j["cmd"] is None]
    payload = {
        "n_jobs": len(jobs),
        "n_runnable": len(runnable),
        "n_blocked": len(blocked),
        "gpu_note": "Planned device is one RTX 3090 24GB. This script refuses a smaller GPU.",
        "jobs": [
            {
                "group": j["group"],
                "arm": j["arm"],
                "seed": j["seed"],
                "blocked": j["block"],
                "cmd": j["cmd"],
            }
            for j in jobs
        ],
    }
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {STATUS} ({len(runnable)} runnable, {len(blocked)} blocked)")
    for j in jobs:
        if j["block"]:
            print(f"BLOCK {j['group']} {j['arm']} s{j['seed']}: {j['block']}")
        else:
            print(f"RUN   {j['group']} {j['arm']} s{j['seed']}")

    if not args.launch and not args.launch_all:
        return 0
    free = _free_vram_mib()
    print(f"free VRAM MiB: {free}")
    if free is None or free < args.min_vram_mib:
        print(
            f"refuse launch: need {args.min_vram_mib} MiB free "
            "(planned RTX 3090). No AUC was written."
        )
        return 2
    pending = [
        j for j in runnable
        if args.force or not _stamp(j).with_suffix(".ok").exists()
    ]
    if not pending:
        print("nothing pending")
        return 0
    queue = pending if args.launch_all else pending[:1]
    rc = 0
    for job in queue:
        rc = _run_job(job, force=args.force)
        if rc != 0:
            return rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
