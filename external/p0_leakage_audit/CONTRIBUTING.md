# Contributing

Thank you for interest in improving this research software.

## Scope

This repository accompanies a scientific manuscript. Changes that alter
experimental claims, reported numbers, or protocol definitions should be
discussed in a GitHub Issue before a large pull request.

## Development setup

1. Create a virtual environment (Python ≥ 3.10).
2. `pip install -r requirements.txt && pip install -e .`
3. Optional pyKT stack: `git submodule update --init --recursive`, then
   `pip install -e ".[pykt]"` and `pip install -e third_party/pykt-toolkit`.
4. Run `pytest -q` from the repository root.

## Pull requests

- Keep diffs focused; do not commit raw datasets, credentials, or large cache
  directories (`data/raw`, `data/processed`, `results/cache`, …).
- If you regenerate paper tables/figures, note the commands used in the PR
  description and keep outputs under `results/tables` / `results/figures` only
  when they are intended for the manuscript.
- Match existing code style; add or update tests when changing `src/` behaviour.

## Licence

By contributing, you agree that your contributions are licensed under the MIT
License (`LICENSE`).
