# Third-party components

## pykt-toolkit (git submodule)

Upstream: [pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit) (MIT Licence).

After cloning this repository, fetch the pinned revision:

```bash
git submodule update --init --recursive
```

Install into the active environment (recommended after `pip install -e ".[pykt]"`):

```bash
pip install -e third_party/pykt-toolkit
```

The submodule commit recorded by Git is the supported revision for paper
reproduction. Prefer this local install over an arbitrary PyPI release when
reproducing the manuscript experiments.

## Other scientific dependencies

NumPy, pandas, scikit-learn, PyTorch, and related packages are installed from
`requirements.txt` / `pyproject.toml` optional extras and remain under their
respective upstream open-source licences.
