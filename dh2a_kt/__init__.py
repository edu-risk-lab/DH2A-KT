"""DH2A-KT: Dynamic Heterogeneous Hypergraph + Agentic Knowledge Tracing.

Idea D — builds on the leakage-controlled audit protocol released with P0
(Dao Minh Tuan et al., APIN submission; repo:
https://github.com/edu-risk-lab/leakage-controlled-kt-audit).

Package deliberately named ``dh2a_kt`` (not ``src``) so it can coexist on
sys.path with the vendored P0 source, which uses the top-level package name
``src`` internally. See ``dh2a_kt/p0_bridge.py``.
"""

__version__ = "0.1.0-scaffold"
