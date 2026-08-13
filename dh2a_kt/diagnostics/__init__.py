"""Diagnostics that test design premises before they are built upon.

``black_confidence`` tests whether GreyKT's C_B signal (per-concept
training frequency, as a proxy for black-box reliability) actually
predicts black-box error. That premise underpins the reliability gate, so
it is checked against real validation predictions before any full GreyKT
training run is committed to.
"""
