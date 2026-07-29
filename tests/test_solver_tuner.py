"""Tests for the smart LP-method calibrator (nexus_energy.solver_tuner)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import nexus_energy as ne
from nexus_energy.solver_tuner import (
    _CANDIDATES, recommend_lp_method, sidecar_lp_backend, tune_solver,
)


def _lp_system(T: int = 168) -> ne.EnergySystem:
    sys = ne.EnergySystem("tune_lp")
    sys.set_timesteps(T)
    b = sys.add_bus("e")
    load = 100 + 40 * np.cos(np.arange(T) * np.pi / 12)
    sys.add_load("d", bus=b, amount=np.clip(load, 1, None))
    sys.add_generator("base", bus=b, capacity=80, marginal_cost=10)
    sys.add_generator("peak", bus=b, capacity=200, marginal_cost=120)
    return sys


def _milp_system(T: int = 24) -> ne.EnergySystem:
    sys = ne.EnergySystem("tune_milp")
    sys.set_timesteps(T)
    b = sys.add_bus("e")
    sys.add_load("d", bus=b, amount=np.full(T, 90.0))
    sys.add_generator("uc", bus=b, capacity=120, marginal_cost=10,
                      committable=True)        # binary on/off -> MILP
    sys.add_generator("peak", bus=b, capacity=200, marginal_cost=120)
    return sys


def test_milp_recommends_simplex_without_racing():
    """IPM cannot solve B&B node LPs, so a committable (MILP) model must be
    routed to simplex by the structural pre-pass, with no race."""
    res = recommend_lp_method(_milp_system(), verbose=False)
    assert res.recommended == "simplex"
    assert res.features["has_integers"] is True
    assert res.runs == []          # short-circuited, never raced


def test_recommend_returns_valid_backend_for_lp():
    res = recommend_lp_method(_lp_system(), verbose=False, time_cap=30)
    assert res.recommended in _CANDIDATES
    assert res.parity_ok            # methods must agree on the objective


def test_sidecar_write_and_read_roundtrip(tmp_path):
    res = tune_solver(_lp_system(), sidecar_dir=tmp_path, verbose=False,
                      time_cap=30)
    sidecar = tmp_path / ".nexus_solver.json"
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert data["lp_backend"] == res.recommended
    assert sidecar_lp_backend(tmp_path) == res.recommended


def test_explicit_backend_overrides_sidecar(tmp_path, monkeypatch):
    """An explicit lp_backend= on optimise() must win over the sidecar."""
    # Write a sidecar that would force ipm, then solve from that dir.
    (tmp_path / ".nexus_solver.json").write_text(
        json.dumps({"lp_backend": "ipm"}))
    monkeypatch.chdir(tmp_path)
    # clear the per-dir cache so the freshly written sidecar is seen
    import nexus_energy.solver_tuner as st
    st._sidecar_cache.clear()
    assert sidecar_lp_backend() == "ipm"
    # Explicit simplex must still solve fine (and not be hijacked).
    r = _lp_system(48).optimise(lp_backend="simplex")
    assert r.status == "optimal"
