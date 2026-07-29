"""
Phase 12 — differentiable dispatch + parallel scenarios + WASM bridge.

Coverage:
  (a) solve_dispatch_with_sensitivities matches numerical gradient within
      1e-4 on an interior 3-gen problem;
  (b) bound-active case: pinned gens get zero dp_dmc, unpinned gens
      absorb capacity sensitivity;
  (c) EconomicDispatchLayer.backward matches the analytical Jacobian
      composed with a random grad-output;
  (d) infeasible demand raises ValueError;
  (e) TorchDispatchLayer raises a friendly RuntimeError when torch
      is missing — documents the torch-optional boundary;
  (f) run_scenarios_parallel with backend="serial" returns results in
      original scenario order;
  (g) run_scenarios_parallel with backend="process" produces identical
      output to serial on a pickle-safe solve_fn;
  (h) ParallelResult.parallel_efficiency is within [0, 1] and > 0
      whenever the per-scenario timings are positive;
  (i) export_lp_for_browser round-trips through json.dumps / loads;
  (j) import_result_from_browser rebuilds an OptimisationResult with
      the expected dispatch arrays;
  (k) committable / extendable systems raise ValueError on export —
      documents the WASM LP-only boundary.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import nexus_energy as ne
from nexus_energy.diff import (
    EconomicDispatchLayer,
    TorchDispatchLayer,
    numerical_jacobian,
    solve_dispatch_with_sensitivities,
    torch_available,
)
from nexus_energy.cloud import run_scenarios_parallel, ParallelResult
from nexus_energy.browser import (
    WASM_SCHEMA_VERSION,
    export_lp_for_browser,
    import_result_from_browser,
)


# ---------------------------------------------------------------------------
# (a) interior gradient check
# ---------------------------------------------------------------------------

def test_interior_gradient_matches_numerical():
    mc = np.array([10.0, 30.0, 50.0])
    cap = np.array([100.0, 100.0, 100.0])
    demand = 150.0
    p, jac = solve_dispatch_with_sensitivities(mc, cap, demand, ridge=1.0)
    assert p.sum() == pytest.approx(demand)

    def fn(c):
        pp, _ = solve_dispatch_with_sensitivities(c, cap, demand, ridge=1.0)
        return pp
    num = numerical_jacobian(fn, mc, eps=1e-4)
    assert np.max(np.abs(jac.dp_dmc - num)) < 1e-4

    def fn_d(x):
        pp, _ = solve_dispatch_with_sensitivities(mc, cap, float(x[0]), ridge=1.0)
        return pp
    num_d = numerical_jacobian(fn_d, np.array([demand]), eps=1e-4).ravel()
    assert np.max(np.abs(jac.dp_ddemand - num_d)) < 1e-4


# ---------------------------------------------------------------------------
# (b) bound-active gradient check
# ---------------------------------------------------------------------------

def test_bound_active_gradient():
    mc = np.array([5.0, 20.0, 80.0])
    cap = np.array([40.0, 100.0, 100.0])
    demand = 90.0
    p, jac = solve_dispatch_with_sensitivities(mc, cap, demand, ridge=0.5)
    # gen 0 at cap, gen 1 absorbs the rest, gen 2 at zero.
    assert p[0] == pytest.approx(40.0)
    assert p[2] == pytest.approx(0.0, abs=1e-6)
    assert p.sum() == pytest.approx(demand)
    # Pinned gens have zero cost sensitivity.
    assert np.allclose(jac.dp_dmc[:, 0], 0.0)
    assert np.allclose(jac.dp_dmc[:, 2], 0.0)
    # Capacity of the at-cap gen moves the dispatch 1:1 for itself.
    assert jac.dp_dcap[0, 0] == pytest.approx(1.0)
    # Numerical check on the full capacity sensitivity.
    def fn_cap(x):
        pp, _ = solve_dispatch_with_sensitivities(mc, x, demand, ridge=0.5)
        return pp
    num_cap = numerical_jacobian(fn_cap, cap, eps=1e-4)
    assert np.max(np.abs(jac.dp_dcap - num_cap)) < 1e-4


# ---------------------------------------------------------------------------
# (c) stateful layer composition
# ---------------------------------------------------------------------------

def test_economic_dispatch_layer_backward():
    layer = EconomicDispatchLayer(ridge=0.5)
    mc = np.array([8.0, 25.0, 90.0])
    cap = np.array([50.0, 60.0, 70.0])
    demand = 100.0
    p = layer.forward(mc, cap, demand)
    assert p.sum() == pytest.approx(demand)

    # Random downstream loss ∇.
    rng = np.random.default_rng(0)
    grad_out = rng.normal(size=p.shape)
    grads = layer.backward(grad_out)

    # Numerical check on the scalar loss L(p) = grad_out · p.
    def loss_fn_mc(c):
        pp, _ = solve_dispatch_with_sensitivities(c, cap, demand, ridge=0.5)
        return np.array([float(grad_out @ pp)])
    num_mc = numerical_jacobian(loss_fn_mc, mc, eps=1e-4).ravel()
    assert np.max(np.abs(grads["marginal_cost"] - num_mc)) < 1e-4

    def loss_fn_d(x):
        pp, _ = solve_dispatch_with_sensitivities(mc, cap, float(x[0]),
                                                  ridge=0.5)
        return np.array([float(grad_out @ pp)])
    num_d = float(numerical_jacobian(loss_fn_d, np.array([demand]),
                                     eps=1e-4).ravel()[0])
    assert abs(grads["demand"] - num_d) < 1e-4


# ---------------------------------------------------------------------------
# (d) infeasible demand
# ---------------------------------------------------------------------------

def test_infeasible_demand_raises():
    with pytest.raises(ValueError, match="infeasible"):
        solve_dispatch_with_sensitivities(
            marginal_cost=np.array([10.0]),
            capacity=np.array([50.0]),
            demand=100.0,
        )


# ---------------------------------------------------------------------------
# (e) TorchDispatchLayer — torch-optional boundary
# ---------------------------------------------------------------------------

def test_torch_dispatch_layer_errors_without_torch():
    if not torch_available:
        with pytest.raises(RuntimeError, match="requires PyTorch"):
            TorchDispatchLayer()


# ---------------------------------------------------------------------------
# (f) & (g) parallel scenario runner
# ---------------------------------------------------------------------------

def _square(x: float) -> float:
    return float(x) ** 2


def test_parallel_runner_serial_preserves_order():
    scenarios = [1.0, 2.0, 3.0, 4.0]
    out = run_scenarios_parallel(_square, scenarios, n_workers=1,
                                 backend="serial")
    assert out.results == [1.0, 4.0, 9.0, 16.0]
    assert out.backend == "serial"
    assert out.n_workers == 1


def test_parallel_runner_process_matches_serial():
    scenarios = [1.5, 2.5, 3.5, 4.5]
    serial = run_scenarios_parallel(_square, scenarios, n_workers=1,
                                    backend="serial")
    proc = run_scenarios_parallel(_square, scenarios, n_workers=2,
                                  backend="process")
    assert proc.backend == "process"
    assert proc.results == serial.results


# ---------------------------------------------------------------------------
# (h) parallel efficiency bounds
# ---------------------------------------------------------------------------

def test_parallel_efficiency_within_bounds():
    res = ParallelResult(
        results=[0, 0, 0, 0],
        per_scenario_seconds=[0.1, 0.1, 0.1, 0.1],
        wall_clock_seconds=0.11,
        n_workers=4,
        backend="process",
    )
    eff = res.parallel_efficiency
    # 4 × 0.1 s / (4 × 0.11) ≈ 0.91
    assert 0.0 <= eff <= 1.0
    assert eff > 0.8

    empty = ParallelResult()
    assert empty.parallel_efficiency == 0.0


# ---------------------------------------------------------------------------
# (i) WASM LP export round-trips through JSON
# ---------------------------------------------------------------------------

def _small_lp_system(T: int = 4) -> ne.EnergySystem:
    sys = ne.EnergySystem("wasm-demo")
    elec = sys.add_bus("elec", carrier="electricity")
    sys.add_load("d", bus=elec, amount=np.full(T, 80.0))
    sys.add_generator("cheap", bus=elec, capacity=60, marginal_cost=10)
    sys.add_generator("peak", bus=elec, capacity=200, marginal_cost=120)
    sys.set_timesteps(T)
    return sys


def test_wasm_export_json_roundtrip():
    sys = _small_lp_system()
    payload = export_lp_for_browser(sys)
    assert payload["schema"] == WASM_SCHEMA_VERSION
    assert payload["timesteps"] == 4
    assert len(payload["generators"]) == 2
    assert len(payload["loads"]) == 1
    # Must survive json.dumps without a custom encoder.
    blob = json.dumps(payload)
    restored = json.loads(blob)
    assert restored == payload


# ---------------------------------------------------------------------------
# (j) result import path
# ---------------------------------------------------------------------------

def test_wasm_import_result_rebuilds_arrays():
    payload = {
        "schema": WASM_SCHEMA_VERSION,
        "status": "optimal",
        "total_cost": 1234.0,
        "solve_time": 0.02,
        "generator_dispatch": {"cheap": [60, 60, 60, 60],
                               "peak": [20, 20, 20, 20]},
        "bus_shadow_prices": {"elec": [120, 120, 120, 120]},
    }
    res = import_result_from_browser(payload)
    assert res.status == "optimal"
    assert res.total_cost == pytest.approx(1234.0)
    assert res.generator_dispatch["cheap"].tolist() == [60, 60, 60, 60]
    assert res.bus_shadow_prices["elec"].tolist() == [120, 120, 120, 120]


def test_wasm_import_rejects_wrong_schema():
    with pytest.raises(ValueError, match="schema"):
        import_result_from_browser({"schema": "0.0"})


# ---------------------------------------------------------------------------
# (k) WASM LP boundary — committable / extendable fail fast
# ---------------------------------------------------------------------------

def test_wasm_export_rejects_committable():
    sys = ne.EnergySystem("uc")
    bus = sys.add_bus("e", carrier="electricity")
    sys.add_load("d", bus=bus, amount=100.0)
    sys.add_generator("g", bus=bus, capacity=200, marginal_cost=40,
                      committable=True, min_up_time=1, min_down_time=1)
    sys.set_timesteps(4)
    with pytest.raises(ValueError, match="committable"):
        export_lp_for_browser(sys)
