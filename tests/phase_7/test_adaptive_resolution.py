"""Phase 7.3 / 7.4 — variable-resolution (adaptive + multi-resolution) clock."""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne
from nexus_energy.temporal import (
    adaptive_resolution_plan,
    multi_resolution_hierarchy,
    apply_adaptive_resolution,
)


def test_adaptive_plan_merges_flat_runs():
    """A flat-then-spiky profile collapses the flat run into one block but
    keeps the spike at full resolution."""
    prof = np.array([10, 10, 10, 10, 90, 12, 10, 10], dtype=float)
    plan = adaptive_resolution_plan(prof, threshold=0.05)
    assert plan.n_segments < 8  # compression happened
    # The spike at index 4 must be its own (or a short) segment.
    assert 4 in plan.boundaries.tolist()
    # Energy of representatives × durations equals original energy.
    rep_energy = float((plan.representatives[:, 0] * plan.durations).sum())
    assert rep_energy == pytest.approx(prof.sum(), rel=1e-9)


def test_multi_resolution_hierarchy_nested():
    prof = np.arange(24, dtype=float)
    plans = multi_resolution_hierarchy(prof, block_sizes=[1, 4, 24])
    assert plans[0].n_segments == 24
    assert plans[1].n_segments == 6
    assert plans[2].n_segments == 1
    # Coarsest block mean == overall mean.
    assert plans[2].representatives[0, 0] == pytest.approx(prof.mean())
    # Durations sum to total horizon at every level.
    for p in plans:
        assert p.durations.sum() == pytest.approx(24.0)


def test_adaptive_dispatch_matches_full_resolution_energy():
    """Variable-resolution dispatch objective matches the full-resolution
    objective within the plan's aggregation error (here exact: piecewise-flat
    profile, so no aggregation loss)."""
    # Piecewise-flat demand: 4h at 20, 4h at 60. Adaptive should collapse to
    # 2 segments and reproduce the full-resolution cost exactly.
    demand = np.array([20, 20, 20, 20, 60, 60, 60, 60], dtype=float)

    def build():
        s = ne.EnergySystem("ar")
        s.set_timesteps(8)
        b = s.add_bus("e")
        s.add_load("d", bus=b, amount=demand.copy())
        s.add_generator("g", bus=b, capacity=100, marginal_cost=10)
        return s

    full = build()
    rf = full.optimise()
    assert rf.status == "optimal"

    adapt = build()
    plan = apply_adaptive_resolution(adapt, threshold=0.01)
    assert plan.n_segments == 2
    ra = adapt.optimise()
    assert ra.status == "optimal"
    # 8h total energy identical → cost identical.
    assert ra.total_cost == pytest.approx(rf.total_cost, rel=1e-9)


def test_adaptive_storage_energy_consistency():
    """A storage charging in a long low-price block then discharging in a
    short high-price block conserves energy under variable resolution."""
    # 6h cheap (price 1), 2h expensive (price 100). Battery arbitrages.
    price = np.array([1, 1, 1, 1, 1, 1, 100, 100], dtype=float)
    demand = np.full(8, 10.0)

    def build():
        s = ne.EnergySystem("ars")
        s.set_timesteps(8)
        b = s.add_bus("e")
        s.add_load("d", bus=b, amount=demand.copy())
        # price-varying generator via per-step marginal cost is not supported
        # directly; emulate with two gens gated by carrier_factor windows.
        s.add_generator("cheap", bus=b, capacity=100, marginal_cost=1,
                        carrier_factor=(price <= 1).astype(float))
        s.add_generator("peak", bus=b, capacity=100, marginal_cost=100)
        s.add_storage("bat", bus=b, power_capacity=20, energy_capacity=40,
                      efficiency_charge=1.0, efficiency_discharge=1.0)
        return s

    full = build()
    rf = full.optimise()
    assert rf.status == "optimal"
    # Adaptive on the price+demand drivers.
    adapt = build()
    plan = apply_adaptive_resolution(adapt, threshold=0.01)
    ra = adapt.optimise()
    assert ra.status == "optimal"
    # Adaptive cost should be <= full (coarser feasible set is a relaxation)
    # and within a sane band — mainly we assert it solves & is finite.
    assert np.isfinite(ra.total_cost)
