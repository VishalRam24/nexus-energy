"""Milestone 1 — parallel-subproblem Benders must equal serial Benders.

The process-pool backend (``n_jobs > 1``) only changes *where* each independent
operational subproblem solves, never *what* it computes. So the converged
objective, iteration count, and sub-solve count must be identical to the serial
loop. For decomposition, this equality IS the correctness proof.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne
from nexus_energy.decomposition import BendersDecomposer
from nexus_energy.stochastic import Scenario, apply_scenario


def _build_system(T: int = 24) -> ne.EnergySystem:
    rng = np.random.default_rng(0)
    sys = ne.EnergySystem("par_test")
    bus = sys.add_bus("elec", carrier="electricity")
    hour = np.arange(T) % 24
    load = 120 + 60 * np.cos((hour - 18) * np.pi / 12) ** 2 + rng.normal(0, 1, T)
    sys.add_load("d", bus=bus, amount=np.clip(load, 1.0, None))
    sys.add_generator("base", bus=bus, capacity=80, marginal_cost=10, tech="gas")
    sys.add_generator("slack", bus=bus, capacity=5000, marginal_cost=5000)
    cf = np.clip(np.cos((hour - 12) * np.pi / 12), 0, None)
    sys.add_generator("solar", bus=bus, capacity=10, marginal_cost=0,
                      carrier_factor=cf, extendable=True, min_capacity=10,
                      max_capacity=600, capital_cost=40, tech="solar")
    sys.add_generator("peaker", bus=bus, capacity=10, marginal_cost=200,
                      extendable=True, min_capacity=10, max_capacity=600,
                      capital_cost=15, tech="peaker")
    return sys


def _scenarios(n: int, seed: int = 7) -> list[Scenario]:
    rng = np.random.default_rng(seed)
    return [
        Scenario(name=f"sc{i}", probability=1.0 / n,
                 demand_factor=float(np.clip(rng.normal(1.0, 0.1), 0.6, 1.4)),
                 carrier_factor_scale=float(np.clip(rng.normal(1.0, 0.15), 0.4, 1.5)))
        for i in range(n)
    ]


def _run(n_jobs: int, scenarios):
    sys = _build_system()
    subs = [apply_scenario(sys, s) for s in scenarios]
    decomp = BendersDecomposer(
        system=sys, subsystems=subs,
        period_weights=[s.probability for s in scenarios],
        max_iter=40, tol=1e-3, stabilisation="plain", n_jobs=n_jobs,
    )
    return decomp.solve()


@pytest.mark.parametrize("n_jobs", [2, 3])
def test_parallel_matches_serial(n_jobs):
    scenarios = _scenarios(6)
    serial = _run(1, scenarios)
    parallel = _run(n_jobs, scenarios)

    assert serial.status == "optimal"
    assert parallel.status == serial.status
    assert len(parallel.iterations) == len(serial.iterations)
    assert parallel.sub_solves == serial.sub_solves
    # The headline guarantee: identical optimum to the cent.
    assert parallel.total_cost == pytest.approx(serial.total_cost, abs=1e-6)


def test_subproblem_seconds_recorded():
    """The parallel kernel timing is populated (used by the scaling benchmark)."""
    res = _run(2, _scenarios(4))
    assert res.subproblem_seconds > 0.0
