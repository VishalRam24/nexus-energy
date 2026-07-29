"""Rajan-Takriti tight min-up / min-down tests."""

from __future__ import annotations

import numpy as np

import nexus_energy as ne


def test_min_up_time_enforced():
    """Unit with min_up_time=3. Demand spike at t=0 for 1 hour only,
    followed by 5 hours of zero. Unit must stay on >= 3 hours once started."""
    demand = np.array([80, 0, 0, 0, 0, 0])
    sys = ne.EnergySystem("mu")
    sys.set_timesteps(6)
    b = sys.add_bus("e")
    sys.add_load("ld", bus=b, amount=demand)
    sys.add_generator("g", bus=b, capacity=100, marginal_cost=20,
                      committable=True, min_up_time=3, startup_cost=10.0)
    # Cheap slack backup so infeasibility isn't a concern.
    sys.add_generator("slack", bus=b, capacity=100, marginal_cost=500)
    result = sys.optimise()
    assert result.status == "optimal"
    gen = sys._generators[0]
    u = np.array([result._raw.value(v) for v in gen._status_vars])
    # If the unit ever started (v[t]=1), then u must be 1 for the next
    # TU-1 steps too. Check by scanning v.
    v = np.array([result._raw.value(v) for v in gen._startup_vars])
    for t in range(6):
        if v[t] > 0.5:
            for s in range(t, min(t + 3, 6)):
                assert u[s] >= 0.5, f"min-up violated: v[{t}]=1, u[{s}]={u[s]}"


def test_min_down_time_enforced():
    """Unit with min_down_time=3. Once shut down, must stay off >= 3 h."""
    demand = np.array([80, 80, 0, 80, 80, 80])
    sys = ne.EnergySystem("md")
    sys.set_timesteps(6)
    b = sys.add_bus("e")
    sys.add_load("ld", bus=b, amount=demand)
    sys.add_generator("g", bus=b, capacity=100, marginal_cost=20,
                      committable=True, min_down_time=3, shutdown_cost=10.0)
    sys.add_generator("slack", bus=b, capacity=100, marginal_cost=500)
    result = sys.optimise()
    assert result.status == "optimal"
    gen = sys._generators[0]
    u = np.array([result._raw.value(v) for v in gen._status_vars])
    w = np.array([result._raw.value(v) for v in gen._shutdown_vars])
    for t in range(6):
        if w[t] > 0.5:
            for s in range(t, min(t + 3, 6)):
                assert u[s] <= 0.5 + 1e-6, \
                    f"min-down violated: w[{t}]=1, u[{s}]={u[s]}"
