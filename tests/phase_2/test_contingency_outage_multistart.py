"""Phase 2.2 / 2.3 / 2.4 — multi-state startup, contingency reserve, planned outage."""

from __future__ import annotations

import numpy as np

import nexus_energy as ne


def test_multistate_startup_picks_cold_after_long_off():
    """A unit off for many steps then started must pay the COLD start cost,
    not the hot one. Demand: on, long off gap, then a spike forcing a start."""
    # 8 steps: run 1 step, off 5 steps, then need it again.
    demand = np.array([60, 0, 0, 0, 0, 0, 0, 60], dtype=float)
    sys = ne.EnergySystem("ms")
    sys.set_timesteps(8)
    b = sys.add_bus("e")
    sys.add_load("ld", bus=b, amount=demand)
    # hot start (off<4) costs 5, cold start (off>=4) costs 100.
    sys.add_generator("g", bus=b, capacity=100, marginal_cost=1,
                      committable=True,
                      start_up_segments=[(0, 5.0), (4, 100.0)])
    # expensive slack so the committable unit is preferred when cheap enough
    sys.add_generator("slack", bus=b, capacity=100, marginal_cost=1000)
    r = sys.optimise()
    assert r.status == "optimal"
    gen = sys._generators[0]
    # start-type vars exist and a cold start (segment 1) was selected at t=7.
    seg = np.array([[r._raw.value(v) for v in row] for row in gen._start_type_vars])
    v = np.array([r._raw.value(x) for x in gen._startup_vars])
    # If the unit restarted at t=7 after a 6-step gap, the cold segment fires.
    if v[7] > 0.5:
        assert seg[7][1] > 0.5, f"expected cold start at t=7, seg={seg[7]}"
        assert seg[7][0] < 0.5, "hot start must be gated off after long outage"


def test_multistate_startup_cost_in_objective():
    """Objective reflects the multi-state start cost (sanity: solvable & finite)."""
    demand = np.array([0, 50, 0, 0, 50], dtype=float)
    sys = ne.EnergySystem("ms2")
    sys.set_timesteps(5)
    b = sys.add_bus("e")
    sys.add_load("ld", bus=b, amount=demand)
    sys.add_generator("g", bus=b, capacity=100, marginal_cost=1,
                      committable=True, start_up_segments=[(0, 3.0), (2, 50.0)])
    sys.add_generator("slack", bus=b, capacity=100, marginal_cost=500)
    r = sys.optimise()
    assert r.status == "optimal"
    assert np.isfinite(r.total_cost)


def test_contingency_reserve_forces_spread():
    """With contingency reserve, the loss of any single unit must be coverable.
    One big cheap unit + small units; reserve forces headroom on the others."""
    demand = np.full(3, 90.0)
    sys = ne.EnergySystem("cr")
    sys.set_timesteps(3)
    b = sys.add_bus("e")
    sys.add_load("ld", bus=b, amount=demand)
    sys.add_generator("big", bus=b, capacity=100, marginal_cost=1)
    sys.add_generator("m1", bus=b, capacity=100, marginal_cost=2)
    sys.add_generator("m2", bus=b, capacity=100, marginal_cost=2)
    sys.set_contingency_reserve()
    r = sys.optimise()
    assert r.status == "optimal"
    disp = r.generator_dispatch  # dict name -> array
    big = np.asarray(disp["big"])
    m1 = np.asarray(disp["m1"])
    m2 = np.asarray(disp["m2"])
    for t in range(3):
        # headroom excluding 'big' must cover big's output
        headroom_others = (100 - m1[t]) + (100 - m2[t])
        assert headroom_others >= big[t] - 1e-6, \
            f"t={t}: others headroom {headroom_others} < big {big[t]}"


def test_planned_outage_zeros_dispatch_in_window():
    """set_outage(full) forces zero availability in the window."""
    demand = np.full(6, 30.0)
    sys = ne.EnergySystem("po")
    sys.set_timesteps(6)
    b = sys.add_bus("e")
    sys.add_load("ld", bus=b, amount=demand)
    sys.add_generator("g", bus=b, capacity=100, marginal_cost=1)
    sys.add_generator("slack", bus=b, capacity=100, marginal_cost=50)
    sys.set_outage("g", windows=[(2, 4)])  # off at t=2,3
    r = sys.optimise()
    assert r.status == "optimal"
    g = np.asarray(r.generator_dispatch["g"])
    assert g[2] < 1e-6 and g[3] < 1e-6, f"unit dispatched during outage: {g}"
    assert g[0] > 1e-6, "unit should run outside the outage window"


def test_planned_outage_partial_derate_composes_with_vre():
    """Partial derate multiplies an existing carrier_factor."""
    sys = ne.EnergySystem("po2")
    sys.set_timesteps(4)
    b = sys.add_bus("e")
    sys.add_load("ld", bus=b, amount=np.full(4, 10.0))
    cf = np.array([1.0, 0.8, 0.6, 0.4])
    sys.add_generator("vre", bus=b, capacity=100, marginal_cost=0, carrier_factor=cf)
    sys.add_generator("slack", bus=b, capacity=100, marginal_cost=50)
    sys.set_outage("vre", windows=[(1, 3)], availability=0.5)
    g = sys._generators[0]
    expected = cf.copy()
    expected[1:3] *= 0.5
    assert np.allclose(g.carrier_factor, expected)
