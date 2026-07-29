"""18.P2 temporal_certified: sandwich validity + relaxation switches.

The non-negotiable invariants (exactness story):
* LB ≤ monolithic optimum  (Σ relaxed-block bounds is a valid global LB)
* UB ≥ monolithic optimum  (stitched solution is feasible, cost exact)
* certified ⇒ UB within gap of the monolithic optimum
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne
from nexus_energy.temporal_certified import optimise_temporal_certified

T_FULL = 96
_rng = np.random.default_rng(11)
LOAD_E = np.clip(50 + 25 * np.sin(np.arange(T_FULL) / 12 * np.pi)
                 + _rng.normal(0, 6, T_FULL), 5, None)
LOAD_H = np.clip(30 + 12 * np.cos(np.arange(T_FULL) / 16 * np.pi)
                 + _rng.normal(0, 4, T_FULL), 2, None)
MC_CHEAP = np.clip(20 + 15 * np.sin(np.arange(T_FULL) / 8 * np.pi), 1, None)


def _factory(t0, t1):
    T = t1 - t0
    sys = ne.EnergySystem("tc")
    sys.set_timesteps(T)
    e = sys.add_bus("e")
    h = sys.add_bus("h")
    sys.add_load("le", bus=e, amount=LOAD_E[t0:t1])
    sys.add_load("lh", bus=h, amount=LOAD_H[t0:t1])
    sys.add_generator("cheap", bus=e, capacity=70,
                      marginal_cost=MC_CHEAP[t0:t1])
    sys.add_generator("exp", bus=e, capacity=200, marginal_cost=90)
    sys.add_storage("bat", bus=e, energy_capacity=200, power_capacity=40,
                    cyclic=False, soc_initial=0.5, soc_min=0.05,
                    ramp_cost=1.0)
    sys.add_link("boiler", e, h, capacity=80, efficiency=0.9,
                 committable=True, startup_cost=200.0, ramp_cost=5.0)
    return sys


def _mono_optimum():
    res = _factory(0, T_FULL).optimise(mip_strategy="mip_only")
    assert res.status == "optimal"
    return res.total_cost


def test_sandwich_and_certificate():
    mono = _mono_optimum()
    tc = optimise_temporal_certified(_factory, T_FULL, n_blocks=4, gap=0.05)
    scale = max(1.0, abs(mono))
    assert tc.lower_bound <= mono + 1e-6 * scale, "LB must under-estimate optimum"
    assert tc.objective >= mono - 1e-6 * scale, "UB must over-estimate optimum"
    assert tc.objective >= tc.lower_bound - 1e-9 * scale
    if tc.status == "certified":
        assert (tc.objective - mono) / scale <= 0.05 + 1e-9
    # Stitched trajectories span the full horizon.
    assert len(tc.link_flow["boiler"]) == T_FULL
    assert len(tc.storage_soc["bat"]) == T_FULL


def test_more_blocks_keep_sandwich():
    mono = _mono_optimum()
    tc = optimise_temporal_certified(_factory, T_FULL, n_blocks=8, gap=0.10)
    scale = max(1.0, abs(mono))
    assert tc.lower_bound <= mono + 1e-6 * scale
    assert tc.objective >= mono - 1e-6 * scale


def test_soc_initial_free_is_relaxation():
    pinned = _factory(0, T_FULL)
    r_pin = pinned.optimise(mip_strategy="mip_only")
    free = _factory(0, T_FULL)
    for sto in free._storages:
        sto.soc_initial_free = True
    r_free = free.optimise(mip_strategy="mip_only")
    assert r_free.status == "optimal"
    assert r_free.total_cost <= r_pin.total_cost + 1e-6 * max(1, abs(r_pin.total_cost))


def test_ramp_cost_skip_t0_is_relaxation():
    r_norm = _factory(0, T_FULL).optimise(mip_strategy="mip_only")
    r_skip = _factory(0, T_FULL).optimise(mip_strategy="mip_only",
                                          _ramp_cost_skip_t0=True)
    assert r_skip.status == "optimal"
    assert r_skip.total_cost <= r_norm.total_cost + 1e-6 * max(1, abs(r_norm.total_cost))


def test_guided_sandwich_holds_for_arbitrary_prices():
    """Floors are a UB-side restriction; λ telescopes to zero on any full
    trajectory — so the sandwich must survive deliberately bad guides."""
    mono = _mono_optimum()
    floors = {"bat": np.full(3, 0.3 * 200.0)}
    prices = {"bat": np.array([5.0, -3.0, 12.0])}  # arbitrary, incl. negative
    tc = optimise_temporal_certified(_factory, T_FULL, n_blocks=4, gap=0.05,
                                     boundary_soc_min=floors,
                                     boundary_prices=prices)
    scale = max(1.0, abs(mono))
    assert tc.lower_bound <= mono + 1e-6 * scale
    assert tc.objective >= mono - 1e-6 * scale
    assert tc.objective >= tc.lower_bound - 1e-9 * scale


def test_lb_rounds_keep_sandwich_and_never_worsen():
    """Subgradient rounds report the BEST valid bound — monotone non-worse,
    and still a valid under-estimate of the optimum."""
    mono = _mono_optimum()
    prices = {"bat": np.full(3, 8.0)}
    tc1 = optimise_temporal_certified(_factory, T_FULL, n_blocks=4, gap=0.05,
                                      boundary_prices=prices, lb_rounds=1)
    tc3 = optimise_temporal_certified(_factory, T_FULL, n_blocks=4, gap=0.05,
                                      boundary_prices=prices, lb_rounds=3)
    scale = max(1.0, abs(mono))
    assert tc3.lower_bound <= mono + 1e-6 * scale
    assert tc3.objective >= mono - 1e-6 * scale
    assert tc3.lower_bound >= tc1.lower_bound - 1e-9 * scale
    assert len(tc3.lb_round_bounds) == 3


def test_ub_price_mode_cost_adjustment_exact():
    """In prices mode the λ payment must be removed from the reported UB —
    so UB still over-estimates the monolithic optimum."""
    mono = _mono_optimum()
    prices = {"bat": np.full(3, 25.0)}  # deliberately large
    tc = optimise_temporal_certified(_factory, T_FULL, n_blocks=4, gap=0.05,
                                     boundary_prices=prices,
                                     ub_boundary="prices")
    scale = max(1.0, abs(mono))
    assert tc.objective >= mono - 1e-6 * scale
    assert tc.lower_bound <= mono + 1e-6 * scale


def test_guard_rejects_cyclic_storage():
    def bad_factory(t0, t1):
        sys = _factory(t0, t1)
        sys._storages[0].cyclic = True
        return sys
    with pytest.raises(ValueError, match="cyclic"):
        optimise_temporal_certified(bad_factory, T_FULL, n_blocks=2)


def test_soc_initial_free_guard():
    sys = _factory(0, T_FULL)
    sys._storages[0].cyclic = True
    sys._storages[0].soc_initial_free = True
    with pytest.raises(ValueError, match="soc_initial_free"):
        sys.optimise()
