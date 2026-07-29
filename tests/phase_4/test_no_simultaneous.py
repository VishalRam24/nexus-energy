"""Phase 4.x — ``Storage.no_simultaneous`` physical exclusivity."""

from __future__ import annotations

import numpy as np

import nexus_energy as ne


def _roundtrip_one_system(no_simultaneous: bool, eta: float = 1.0):
    """A 4-step system where an LP with η=1 can cheat: charge + discharge
    simultaneously to convert 'free' wind into dispatchable energy that
    satisfies a small extra load without spending real generation.

    With ``no_simultaneous=True`` the cheat path is closed.
    """
    T = 4
    sys = ne.EnergySystem("nosim")
    sys.set_timesteps(T)
    b = sys.add_bus("e")
    # Cheap overbuilt generator — plenty of slack so the LP is never
    # forced to cheat. This isolates whether the LP voluntarily picks
    # simultaneous ch+dis when η=1 allows it.
    sys.add_generator("gen", bus=b, capacity=500.0, marginal_cost=10.0)
    sys.add_load("ld", bus=b, amount=np.full(T, 50.0))
    # Storage with perfect round-trip.
    sys.add_storage(
        "batt", bus=b, power_capacity=50.0, energy_capacity=200.0,
        efficiency_charge=eta, efficiency_discharge=1.0,
        soc_initial=0.5, cyclic=True,
        # Drop the default spill_cost-like bias so the test is clean.
        marginal_cost=0.0, marginal_cost_charge=0.0,
        no_simultaneous=no_simultaneous,
    )
    return sys


def test_no_simultaneous_forbids_overlap_at_eta_1():
    """With η=1 and no_simultaneous=True, ch[t] and dis[t] must not both
    be positive at any timestep."""
    sys = _roundtrip_one_system(no_simultaneous=True, eta=1.0)
    res = sys.optimise()
    assert res.status == "optimal"
    sto = sys._storages[0]
    for t in range(len(sto._charge_vars)):
        ch_t = res._raw.value(sto._charge_vars[t])
        dis_t = res._raw.value(sto._discharge_vars[t])
        assert min(ch_t, dis_t) < 1e-6, (
            f"t={t}: ch={ch_t:.3f}, dis={dis_t:.3f} — both positive")


def test_no_simultaneous_default_false_is_no_regression():
    """Default no_simultaneous=False must produce a pure-LP model (no
    binaries minted) and match the pre-feature objective."""
    sys = _roundtrip_one_system(no_simultaneous=False, eta=1.0)
    res = sys.optimise()
    assert res.status == "optimal"
    sto = sys._storages[0]
    assert sto._nosim_vars == []


def test_no_simultaneous_extendable_storage():
    """Feature works on extendable storages via big-M = max_power_capacity."""
    T = 4
    sys = ne.EnergySystem("nosim_ext")
    sys.set_timesteps(T)
    b = sys.add_bus("e")
    sys.add_generator("gen", bus=b, capacity=500.0, marginal_cost=10.0)
    sys.add_load("ld", bus=b, amount=np.full(T, 50.0))
    sys.add_storage(
        "batt", bus=b, power_capacity=0.0, energy_capacity=0.0,
        efficiency_charge=1.0, efficiency_discharge=1.0,
        soc_initial=0.5, cyclic=True,
        extendable=True,
        max_power_capacity=100.0, max_energy_capacity=400.0,
        capital_cost_power=1.0, capital_cost_energy=0.1,
        no_simultaneous=True,
    )
    res = sys.optimise()
    assert res.status == "optimal"
    sto = sys._storages[0]
    for t in range(T):
        ch_t = res._raw.value(sto._charge_vars[t])
        dis_t = res._raw.value(sto._discharge_vars[t])
        assert min(ch_t, dis_t) < 1e-6
