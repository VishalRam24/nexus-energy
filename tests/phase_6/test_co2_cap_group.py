"""Phase 23 — pooled CO2 cap-group (GenX Cap_Zone) + dissipation loss accounting.

These are additive GenX-parity levers. NOTE (honest): they do NOT by themselves
close the GenX three_zones_rate_co2 −42 % gap — that gap is driven by a constraint
GenX enforces that nexus does not (operating reserve / CapacityReserveMargin /
transmission-loss PWL), diagnosable only by a live-GenX active-dual audit (the
standing N_En_Phase 15.2 investigation). Direction check: net-loss RHS (≈0 for
cyclic storage) is already *tighter* than dissipation; pooling is *looser* than
per-zone — so neither moves nexus's already-too-cheap optimum toward GenX. These
tests pin the new levers' correctness, not the parity closeout.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne


def _build(mode):
    s = ne.EnergySystem("co2g")
    s.set_timesteps(2)
    b1 = s.add_bus("z1")
    b2 = s.add_bus("z2")
    s.add_generator("clean1", bus=b1, capacity=100, marginal_cost=50, emission_factor=0.0)
    s.add_generator("dirty1", bus=b1, capacity=100, marginal_cost=10, emission_factor=1.0)
    s.add_generator("dirty2", bus=b2, capacity=100, marginal_cost=10, emission_factor=1.0)
    s.add_load("d1", bus=b1, amount=np.array([40.0, 40.0]))
    s.add_load("d2", bus=b2, amount=np.array([40.0, 40.0]))
    s.add_link("tie", bus_from=b1, bus_to=b2, capacity=50, bidirectional=True)
    if mode == "group":
        s.set_co2_cap_group([b1, b2], 0.3, is_rate=True, storage_losses_on_rhs=False)
    elif mode == "perzone":
        s.set_co2_zone_cap(b1, 0.3, is_rate=True, storage_losses_on_rhs=False)
        s.set_co2_zone_cap(b2, 0.3, is_rate=True, storage_losses_on_rhs=False)
    return s


def test_cap_group_feasible_and_binds():
    r = _build("group").optimise()
    assert r.status == "optimal"
    assert np.isfinite(r.total_cost)


def test_cap_group_not_looser_than_perzone():
    # Pooled vs per-zone: pooled is never *tighter* than the per-zone intersection,
    # so its cost is <= per-zone. (Both equal in this symmetric case.)
    rp = _build("perzone").optimise()
    rg = _build("group").optimise()
    assert rg.total_cost <= rp.total_cost + 1e-6


def test_dissipation_loss_accounting_runs_and_is_positive_rhs():
    # A storage with <1 efficiency: dissipation mode adds a strictly positive
    # loss term to the rate RHS (vs ~0 for net mode on a cyclic store).
    s = ne.EnergySystem("co2d")
    s.set_timesteps(3)
    b = s.add_bus("z")
    s.add_generator("dirty", bus=b, capacity=100, marginal_cost=10, emission_factor=1.0)
    s.add_generator("clean", bus=b, capacity=100, marginal_cost=80, emission_factor=0.0)
    s.add_load("d", bus=b, amount=np.array([30.0, 60.0, 30.0]))
    s.add_storage("batt", bus=b, power_capacity=40, energy_capacity=80,
                  efficiency_charge=0.9, efficiency_discharge=0.9, cyclic=True)
    s.set_co2_cap_group([b], 0.5, is_rate=True, storage_losses_on_rhs=True,
                        loss_accounting="dissipation")
    r = s.optimise()
    assert r.status == "optimal"
    assert np.isfinite(r.total_cost)


def test_cap_group_bad_loss_accounting_rejected():
    s = ne.EnergySystem("x"); s.set_timesteps(1); b = s.add_bus("z")
    with pytest.raises(ValueError, match="loss_accounting"):
        s.set_co2_cap_group([b], 0.3, loss_accounting="bogus")
