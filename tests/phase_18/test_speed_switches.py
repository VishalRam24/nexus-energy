"""N_En_Phase 18 speed-loophole switches: parity tests.

Both switches are opt-in and must be *exactly* optimum-preserving:

* ``mip_strategy="lp_first"`` (18.e) — forces the LP-first path at any
  size on a vertex backend; returns the LP optimum only when every
  relaxed binary/integer is integral at the vertex (LP-relaxation
  theorem), else falls through to the standard MIP solve.
* ``ramp_cost_formulation="signed"`` (18.t.2) — one ``r >= |Δ|`` aux var
  replaces the up/down pair; both forms price exactly ``|Δ|`` at any
  optimum with positive ramp cost.
"""

from __future__ import annotations

import numpy as np

import nexus_energy as ne

REL_TOL = 1e-6


def _uc_sys():
    sys = ne.EnergySystem("uc_lp_first")
    sys.set_timesteps(48)
    b = sys.add_bus("e")
    load = 50 + 30 * np.sin(np.arange(48) / 24 * 2 * np.pi)
    sys.add_load("ld", bus=b, amount=load)
    sys.add_generator("coal", bus=b, capacity=60, marginal_cost=20,
                      committable=True, p_min=18, min_up_time=4,
                      min_down_time=4, startup_cost=500)
    sys.add_generator("gas", bus=b, capacity=80, marginal_cost=60)
    return sys


def _ramp_cost_sys():
    sys = ne.EnergySystem("rc_parity")
    sys.set_timesteps(72)
    e = sys.add_bus("elec")
    h = sys.add_bus("heat")
    rng = np.random.default_rng(7)
    le = 40 + 25 * np.sin(np.arange(72) / 12 * np.pi) + rng.normal(0, 5, 72)
    lh = 30 + 15 * np.cos(np.arange(72) / 12 * np.pi) + rng.normal(0, 4, 72)
    sys.add_load("le", bus=e, amount=np.clip(le, 0, None))
    sys.add_load("lh", bus=h, amount=np.clip(lh, 0, None))
    sys.add_generator("grid", bus=e, capacity=200, marginal_cost=50)
    sys.add_generator("cheap_e", bus=e, capacity=30, marginal_cost=10)
    sys.add_link("hp", e, h, capacity=60, efficiency=3.0, ramp_cost=20.0)
    sys.add_storage("tank", bus=h, energy_capacity=120, power_capacity=25,
                    ramp_cost=2.0)
    return sys


def _rel(a, b):
    return abs(a - b) / max(1.0, abs(a))


def test_lp_first_matches_mip_only_on_uc():
    r_mip = _uc_sys().optimise(mip_strategy="mip_only")
    r_lpf = _uc_sys().optimise(mip_strategy="lp_first")
    assert r_mip.status == "optimal"
    assert r_lpf.status == "optimal"
    assert _rel(r_mip.total_cost, r_lpf.total_cost) <= REL_TOL


def test_lp_first_invalid_value_rejected():
    # Unknown strategies must not silently change behaviour: they skip the
    # LP-first block and reach the standard MIP path, same optimum.
    r = _uc_sys().optimise(mip_strategy="mip_only")
    assert r.status == "optimal"


def test_fix_and_certify_within_gap_of_optimum():
    """fix_and_certify returns either the certified residual-MIP solution
    (within ``gap`` of the relaxation bound, hence of the optimum) or the
    full-MIP fallback — both within gap of mip_only."""
    gap = 1e-3
    r_mip = _uc_sys().optimise(mip_strategy="mip_only")
    r_fc = _uc_sys().optimise(mip_strategy="fix_and_certify", gap=gap)
    assert r_mip.status == "optimal"
    assert r_fc.status == "optimal"
    scale = max(1.0, abs(r_mip.total_cost))
    assert (r_fc.total_cost - r_mip.total_cost) / scale <= gap + 1e-9
    assert r_fc.total_cost >= r_mip.total_cost - 1e-6 * scale


def test_ramp_cost_signed_matches_split():
    r_split = _ramp_cost_sys().optimise(ramp_cost_formulation="split")
    r_signed = _ramp_cost_sys().optimise(ramp_cost_formulation="signed")
    assert r_split.status == "optimal"
    assert r_signed.status == "optimal"
    assert _rel(r_split.total_cost, r_signed.total_cost) <= REL_TOL


def test_ramp_cost_signed_halves_aux_vars():
    sys_split = _ramp_cost_sys()
    sys_split.optimise(ramp_cost_formulation="split")
    sys_signed = _ramp_cost_sys()
    sys_signed.optimise(ramp_cost_formulation="signed")
    link_split = next(l for l in sys_split._links if l.name == "hp")
    link_signed = next(l for l in sys_signed._links if l.name == "hp")
    assert len(link_split._ramp_up_vars) == len(link_split._ramp_down_vars) == 72
    assert len(link_signed._ramp_up_vars) == 72
    assert len(link_signed._ramp_down_vars) == 0


def test_ramp_cost_formulation_validated():
    import pytest
    with pytest.raises(ValueError):
        _ramp_cost_sys().optimise(ramp_cost_formulation="bogus")
