"""N_En_Phase 18.a — PersistentDispatchSession (warm-started MPC resolves).

The load-bearing test is EQUIVALENCE: after any sequence of advance()
updates, the session's hot-resolved answer must match a freshly built
cold solve of an identical system to LP tolerance. Plus: warm resolves
take ≤ cold iterations, structure changes fall back to rebuild, MIP
systems are refused, and a 2-window MPC rollout carries SOC correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_energy.core import EnergySystem
from nexus_energy.mpc import PersistentDispatchSession

T = 24
RNG = np.random.default_rng(42)


def make_system(demand=None, cf_solar=None, mc_gas=45.0, soc0_free=True):
    sys = EnergySystem("mpc_site")
    sys.set_timesteps(T, dt=1.0)
    elec = sys.add_carrier("electricity", unit="MWh")
    b = sys.add_bus("site", carrier="electricity")
    base_d = (60 + 25 * np.sin(np.arange(T) / 24 * 2 * np.pi - 0.7)
              if demand is None else np.asarray(demand, float))
    sys.add_load("load", bus=b, amount=base_d)
    solar_cf = (np.clip(np.sin((np.arange(T) - 6) / 12 * np.pi), 0, None)
                if cf_solar is None else np.asarray(cf_solar, float))
    sys.add_generator("solar", bus=b, capacity=50.0, marginal_cost=0.0,
                      carrier_factor=solar_cf)
    sys.add_generator("gas", bus=b, capacity=120.0, marginal_cost=mc_gas)
    sys.add_generator("peaker", bus=b, capacity=60.0, marginal_cost=180.0)
    sys.add_storage("batt", bus=b, power_capacity=25.0, energy_capacity=80.0,
                    efficiency_charge=0.95, efficiency_discharge=0.95,
                    soc_initial=0.5, cyclic=False,
                    soc_initial_free=soc0_free)
    return sys


def assert_equivalent(res_a, res_b, tag=""):
    """Objective + per-generator ENERGY equivalence.

    Hour-by-hour dispatch is NOT compared: same-mc hour shuffles are
    degenerate alternative optima (identical cost), and the two solver
    paths legitimately pick different vertices. Totals are invariant.
    """
    assert res_a.status == "optimal" and res_b.status == "optimal"
    rel = abs(res_a.total_cost - res_b.total_cost) / max(abs(res_b.total_cost), 1.0)
    assert rel < 1e-6, f"{tag}: objective rel diff {rel:.2e}"
    for name, arr in res_b.generator_dispatch.items():
        ta = float(res_a.generator_dispatch[name].sum())
        tb = float(arr.sum())
        diff = abs(ta - tb) / max(abs(tb), 1.0)
        assert diff < 1e-5, f"{tag}: {name} total energy rel diff {diff:.2e}"


def test_equivalence_over_cycles():
    sys = make_system()
    sess = PersistentDispatchSession(sys)
    base = sess.build()
    assert base.status == "optimal"

    soc = 40.0
    for cycle in range(5):
        new_d = 60 + 25 * np.sin(np.arange(T) / 24 * 2 * np.pi - 0.7) \
            + RNG.normal(0, 5, T)
        new_cf = np.clip(
            np.sin((np.arange(T) - 6 + cycle) / 12 * np.pi)
            + RNG.normal(0, 0.05, T), 0, 1)
        new_mc = 45.0 + 5 * cycle + RNG.normal(0, 1)
        res = sess.advance(demand={"site": new_d},
                           cf={"solar": new_cf},
                           mc={"gas": float(new_mc)},
                           soc_init={"batt": soc})
        # Cold reference: identical fresh system with pinned start SOC.
        ref_sys = make_system(demand=new_d, cf_solar=new_cf,
                              mc_gas=float(new_mc))
        for st in ref_sys._storages:
            st.soc_initial_free_min = soc
            st.soc_initial_free_max = soc
        ref = ref_sys.optimise()
        assert_equivalent(res, ref, tag=f"cycle {cycle}")
        soc = float(res.storage_soc["batt"][-1])
    assert sess.n_rebuilds == 0
    assert sess.n_resolves == 5


def test_warm_iterations_not_worse():
    sys = make_system()
    sess = PersistentDispatchSession(sys)
    base = sess.build()
    cold_iters = None
    try:
        cold_iters = base._raw.iterations
    except Exception:
        pass
    res = sess.advance(demand={"site": 62 + 24 * np.sin(
        np.arange(T) / 24 * 2 * np.pi - 0.7)})
    assert res.status == "optimal"
    # The normal optimise() path may go through IPM (iterations
    # unreported / 0) — only compare when a meaningful count exists.
    if cold_iters and sess.last_iterations is not None and int(cold_iters) > 0:
        print(f"\n[18.a] cold iters={cold_iters} warm iters={sess.last_iterations}")
        assert sess.last_iterations <= int(cold_iters)
    else:
        print(f"\n[18.a] warm iters={sess.last_iterations} "
              f"(cold count unavailable: {cold_iters!r})")


def test_structure_change_falls_back_to_rebuild():
    sys = make_system()
    sess = PersistentDispatchSession(sys)
    sess.build()
    # Unknown generator name → rebuild path (still correct).
    res = sess.advance(mc={"does_not_exist": 99.0})
    assert res.status == "optimal"
    assert sess.n_rebuilds == 1


def test_mip_guard():
    sys = make_system()
    for g in sys._generators:
        if g.name == "gas":
            g.committable = True
    sess = PersistentDispatchSession(sys)
    with pytest.raises(NotImplementedError, match="committable"):
        sess.build()


def test_soc_pin_respected():
    sys = make_system()
    sess = PersistentDispatchSession(sys)
    sess.build()
    res = sess.advance(soc_init={"batt": 10.0})
    # Start SOC pinned: first-period SOC must be reachable from 10 MWh
    # within the 25 MW power limit (charge ≤ 25·0.95, discharge ≤ 25/0.95).
    soc0 = float(res.storage_soc["batt"][0])
    assert 10.0 - 25.0 / 0.95 - 1e-6 <= soc0 <= 10.0 + 25.0 * 0.95 + 1e-6


# ---------------------------------------------------------------------------
# 20.x.5 — UC once, warm LP resolves with fixed commitment
# ---------------------------------------------------------------------------

def test_commitment_fixed_session():
    from nexus_energy.mpc import commitment_fixed_session

    def build_uc():
        s = make_system(soc0_free=True)
        for g in s._generators:
            if g.name == "gas":
                g.committable = True
                g.p_min = 20.0
                g.startup_cost = 500.0
        return s

    uc_sys = build_uc()
    uc_res = uc_sys.optimise(verbose=False)
    assert uc_res.status == "optimal"
    u = uc_res.unit_status["gas"]
    assert u.shape == (T,)

    sess, base = commitment_fixed_session(build_uc(), uc_res)
    assert base.status == "optimal"
    gas = base.generator_dispatch["gas"]
    for t in range(T):
        if u[t] < 0.5:
            assert gas[t] <= 1e-6           # off stays off
        else:
            assert gas[t] >= 20.0 - 1e-6    # min stable output enforced

    # Hot resolve with new demand stays consistent with the schedule.
    res = sess.advance(demand={"site": 65 + 20 * np.sin(
        np.arange(T) / 24 * 2 * np.pi)})
    assert res.status == "optimal"
    gas2 = res.generator_dispatch["gas"]
    for t in range(T):
        if u[t] < 0.5:
            assert gas2[t] <= 1e-6
    assert sess.n_rebuilds == 0
