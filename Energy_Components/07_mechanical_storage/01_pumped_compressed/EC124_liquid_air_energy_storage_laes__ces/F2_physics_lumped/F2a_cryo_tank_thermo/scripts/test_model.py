"""
EC124 -- Liquid Air Energy Storage (LAES / CES) -- F2a Cryo-Tank Thermo
Test suite: energy/mass conservation, RTE band, cold/hot recycle monotonicity,
boil-off physics, edge cases, predict() interface, benchmark timing.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import LAES_F2a
from predict import ComponentModel

PASS = "✓"
FAIL = "✗"


def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def make_model():
    cm = ComponentModel()
    return cm._model, cm


# ---------------------------------------------------------------------------
def test_rte_in_band():
    print("\n[Test 1] Round-trip efficiency in 50-60% band (cold recycle)")
    m, _ = make_model()
    r = m.round_trip(eps_cr=0.60)
    rte = r["eta_RT"]
    assert_true(0.50 <= rte <= 0.62, f"RTE={rte*100:.1f}% in [50, 62]%")
    assert_true(r["E_out_kWh"] < r["E_in_kWh"], "E_out < E_in (lossy, 2nd law)")


def test_rte_below_one():
    print("\n[Test 2] RTE < 1 across operating conditions (no free energy)")
    m, _ = make_model()
    for eps in [0.0, 0.3, 0.6, 0.9]:
        for dT in [0.0, 60.0, 200.0]:
            r = m.round_trip(eps_cr=eps, hot_recycle_dT_K=dT)
            assert_true(0.0 < r["eta_RT"] < 1.0,
                        f"eps={eps},dT={dT}: RTE={r['eta_RT']*100:.1f}% in (0,100)%")


def test_cold_recycle_raises_rte():
    print("\n[Test 3] Cold recycle monotonically raises RTE")
    m, _ = make_model()
    rtes = [m.round_trip(eps_cr=e)["eta_RT"] for e in [0.0, 0.2, 0.4, 0.6, 0.8]]
    for a, b in zip(rtes, rtes[1:]):
        assert_true(b > a, f"RTE rises with eps_cr: {b*100:.1f}% > {a*100:.1f}%")
    assert_true(rtes[-1] - rtes[0] > 0.10,
                f"Cold recycle gives >10pp gain ({(rtes[-1]-rtes[0])*100:.1f}pp)")


def test_hot_recycle_raises_rte():
    print("\n[Test 4] Waste-heat (hot) recycle raises RTE")
    m, _ = make_model()
    base = m.round_trip(eps_cr=0.60, hot_recycle_dT_K=0.0)["eta_RT"]
    hot = m.round_trip(eps_cr=0.60, hot_recycle_dT_K=150.0)["eta_RT"]
    assert_true(hot > base, f"Hot recycle: RTE {hot*100:.1f}% > {base*100:.1f}%")


def test_mass_conservation_charge():
    print("\n[Test 5] Charge mass balance: m_liq = integral(m_dot)")
    m, _ = make_model()
    md = 100.0
    dur = 5000.0  # s
    r = m.simulate("charge", dur, m_dot=md, m_liq0=0.0)
    expected = md * dur
    got = r["m_liq"][-1]
    assert_true(abs(got - expected) / expected < 1e-3,
                f"m_liq={got:.1f} kg ~ m_dot*t={expected:.1f} kg")


def test_mass_conservation_discharge():
    print("\n[Test 6] Discharge mass balance: liquid drains at m_dot")
    m, _ = make_model()
    md = 100.0
    m0 = 1.0e6
    dur = 4000.0
    r = m.simulate("discharge", dur, m_dot=md, m_liq0=m0)
    expected = m0 - md * dur
    got = r["m_liq"][-1]
    assert_true(abs(got - expected) / m0 < 1e-3,
                f"m_liq={got:.0f} kg ~ m0 - m_dot*t={expected:.0f} kg")


def test_boil_off_physics():
    print("\n[Test 7] Boil-off: heat leak drains tank, hotter = faster")
    m, _ = make_model()
    bor_cold = m.boil_off_per_day(280.0)
    bor_warm = m.boil_off_per_day(313.0)
    assert_true(bor_warm > bor_cold > 0, f"BOR(313K)={bor_warm*100:.3f}% > BOR(280K)={bor_cold*100:.3f}%")
    assert_true(0.001 < bor_cold < 0.02, f"BOR near ~0.5%/day ({bor_cold*100:.3f}%)")
    # tank loses mass over storage
    r = m.simulate("store", 24 * 3600.0, m_liq0=1.0e6, T_amb_K=298.15)
    assert_true(r["m_liq"][-1] < 1.0e6, "Liquid mass decreases during storage")


def test_store_no_heatleak_when_isothermal():
    print("\n[Test 8] No boil-off when ambient = storage temperature")
    m, _ = make_model()
    r = m.simulate("store", 24 * 3600.0, m_liq0=1.0e6, T_amb_K=m.T_storage)
    assert_true(abs(r["m_liq"][-1] - 1.0e6) < 1.0, "No boil-off at T_amb=T_storage")


def test_charge_power_positive():
    print("\n[Test 9] Charge draws power, discharge delivers power")
    m, _ = make_model()
    P_in = m.charge_power_kw(100.0)
    P_out = m.discharge_power_kw(100.0)
    assert_true(P_in > 0, f"Charge power={P_in/1e3:.1f} MW > 0")
    assert_true(P_out > 0, f"Discharge power={P_out/1e3:.1f} MW > 0")
    assert_true(P_out < P_in, "Discharge power < charge power for same flow (lossy)")


def test_soc_bounds_and_storage():
    print("\n[Test 10] SOC stays in [0,1]; tank cannot overfill or go negative")
    m, _ = make_model()
    # overfill attempt
    r = m.simulate("charge", 1e6, m_dot=200.0, m_liq0=m.m_tank_max * 0.99)
    assert_true(np.all(r["soc"] <= 1.0 + 1e-9), "SOC capped at 1.0")
    # over-drain attempt
    r2 = m.simulate("discharge", 1e6, m_dot=200.0, m_liq0=m.m_tank_max * 0.01)
    assert_true(np.all(r2["soc"] >= -1e-9), "SOC floored at 0.0")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"mode": "round_trip", "store_hours": 6.0,
                    "cold_recycle_eff": 0.60})
    for key in ["E_in_kWh", "E_out_kWh", "eta_RT", "boil_off_loss_kg",
                "w_liq_eff_kwh_per_kg", "w_exp_kwh_per_kg"]:
        assert_true(key in r, f"Key '{key}' in round_trip output")
    rs = cm.predict({"mode": "charge", "duration_s": 1000.0, "m_dot_kgs": 50.0})
    for key in ["t", "m_liq", "soc", "Q_cold", "power_kW", "energy_kWh"]:
        assert_true(key in rs, f"Key '{key}' in single-mode output")
    assert_true(len(rs["t"]) == len(rs["m_liq"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC124", "component_id == EC124")


def test_benchmark():
    print("\n[Test 12] Benchmark: full round-trip with storage")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.round_trip(store_hours=12.0, eps_cr=0.60, hot_recycle_dT_K=100.0)
    elapsed = time.perf_counter() - t0
    print(f"  round_trip in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_rte_in_band,
        test_rte_below_one,
        test_cold_recycle_raises_rte,
        test_hot_recycle_raises_rte,
        test_mass_conservation_charge,
        test_mass_conservation_discharge,
        test_boil_off_physics,
        test_store_no_heatleak_when_isothermal,
        test_charge_power_positive,
        test_soc_bounds_and_storage,
        test_predict_interface,
        test_benchmark,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except (AssertionError, Exception) as e:
            failed += 1
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"EC124 LAES F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
