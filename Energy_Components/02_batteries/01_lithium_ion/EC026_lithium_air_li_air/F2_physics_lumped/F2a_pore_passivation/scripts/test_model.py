"""
EC026 -- Lithium-Air (Li-O2) -- F2a Physics-Lumped (Pore Passivation)
Test suite: Coulomb conservation, V_dis < E_eq < V_chg hysteresis, pore-saturation
capacity cutoff, efficiency bounds, thermal ODE, predict() interface, benchmark.
Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import LiAirF2a
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
def test_equilibrium_range():
    print("\n[Test 1] Equilibrium potential near Li2O2 plateau (~2.96 V)")
    m, _ = make_model()
    E = m.equilibrium_voltage(0.5, 298.15)
    assert_true(2.5 < E < 3.3, f"E_eq(0.5,298)={E:.4f} V near 2.96 plateau")
    # entropic coefficient is negative -> hotter cell has lower E_eq
    E_hot = m.equilibrium_voltage(0.5, 333.15)
    assert_true(E_hot < E, f"dOCV/dT<0: E_eq(hot)={E_hot:.4f} < {E:.4f}")


def test_hysteresis_gap():
    print("\n[Test 2] V_discharge < E_eq < V_charge (~1 V round-trip gap)")
    m, _ = make_model()
    E = m.equilibrium_voltage(0.5, 298.15)
    Vd, Vc, gap = m.round_trip_voltage_gap(1.0, soc=0.5, theta=0.1, T=298.15)
    assert_true(Vd < E, f"V_dis={Vd:.3f} < E_eq={E:.3f}")
    assert_true(Vc > E, f"V_chg={Vc:.3f} > E_eq={E:.3f}")
    assert_true(0.5 < gap < 1.6, f"round-trip gap={gap:.3f} V (~1 V Li-air hysteresis)")


def test_oer_worse_than_orr():
    print("\n[Test 3] OER (charge) overpotential >> ORR (discharge)")
    m, _ = make_model()
    eta_dis = m.kinetic_overpotential(+1.0, 298.15)   # ORR
    eta_chg = m.kinetic_overpotential(-1.0, 298.15)   # OER
    assert_true(eta_chg > eta_dis, f"eta_OER={eta_chg:.3f} > eta_ORR={eta_dis:.3f}")


def test_coulomb_conservation():
    print("\n[Test 4] Coulomb conservation: dSOC matches charge passed")
    m, _ = make_model()
    I = 0.5
    dur = 1800.0
    r = m.simulate(I, soc0=1.0, theta0=0.0, dt=30.0, duration_s=dur)
    # only count while current was actually flowing (not gated off)
    flowed = r["current"] > 0
    charge_Ah = np.trapz(r["current"], r["t"]) / 3600.0
    dsoc = r["soc"][0] - r["soc"][-1]
    expected_dsoc = charge_Ah / m.capacity_ref
    assert_true(abs(dsoc - expected_dsoc) < 1e-2,
                f"dSOC={dsoc:.4f} ~ Q/C={expected_dsoc:.4f}")


def test_pore_fill_tracks_charge():
    print("\n[Test 5] Pore fill theta tied to same charge throughput")
    m, _ = make_model()
    r = m.simulate(0.5, soc0=1.0, theta0=0.0, dt=30.0, duration_s=1800.0)
    charge_Ah = np.trapz(r["current"], r["t"]) / 3600.0
    dtheta = r["theta"][-1] - r["theta"][0]
    expected = charge_Ah / m.Q_pore_max
    assert_true(abs(dtheta - expected) < 1e-2,
                f"dtheta={dtheta:.4f} ~ Q/Q_pore={expected:.4f}")
    assert_true(np.all(np.diff(r["theta"]) >= -1e-9), "theta monotonically increases on discharge")


def test_pore_saturation_cutoff():
    print("\n[Test 6] Capacity cut off at pore saturation (sudden death)")
    m, _ = make_model()
    # full deep discharge of a fresh cell
    r = m.simulate(1.0, soc0=1.0, theta0=0.0, dt=30.0, duration_s=4000.0)
    assert_true(r["theta"][-1] <= m.pore_cutoff + 1e-6,
                f"theta capped at cutoff {m.pore_cutoff}: theta_end={r['theta'][-1]:.4f}")
    # current must shut off once cutoff reached (gated to 0)
    assert_true(np.isclose(r["current"][-1], 0.0),
                "current gated to 0 after pore saturation")
    # while still loaded, voltage collapses toward the cutoff (passivation sag)
    loaded = r["current"] > 0
    v_min_loaded = r["voltage"][loaded].min()
    assert_true(v_min_loaded < r["voltage"][0] - 0.1,
                f"loaded V sags before death: Vmin={v_min_loaded:.3f} < V0={r['voltage'][0]:.3f}")


def test_passivation_grows():
    print("\n[Test 7] Passivation overpotential grows and diverges near cutoff")
    m, _ = make_model()
    e_lo = m.passivation_overpotential(0.1, current=1.0)
    e_hi = m.passivation_overpotential(0.85, current=1.0)
    assert_true(e_hi > e_lo * 3, f"eta_pass(0.85)={e_hi:.3f} >> eta_pass(0.1)={e_lo:.3f}")
    # zero on charge (film being stripped)
    assert_true(m.passivation_overpotential(0.5, current=-1.0) == 0.0,
                "no passivation penalty during charge")


def test_efficiency_bounds():
    print("\n[Test 8] Voltaic efficiency in (0, 1)")
    m, _ = make_model()
    rd = m.simulate(0.5, soc0=0.9, theta0=0.05, dt=30.0, duration_s=600.0)
    for eta in rd["efficiency"]:
        assert_true(0.0 < eta < 1.0, f"discharge eta={eta:.4f} in (0,1)")
    rc = m.simulate(-0.5, soc0=0.2, theta0=0.5, dt=30.0, duration_s=600.0)
    for eta in rc["efficiency"]:
        assert_true(0.0 < eta < 1.0, f"charge eta={eta:.4f} in (0,1)")


def test_thermal_ode_heats():
    print("\n[Test 9] Thermal ODE: cell self-heats under discharge, bounded")
    m, _ = make_model()
    r = m.simulate(1.5, soc0=1.0, theta0=0.0, T0=298.15, dt=20.0, duration_s=1200.0)
    assert_true(r["temperature"].max() > 298.15, "cell heats above ambient")
    assert_true(r["temperature"].max() < 360.0, f"T bounded: Tmax={r['temperature'].max():.2f} K")


def test_charge_clears_pores():
    print("\n[Test 10] Charging removes Li2O2 (theta decreases, V > E_eq)")
    m, _ = make_model()
    r = m.simulate(-0.5, soc0=0.3, theta0=0.6, dt=30.0, duration_s=1200.0)
    assert_true(r["theta"][-1] < r["theta"][0], f"theta {r['theta'][0]:.3f}->{r['theta'][-1]:.3f}")
    mid = len(r["t"]) // 2
    assert_true(r["voltage"][mid] > r["equilibrium_voltage"][mid],
                "charge voltage above equilibrium")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"current_A": 0.5, "dt": 30.0, "duration_s": 300.0})
    for key in ["t", "voltage", "current", "soc", "theta",
                "equilibrium_voltage", "power", "efficiency",
                "temperature", "overpotentials"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["voltage"]) == len(r["theta"]),
                "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC026", "get_info id == EC026")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h discharge sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(1.0, soc0=1.0, theta0=0.0, dt=10.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_equilibrium_range,
        test_hysteresis_gap,
        test_oer_worse_than_orr,
        test_coulomb_conservation,
        test_pore_fill_tracks_charge,
        test_pore_saturation_cutoff,
        test_passivation_grows,
        test_efficiency_bounds,
        test_thermal_ode_heats,
        test_charge_clears_pores,
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
    print(f"EC026 Li-Air F2a (pore passivation) -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
