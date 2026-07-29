"""
EC043 -- Hybrid Supercapacitor (Lithium-Ion Capacitor) -- F2a Asymmetric Hybrid
Test suite: physics sanity (charge conservation, monotonicity, voltage window,
energy bounds, thermal balance), edge cases, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import HybridSupercapacitorF2a
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
def test_ocv_window():
    print("\n[Test 1] OCV stays inside the 2.2-3.8 V sloping window")
    m, _ = make_model()
    assert_true(abs(m.ocv(0.0) - m.V_min) < 1e-6, f"OCV(empty)={m.ocv(0.0):.3f} = V_min")
    assert_true(abs(m.ocv(m.Q_max) - m.V_max) < 1e-6, f"OCV(full)={m.ocv(m.Q_max):.3f} = V_max")
    for q in np.linspace(0, m.Q_max, 25):
        v = m.ocv(q)
        assert_true(m.V_min - 1e-9 <= v <= m.V_max + 1e-9, f"OCV={v:.3f} in [2.2,3.8]")


def test_ocv_monotone_sloping():
    print("\n[Test 2] OCV monotonically increases with charge (sloping, not flat)")
    m, _ = make_model()
    qs = np.linspace(0, m.Q_max, 200)
    v = np.array([m.ocv(q) for q in qs])
    assert_true(np.all(np.diff(v) >= -1e-9), "OCV non-decreasing in q")
    # genuinely sloping (not an EDLC-flat or battery-flat): spread > 1 V
    assert_true((v.max() - v.min()) > 1.0, f"voltage swing {v.max()-v.min():.2f} V > 1 V (sloping)")


def test_faradaic_and_edlc_branches():
    print("\n[Test 3] Both faradaic + EDLC branches contribute")
    m, _ = make_model()
    # faradaic fraction f_far in (0,1) => neither pure-EDLC linear nor pure-battery
    assert_true(0.0 < m.f_far < 1.0, f"f_far={m.f_far} in (0,1)")
    # off mid-charge, OCV deviates from a pure linear EDLC interpolation
    # (the symmetric logit faradaic branch crosses the linear line only at s=0.5)
    s = 0.25
    q = s * m.Q_max
    v_lin = m.V_min + (m.V_max - m.V_min) * s
    assert_true(abs(m.ocv(q) - v_lin) > 1e-3, "faradaic branch bends OCV away from linear EDLC")


def test_charge_conservation():
    print("\n[Test 4] Charge conservation: dq = -I*dt (no leakage limit)")
    m, _ = make_model()
    I = 80.0  # discharge
    q0 = 0.7 * m.Q_max
    dur = 10.0
    r = m.simulate(I, q0, 298.15, 0.1, dur)
    dq_actual = r["charge"][-1] - r["charge"][0]
    # leakage is tiny; charge removed ~ I*dt within a few percent
    dq_expected = -I * dur
    assert_true(abs(dq_actual - dq_expected) / abs(dq_expected) < 0.02,
                f"dQ={dq_actual:.1f} C ~ -I*t={dq_expected:.1f} C (within 2%)")


def test_charge_discharge_symmetry():
    print("\n[Test 5] Charging raises SOC, discharging lowers it")
    m, _ = make_model()
    r_ch = m.simulate(-60.0, 0.4 * m.Q_max, 298.15, 0.5, 20.0)
    r_dis = m.simulate(60.0, 0.6 * m.Q_max, 298.15, 0.5, 20.0)
    assert_true(r_ch["soc"][-1] > r_ch["soc"][0], "charge: SOC rises")
    assert_true(r_dis["soc"][-1] < r_dis["soc"][0], "discharge: SOC falls")


def test_esr_drop():
    print("\n[Test 6] Terminal voltage drops below OCV on discharge (IR drop)")
    m, _ = make_model()
    q = 0.6 * m.Q_max
    v_oc = m.ocv(q)
    v_dis = m.terminal_voltage(q, 200.0, 298.15)   # discharge
    v_chg = m.terminal_voltage(q, -200.0, 298.15)  # charge
    assert_true(v_dis < v_oc, f"V_dis={v_dis:.4f} < OCV={v_oc:.4f}")
    assert_true(v_chg > v_oc, f"V_chg={v_chg:.4f} > OCV={v_oc:.4f}")


def test_energy_bounds():
    print("\n[Test 7] Stored energy monotone in q and bounded by V_max*Q_max")
    m, _ = make_model()
    e_half = m.stored_energy(0.5 * m.Q_max)
    e_full = m.stored_energy(m.Q_max)
    assert_true(0.0 < e_half < e_full, f"E(half)={e_half:.0f} < E(full)={e_full:.0f} J")
    # energy cannot exceed V_max * Q_max (loose thermodynamic ceiling)
    ceiling = m.V_max * m.Q_max
    assert_true(e_full < ceiling, f"E(full)={e_full:.0f} J < V_max*Q_max={ceiling:.0f} J")
    # and must exceed V_min * Q_max (floor, since OCV >= V_min)
    floor = m.V_min * m.Q_max
    assert_true(e_full > floor, f"E(full)={e_full:.0f} J > V_min*Q_max={floor:.0f} J")


def test_efficiency_range():
    print("\n[Test 8] Voltaic efficiency strictly in (0,1) under load")
    m, _ = make_model()
    r = m.simulate(150.0, 0.7 * m.Q_max, 298.15, 0.5, 15.0)
    for eta in r["efficiency"]:
        assert_true(0.0 < eta < 1.0, f"eta={eta:.4f} in (0,1)")


def test_thermal_balance():
    print("\n[Test 9] Thermal ODE: Joule heating warms cell, then balances")
    m, _ = make_model()
    # high current burst heats it up
    r = m.simulate(400.0, 0.8 * m.Q_max, 298.15, 0.1, 8.0)
    assert_true(r["temperature"][-1] > r["temperature"][0], "cell heats up under load")
    # at zero current + at ambient, temperature stays put (balance)
    r0 = m.simulate(0.0, 0.5 * m.Q_max, 298.15, 1.0, 30.0)
    assert_true(abs(r0["temperature"][-1] - 298.15) < 0.5,
                f"no load @ T_amb: T stays ~298.15 K (got {r0['temperature'][-1]:.3f})")


def test_leakage_self_discharge():
    print("\n[Test 10] Leakage slowly discharges an open cell (I=0)")
    m, _ = make_model()
    r = m.simulate(0.0, 0.9 * m.Q_max, 298.15, 10.0, 3000.0)
    assert_true(r["charge"][-1] < r["charge"][0], "open-circuit charge decays via R_leak")
    assert_true(r["charge"][-1] > 0.0, "but does not fully drain in 3000 s")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"current_A": 50.0, "dt": 1.0, "duration_s": 5.0})
    for key in ["t", "charge", "soc", "v_oc", "v_terminal", "power",
                "efficiency", "temperature", "energy_J"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["v_terminal"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC043", "get_info reports EC043")


def test_benchmark():
    print("\n[Test 12] Benchmark: 60 s sim at dt=0.1")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(100.0, 0.7 * m.Q_max, 298.15, 0.1, 60.0)
    elapsed = time.perf_counter() - t0
    print(f"  60 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_ocv_window,
        test_ocv_monotone_sloping,
        test_faradaic_and_edlc_branches,
        test_charge_conservation,
        test_charge_discharge_symmetry,
        test_esr_drop,
        test_energy_bounds,
        test_efficiency_range,
        test_thermal_balance,
        test_leakage_self_discharge,
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
    print(f"EC043 Hybrid Supercapacitor F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
