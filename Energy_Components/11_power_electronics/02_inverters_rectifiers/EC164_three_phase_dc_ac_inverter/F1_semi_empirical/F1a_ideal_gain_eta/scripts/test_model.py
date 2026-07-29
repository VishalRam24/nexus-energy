"""
EC164 — Three-Phase DC-AC Inverter — F1a Ideal Gain + Efficiency
Test suite: physics sanity checks, edge cases, benchmark.
"""

import sys
import os
import time
import math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import ThreePhaseInverterModel
from predict import ComponentModel

DEFAULT_PARAMS = {
    "V_dc_rated": 800.0,
    "P_rated":    100000.0,
    "eta_rated":  0.98,
    "k1":         0.02,
    "k2":         0.005,
    "f_sw":       10000.0,
}

PASS = "\u2713"
FAIL = "\u2717"


def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def make_model():
    return ThreePhaseInverterModel(DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Test 1 — AC voltage gain model
# ---------------------------------------------------------------------------

def test_ac_voltage_gain():
    print("\n[Test 1] AC voltage gain (SVPWM)")
    m_mdl = make_model()
    SQRT2 = math.sqrt(2.0)

    # m=1 -> V_ac_rms = V_dc/sqrt(2)
    V_ac = m_mdl.ac_rms_voltage(800.0, 1.0)
    expected = 800.0 / SQRT2
    assert_true(abs(V_ac - expected) < 1e-9,
                f"V_ac_rms = V_dc/sqrt(2) at m=1: {V_ac:.4f} V")

    # m=0 -> V_ac = 0
    V_ac_0 = m_mdl.ac_rms_voltage(800.0, 0.0)
    assert_true(V_ac_0 == 0.0, "V_ac_rms = 0 at m=0")

    # Proportional to m
    V1 = m_mdl.ac_rms_voltage(800.0, 0.5)
    V2 = m_mdl.ac_rms_voltage(800.0, 1.0)
    assert_true(abs(V2 / V1 - 2.0) < 1e-9, "V_ac proportional to m")


# ---------------------------------------------------------------------------
# Test 2 — V_ac < V_dc for all valid m
# ---------------------------------------------------------------------------

def test_vac_below_vdc():
    print("\n[Test 2] V_ac_rms < V_dc for all m in [0,1]")
    m_mdl = make_model()
    V_dc = 800.0
    for m in np.linspace(0.0, 1.0, 21):
        V_ac = m_mdl.ac_rms_voltage(V_dc, m)
        assert_true(V_ac <= V_dc,
                    f"V_ac ({V_ac:.2f}) <= V_dc ({V_dc}) at m={m:.2f}")


# ---------------------------------------------------------------------------
# Test 3 — Efficiency at rated load equals eta_rated
# ---------------------------------------------------------------------------

def test_eta_at_rated():
    print("\n[Test 3] Efficiency = eta_rated at full load")
    m_mdl = make_model()
    eta = m_mdl.efficiency(DEFAULT_PARAMS["P_rated"])
    assert_true(abs(eta - DEFAULT_PARAMS["eta_rated"]) < 1e-9,
                f"eta={eta:.6f} == eta_rated at PLR=1")


# ---------------------------------------------------------------------------
# Test 4 — Efficiency drops at low load
# ---------------------------------------------------------------------------

def test_eta_drops_low_load():
    print("\n[Test 4] Efficiency decreases at low part-load ratio")
    m_mdl = make_model()
    eta_full = m_mdl.efficiency(DEFAULT_PARAMS["P_rated"])
    eta_10   = m_mdl.efficiency(0.10 * DEFAULT_PARAMS["P_rated"])
    assert_true(eta_10 < eta_full,
                f"eta(10% load)={eta_10:.4f} < eta(100% load)={eta_full:.4f}")


# ---------------------------------------------------------------------------
# Test 5 — Efficiency <= 1.0 at all loads
# ---------------------------------------------------------------------------

def test_eta_below_unity():
    print("\n[Test 5] Efficiency <= 1.0 for all loads")
    m_mdl = make_model()
    for plr in np.linspace(0.01, 1.0, 50):
        eta = m_mdl.efficiency(plr * DEFAULT_PARAMS["P_rated"])
        assert_true(eta <= 1.0, f"eta={eta:.4f} <= 1.0 at PLR={plr:.2f}")


# ---------------------------------------------------------------------------
# Test 6 — P_loss > 0 for p_load > 0
# ---------------------------------------------------------------------------

def test_ploss_positive():
    print("\n[Test 6] Power loss > 0 for all p_load > 0")
    m_mdl = make_model()
    for plr in [0.1, 0.5, 1.0]:
        p_loss = m_mdl.power_loss(plr * DEFAULT_PARAMS["P_rated"])
        assert_true(p_loss > 0,
                    f"P_loss={p_loss:.2f} W > 0 at PLR={plr}")


# ---------------------------------------------------------------------------
# Test 7 — P_in > P_out (conservation)
# ---------------------------------------------------------------------------

def test_power_conservation():
    print("\n[Test 7] P_in > P_out (losses > 0)")
    m_mdl = make_model()
    for plr in [0.25, 0.5, 1.0]:
        p_load = plr * DEFAULT_PARAMS["P_rated"]
        p_in   = m_mdl.power_input(p_load)
        p_loss = m_mdl.power_loss(p_load)
        assert_true(abs((p_in - p_load) - p_loss) < 1e-6,
                    f"P_in - P_out == P_loss at PLR={plr}")


# ---------------------------------------------------------------------------
# Test 8 — Current proportional to load at fixed V_ac
# ---------------------------------------------------------------------------

def test_current_proportional():
    print("\n[Test 8] AC current proportional to load at fixed m, V_dc")
    m_mdl = make_model()
    I1 = m_mdl.ac_rms_current(800.0, 0.9, 50000.0)
    I2 = m_mdl.ac_rms_current(800.0, 0.9, 100000.0)
    assert_true(abs(I2 / I1 - 2.0) < 1e-9,
                f"I_ac proportional to P_load (ratio={I2/I1:.10f})")


# ---------------------------------------------------------------------------
# Test 9 — ComponentModel predict() interface
# ---------------------------------------------------------------------------

def test_predict_interface():
    print("\n[Test 9] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"v_dc": 800.0, "p_load": 80000.0, "modulation_index": 0.9})
    required = ["v_ac_rms_V", "i_ac_rms_A", "efficiency", "p_in_W", "p_loss_W", "PLR"]
    for k in required:
        assert_true(k in out, f"Output key '{k}' present")
    assert_true(out["v_ac_rms_V"] < 800.0, "V_ac < V_dc")
    assert_true(out["efficiency"] <= 1.0,  "efficiency <= 1")
    assert_true(out["p_loss_W"] > 0,       "P_loss > 0")


# ---------------------------------------------------------------------------
# Test 10 — get_info() completeness
# ---------------------------------------------------------------------------

def test_get_info():
    print("\n[Test 10] get_info() completeness")
    cm = ComponentModel()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "inputs", "outputs", "source"]:
        assert_true(k in info, f"'{k}' in get_info()")


# ---------------------------------------------------------------------------
# Test 11 — Benchmark
# ---------------------------------------------------------------------------

def test_benchmark():
    print("\n[Test 11] Benchmark: 10,000 evaluate() calls")
    m_mdl = make_model()
    p_vals = np.linspace(1000.0, 100000.0, 10000)
    t0 = time.perf_counter()
    for p in p_vals:
        m_mdl.evaluate(800.0, float(p), 0.9)
    elapsed = time.perf_counter() - t0
    print(f"  10,000 evaluations in {elapsed*1000:.1f} ms  ({elapsed/10000*1e6:.2f} µs/call)")
    assert_true(elapsed < 5.0, "10,000 calls complete in < 5 s")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_ac_voltage_gain,
        test_vac_below_vdc,
        test_eta_at_rated,
        test_eta_drops_low_load,
        test_eta_below_unity,
        test_ploss_positive,
        test_power_conservation,
        test_current_proportional,
        test_predict_interface,
        test_get_info,
        test_benchmark,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  ASSERTION ERROR: {e}")
        except Exception as e:
            failed += 1
            print(f"  UNEXPECTED ERROR in {t.__name__}: {e}")

    print(f"\n{'='*50}")
    print(f"EC164 Three-Phase Inverter — Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)
