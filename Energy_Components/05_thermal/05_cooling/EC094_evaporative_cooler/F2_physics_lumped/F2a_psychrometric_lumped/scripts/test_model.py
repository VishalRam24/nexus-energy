"""
EC094 -- Evaporative Cooler -- F2a Psychrometric (Physics-Lumped)
Test suite: psychrometric sanity, conservation, ODE convergence, edge cases.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import EvaporativeCooler_F2a
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
def test_effectiveness_range():
    print("\n[Test 1] Saturation effectiveness in (0, 1)")
    m, _ = make_model()
    for ntu in [0.1, 0.5, 1.897, 3.0, 6.0]:
        eps = m.saturation_effectiveness(ntu)
        assert_true(0.0 < eps < 1.0, f"NTU={ntu} -> eps={eps:.4f} in (0,1)")


def test_wetbulb_below_drybulb():
    print("\n[Test 2] Wet-bulb <= dry-bulb; equal at saturation")
    m, _ = make_model()
    for T, RH in [(35, 0.2), (25, 0.5), (40, 0.1), (15, 0.8)]:
        Twb = m.wet_bulb(T, RH)
        assert_true(Twb <= T + 1e-6, f"T_wb({T},{RH})={Twb:.2f} <= T_db={T}")
    # at RH=100% wet-bulb ~ dry-bulb
    Twb_sat = m.wet_bulb(30, 1.0)
    assert_true(abs(Twb_sat - 30) < 1.5, f"At RH=1, T_wb={Twb_sat:.2f} ~ T_db=30")


def test_tout_between_wb_and_db():
    print("\n[Test 3] T_out bounded by [T_wb, T_db]")
    m, _ = make_model()
    for T, RH in [(38, 0.2), (30, 0.4), (45, 0.15)]:
        ss = m.steady_state(T, RH, 1.0)
        assert_true(ss["T_wb"] - 1e-6 <= ss["T_out"] <= T + 1e-6,
                    f"T_wb={ss['T_wb']:.2f} <= T_out={ss['T_out']:.2f} <= T_db={T}")


def test_energy_conservation():
    print("\n[Test 4] Sensible cooling ~ latent heat absorbed (adiabatic saturation)")
    m, _ = make_model()
    ss = m.steady_state(38.0, 0.20, 1.0)
    assert_true(ss["energy_residual"] < 0.06,
                f"|Q_sens-Q_latent|/Q_sens = {ss['energy_residual']*100:.2f}% (< 6%)")


def test_water_mass_balance():
    print("\n[Test 5] Water consumption = m_dry*(w_out-w_in) > 0 and scales with flow")
    m, _ = make_model()
    ss1 = m.steady_state(38.0, 0.20, 1.0)
    ss2 = m.steady_state(38.0, 0.20, 2.0)
    assert_true(ss1["m_dot_water_kg_s"] > 0, f"water={ss1['m_dot_water_kg_s']*1000:.3f} g/s > 0")
    ratio = ss2["m_dot_water_kg_s"] / ss1["m_dot_water_kg_s"]
    assert_true(1.9 < ratio < 2.1, f"Doubling air flow ~doubles water (ratio={ratio:.3f})")


def test_humidity_increases():
    print("\n[Test 6] Outlet humidity ratio rises, RH_out > RH_in")
    m, _ = make_model()
    ss = m.steady_state(38.0, 0.20, 1.0)
    assert_true(ss["w_out"] > ss["w_in"], f"w_out={ss['w_out']:.5f} > w_in={ss['w_in']:.5f}")
    assert_true(ss["RH_out"] > 0.20, f"RH_out={ss['RH_out']:.3f} > 0.20")


def test_drier_air_cools_more():
    print("\n[Test 7] Drier inlet air -> more cooling (lower T_out)")
    m, _ = make_model()
    ss_dry = m.steady_state(38.0, 0.10, 1.0)
    ss_humid = m.steady_state(38.0, 0.60, 1.0)
    assert_true(ss_dry["T_out"] < ss_humid["T_out"],
                f"T_out(10%RH)={ss_dry['T_out']:.2f} < T_out(60%RH)={ss_humid['T_out']:.2f}")


def test_transient_relaxes_and_cools():
    print("\n[Test 8] ODE transient: pad cools from dry start toward wet-bulb approach")
    m, _ = make_model()
    r = m.simulate(38.0, 0.20, 1.0, T_pad0=38.0, dt=2.0, duration_s=300.0)
    assert_true(r["success"], "solve_ivp succeeded")
    assert_true(r["T_pad"][-1] < r["T_pad"][0],
                f"Pad cooled: T_pad {r['T_pad'][0]:.2f} -> {r['T_pad'][-1]:.2f} C")
    Twb = r["T_wb"][-1]
    assert_true(r["T_pad"][-1] >= Twb - 0.5,
                f"Pad final {r['T_pad'][-1]:.2f}C >= T_wb {Twb:.2f}C (cannot subcool past wet-bulb)")


def test_transient_steady_state():
    print("\n[Test 9] Transient reaches steady state (dT -> 0)")
    m, _ = make_model()
    r = m.simulate(35.0, 0.30, 1.0, dt=2.0, duration_s=600.0)
    dT = abs(r["T_pad"][-1] - r["T_pad"][-2])
    assert_true(dT < 0.05, f"Near SS: dT={dT:.5f} C between last two steps")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"T_db_C": 36.0, "RH": 0.25, "dt": 2.0, "duration_s": 60.0})
    for key in ["t", "T_db", "T_wb", "T_pad", "T_out", "Q_sens_W", "COP", "steady_state"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_out"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC094", "get_info id == EC094")


def test_step_response():
    print("\n[Test 11] Inlet step-up in T_db raises T_out")
    m, _ = make_model()
    def step_T(t):
        return 30.0 if t < 150 else 42.0
    r = m.simulate(step_T, 0.20, 1.0, dt=2.0, duration_s=300.0)
    i_before = np.argmin(np.abs(r["t"] - 148.0))
    i_after = np.argmin(np.abs(r["t"] - 290.0))
    assert_true(r["T_out"][i_after] > r["T_out"][i_before],
                f"T_out rises after inlet step: {r['T_out'][i_before]:.2f} -> {r['T_out'][i_after]:.2f}")


def test_benchmark():
    print("\n[Test 12] Benchmark: 300s sim at dt=1")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(38.0, 0.20, 1.0, dt=1.0, duration_s=300.0)
    elapsed = time.perf_counter() - t0
    print(f"  300s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_effectiveness_range,
        test_wetbulb_below_drybulb,
        test_tout_between_wb_and_db,
        test_energy_conservation,
        test_water_mass_balance,
        test_humidity_increases,
        test_drier_air_cools_more,
        test_transient_relaxes_and_cools,
        test_transient_steady_state,
        test_predict_interface,
        test_step_response,
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
    print(f"EC094 Evaporative Cooler F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
