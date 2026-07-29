"""
EC109 -- Simple Cycle Gas Turbine -- F2a Brayton Cycle
Test suite: physics sanity, efficiency range, sweeps, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SimpleGasTurbine_F2a
from predict import ComponentModel

PASS = "\u2713"
FAIL = "\u2717"


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
def test_temperatures_physical():
    print("\n[Test 1] Brayton cycle: temperatures are physical")
    m, _ = make_model()
    r = m.brayton_cycle()
    assert_true(r["T2"] > r["T1"], f"T2={r['T2']:.0f} > T1={r['T1']:.0f}")
    assert_true(r["T3"] > r["T2"], f"T3={r['T3']:.0f} > T2={r['T2']:.0f}")
    assert_true(r["T4"] < r["T3"], f"T4={r['T4']:.0f} < T3={r['T3']:.0f}")
    assert_true(r["T4"] > r["T1"], f"T4={r['T4']:.0f} > T1={r['T1']:.0f} (exhaust hot)")


def test_electrical_efficiency_range():
    print("\n[Test 2] Electrical efficiency in range (25-42%)")
    m, _ = make_model()
    r = m.brayton_cycle()
    assert_true(0.25 < r["eta_electrical"] < 0.42,
                f"eta_elec={r['eta_electrical']:.3f} in (0.25, 0.42)")


def test_power_output_positive():
    print("\n[Test 3] Net power output is positive")
    m, _ = make_model()
    r = m.brayton_cycle()
    assert_true(r["W_elec_MW"] > 0, f"W_elec={r['W_elec_MW']:.1f} MW > 0")
    assert_true(r["W_turb_MW"] > r["W_comp_MW"],
                f"W_turb={r['W_turb_MW']:.1f} > W_comp={r['W_comp_MW']:.1f}")


def test_turbine_work_ratio():
    print("\n[Test 4] Back-work ratio (W_comp/W_turb) in 40-60%")
    m, _ = make_model()
    r = m.brayton_cycle()
    bwr = r["W_comp_MW"] / r["W_turb_MW"]
    assert_true(0.35 < bwr < 0.65, f"BWR={bwr:.3f} in (0.35, 0.65)")


def test_cp_temperature_dependence():
    print("\n[Test 5] cp_air increases with temperature")
    m, _ = make_model()
    cp_300 = m.cp_air(300.0)
    cp_600 = m.cp_air(600.0)
    cp_1000 = m.cp_air(1000.0)
    assert_true(cp_600 > cp_300, f"cp(600)={cp_600:.1f} > cp(300)={cp_300:.1f}")
    assert_true(cp_1000 > cp_600, f"cp(1000)={cp_1000:.1f} > cp(600)={cp_600:.1f}")


def test_PR_sweep_has_optimum():
    print("\n[Test 6] PR sweep: efficiency varies with PR")
    m, _ = make_model()
    PR_arr, results = m.sweep_pressure_ratio(np.arange(8, 36, 2))
    etas = [r["eta_electrical"] for r in results]
    assert_true(all(e > 0 for e in etas), "All efficiencies positive")
    assert_true(max(etas) - min(etas) > 0.02, f"Efficiency spread: {max(etas)-min(etas):.3f}")


def test_TIT_increases_efficiency():
    print("\n[Test 7] Higher TIT -> higher efficiency")
    m, _ = make_model()
    r_low = m.brayton_cycle(TIT=1173.15)
    r_high = m.brayton_cycle(TIT=1573.15)
    assert_true(r_high["eta_electrical"] > r_low["eta_electrical"],
                f"eta(TIT=1573)={r_high['eta_electrical']:.3f} > eta(TIT=1173)={r_low['eta_electrical']:.3f}")


def test_part_load():
    print("\n[Test 8] Part-load: power and efficiency decrease")
    m, _ = make_model()
    r_full = m.brayton_cycle()
    r_part = m.part_load(0.5)
    assert_true(r_part["W_elec_MW"] < r_full["W_elec_MW"],
                f"W_part={r_part['W_elec_MW']:.1f} < W_full={r_full['W_elec_MW']:.1f}")
    assert_true(r_part["eta_electrical"] < r_full["eta_electrical"],
                f"eta_part={r_part['eta_electrical']:.3f} < eta_full={r_full['eta_electrical']:.3f}")


def test_ambient_temperature_effect():
    print("\n[Test 9] Higher ambient -> lower power output")
    m, _ = make_model()
    r_cold = m.brayton_cycle(T_amb=268.15)  # -5 C
    r_hot = m.brayton_cycle(T_amb=313.15)   # 40 C
    assert_true(r_cold["W_elec_MW"] > r_hot["W_elec_MW"],
                f"W_cold={r_cold['W_elec_MW']:.1f} > W_hot={r_hot['W_elec_MW']:.1f}")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({})
    for key in ["W_elec_MW", "eta_electrical", "heat_rate_kJ_kWh", "T_exhaust_K", "SFC_kg_kWh"]:
        assert_true(key in r, f"Key '{key}' in output")


def test_benchmark():
    print("\n[Test 11] Benchmark: 1000 cycle evaluations")
    m, _ = make_model()
    t0 = time.perf_counter()
    for _ in range(1000):
        m.brayton_cycle()
    elapsed = time.perf_counter() - t0
    print(f"  1000 evaluations in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "1000 evals in < 5 s")


if __name__ == "__main__":
    tests = [
        test_temperatures_physical,
        test_electrical_efficiency_range,
        test_power_output_positive,
        test_turbine_work_ratio,
        test_cp_temperature_dependence,
        test_PR_sweep_has_optimum,
        test_TIT_increases_efficiency,
        test_part_load,
        test_ambient_temperature_effect,
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
    print(f"EC109 Gas Turbine F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
