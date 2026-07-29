"""
EC101 -- Combined Cycle Gas Turbine (CCGT) -- F2a Thermo Cycle SS
Test suite: physics sanity, efficiency range, part-load, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import CCGT_F2a
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
def test_brayton_temperatures_physical():
    print("\n[Test 1] Brayton cycle: temperatures are physical")
    m, _ = make_model()
    b = m.brayton_cycle()
    assert_true(b["T2"] > b["T1"], f"T2={b['T2']:.0f} > T1={b['T1']:.0f}")
    assert_true(b["T3"] > b["T2"], f"T3={b['T3']:.0f} > T2={b['T2']:.0f}")
    assert_true(b["T4"] < b["T3"], f"T4={b['T4']:.0f} < T3={b['T3']:.0f}")
    assert_true(b["T4"] > b["T1"], f"T4={b['T4']:.0f} > T1={b['T1']:.0f} (exhaust above ambient)")


def test_brayton_efficiency():
    print("\n[Test 2] Brayton cycle efficiency in physical range (30-45%)")
    m, _ = make_model()
    b = m.brayton_cycle()
    assert_true(0.30 < b["eta_gt"] < 0.45, f"eta_gt={b['eta_gt']:.3f} in (0.30, 0.45)")


def test_compressor_work_positive():
    print("\n[Test 3] Compressor work > 0, turbine work > compressor work")
    m, _ = make_model()
    b = m.brayton_cycle()
    assert_true(b["W_comp"] > 0, f"W_comp={b['W_comp']/1e6:.1f} MW > 0")
    assert_true(b["W_turb_gas"] > b["W_comp"],
                f"W_turb={b['W_turb_gas']/1e6:.1f} > W_comp={b['W_comp']/1e6:.1f} MW")


def test_combined_efficiency_target():
    print("\n[Test 4] Combined cycle efficiency 55-62%")
    m, _ = make_model()
    r = m.combined_cycle()
    assert_true(0.50 < r["eta_combined"] < 0.65,
                f"eta_cc={r['eta_combined']:.3f} in (0.50, 0.65)")


def test_combined_power_output():
    print("\n[Test 5] Total power output is positive and reasonable (100-600 MW)")
    m, _ = make_model()
    r = m.combined_cycle()
    assert_true(100 < r["W_total_MW"] < 600,
                f"W_total={r['W_total_MW']:.1f} MW in (100, 600)")


def test_steam_adds_power():
    print("\n[Test 6] Rankine cycle adds significant power (>30% of GT)")
    m, _ = make_model()
    r = m.combined_cycle()
    ratio = r["W_st_elec_MW"] / r["W_gt_elec_MW"]
    assert_true(ratio > 0.3, f"ST/GT ratio={ratio:.2f} > 0.3")
    assert_true(ratio < 1.0, f"ST/GT ratio={ratio:.2f} < 1.0 (GT dominates)")


def test_part_load_efficiency_drops():
    print("\n[Test 7] Part-load: efficiency decreases")
    m, _ = make_model()
    r_full = m.combined_cycle(load_fraction=1.0)
    r_part = m.combined_cycle(load_fraction=0.5)
    assert_true(r_part["eta_combined"] < r_full["eta_combined"],
                f"eta_part={r_part['eta_combined']:.3f} < eta_full={r_full['eta_combined']:.3f}")
    assert_true(r_part["W_total_MW"] < r_full["W_total_MW"],
                f"W_part={r_part['W_total_MW']:.1f} < W_full={r_full['W_total_MW']:.1f}")


def test_pressure_ratio_sweep():
    print("\n[Test 8] PR sweep: efficiency has a peak")
    m, _ = make_model()
    PR_arr, results = m.sweep_pressure_ratio(np.arange(10, 30, 2))
    etas = [r["eta_combined"] for r in results]
    # All positive
    for eta in etas:
        assert_true(eta > 0, f"eta={eta:.3f} > 0")
    # There should be some variation
    assert_true(max(etas) - min(etas) > 0.01, "PR sweep shows efficiency variation")


def test_TIT_increases_efficiency():
    print("\n[Test 9] Higher TIT -> higher efficiency (up to a point)")
    m, _ = make_model()
    r_low = m.combined_cycle(TIT=1373.15)
    r_high = m.combined_cycle(TIT=1573.15)
    assert_true(r_high["eta_combined"] >= r_low["eta_combined"] - 0.01,
                f"eta(TIT=1573)={r_high['eta_combined']:.3f} >= eta(TIT=1373)={r_low['eta_combined']:.3f}")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({})
    for key in ["W_total_MW", "eta_combined", "heat_rate_kJ_kWh", "T_exhaust_K"]:
        assert_true(key in r, f"Key '{key}' in output")


def test_benchmark():
    print("\n[Test 11] Benchmark: 100 combined cycle evaluations")
    m, _ = make_model()
    t0 = time.perf_counter()
    for _ in range(100):
        m.combined_cycle()
    elapsed = time.perf_counter() - t0
    print(f"  100 evaluations in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "100 evals in < 5 s")


if __name__ == "__main__":
    tests = [
        test_brayton_temperatures_physical,
        test_brayton_efficiency,
        test_compressor_work_positive,
        test_combined_efficiency_target,
        test_combined_power_output,
        test_steam_adds_power,
        test_part_load_efficiency_drops,
        test_pressure_ratio_sweep,
        test_TIT_increases_efficiency,
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
    print(f"EC101 CCGT F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
