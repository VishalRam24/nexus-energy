"""
EC142 -- Biogas Upgrading to Biomethane -- F2a Water Scrubbing
Test suite: mass conservation, purity/recovery/slip physics, Henry's-law
temperature behaviour, NTU limits, edge cases, predict() interface, benchmark.

Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import BiogasUpgradingF2a
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
def test_mass_conservation():
    print("\n[Test 1] Per-species mass conservation (in = out_gas + absorbed)")
    m, _ = make_model()
    r = m.simulate(500.0, 0.60, dt=5.0, duration_s=300.0)
    res = m.mass_balance_residual(r, idx=-1)
    assert_true(res["CO2"] < 1e-9, f"CO2 balance residual {res['CO2']:.2e} ~ 0")
    assert_true(res["CH4"] < 1e-9, f"CH4 balance residual {res['CH4']:.2e} ~ 0")
    # total molar balance
    tot_in = r["n_in_CO2"] + r["n_in_CH4"]
    tot_out = r["n_out_CO2"][-1] + r["n_out_CH4"][-1] + r["Nabs_CO2"][-1] + r["Nabs_CH4"][-1]
    assert_true(abs(tot_in - tot_out) / tot_in < 1e-9, "Total molar balance closes")


def test_purity_above_threshold():
    print("\n[Test 2] Product CH4 purity > 0.96 (pipeline grade)")
    m, _ = make_model()
    for Q in [300.0, 500.0, 800.0]:
        r = m.simulate(Q, 0.60, dt=5.0, duration_s=300.0)
        p = r["purity_CH4_ss"]
        assert_true(p > 0.96, f"Q={Q}: purity={p:.4f} > 0.96")


def test_recovery_below_one():
    print("\n[Test 3] CH4 recovery < 1 (some methane always slips)")
    m, _ = make_model()
    r = m.simulate(500.0, 0.60, dt=5.0, duration_s=300.0)
    rec = r["CH4_recovery_ss"]
    assert_true(0.90 < rec < 1.0, f"recovery={rec:.4f} in (0.90, 1.0)")


def test_slip_accounted():
    print("\n[Test 4] Methane slip accounted: recovery + slip = 1")
    m, _ = make_model()
    r = m.simulate(500.0, 0.60, dt=5.0, duration_s=300.0)
    rec = r["CH4_recovery_ss"]
    slip = r["CH4_slip_ss"]
    assert_true(slip > 0.0, f"slip={slip:.4f} > 0 (physical)")
    assert_true(abs(rec + slip - 1.0) < 1e-9, f"recovery+slip={rec+slip:.6f} == 1")


def test_co2_preferentially_removed():
    print("\n[Test 5] CO2 removed far more than CH4 (selectivity)")
    m, _ = make_model()
    r = m.simulate(500.0, 0.60, dt=5.0, duration_s=300.0)
    assert_true(r["CO2_removal_ss"] > r["CH4_slip_ss"] * 10,
                f"CO2_removal={r['CO2_removal_ss']:.4f} >> CH4_slip={r['CH4_slip_ss']:.4f}")
    assert_true(r["CO2_removal_ss"] <= 1.0 + 1e-9, "CO2 removal <= 1")


def test_henry_temperature():
    print("\n[Test 6] Henry's law: colder water dissolves more CO2")
    m, _ = make_model()
    H_cold = m.henry_cp("CO2", 278.15)
    H_warm = m.henry_cp("CO2", 308.15)
    assert_true(H_cold > H_warm, f"H(5C)={H_cold:.2e} > H(35C)={H_warm:.2e}")
    # CO2 ~25x more soluble than CH4
    ratio = m.henry_cp("CO2", 283.15) / m.henry_cp("CH4", 283.15)
    assert_true(ratio > 15.0, f"CO2/CH4 solubility ratio={ratio:.1f} > 15")


def test_ntu_monotone_purity():
    print("\n[Test 7] Higher NTU (more water) raises CO2 removal")
    m, _ = make_model()
    r_lowL = m.simulate(500.0, 0.60, T_col_K=None, dt=5.0, duration_s=300.0)
    base_rem = r_lowL["CO2_removal_ss"]
    # double the water flow -> higher NTU -> >= CO2 removal
    m2, _ = make_model()
    m2.L_water = m2.L_water * 0.5   # less water -> lower NTU -> less removal
    r2 = m2.simulate(500.0, 0.60, dt=5.0, duration_s=300.0)
    assert_true(base_rem >= r2["CO2_removal_ss"] - 1e-9,
                f"more water removal {base_rem:.4f} >= less water {r2['CO2_removal_ss']:.4f}")


def test_transient_relaxes():
    print("\n[Test 8] Transient: removal starts at 0 and rises to steady state")
    m, _ = make_model()
    r = m.simulate(500.0, 0.60, dt=2.0, duration_s=300.0)
    assert_true(r["CO2_removal"][0] < r["CO2_removal_ss"],
                f"start={r['CO2_removal'][0]:.4f} < ss={r['CO2_removal_ss']:.4f}")
    dlast = abs(r["CO2_removal"][-1] - r["CO2_removal"][-2])
    assert_true(dlast < 1e-3, f"Near steady state: |d removal|={dlast:.2e}")
    assert_true(np.all(np.diff(r["CO2_removal"]) >= -1e-9), "Removal monotone increasing")


def test_sec_physical():
    print("\n[Test 9] Specific energy demand in physical range")
    m, _ = make_model()
    r = m.simulate(500.0, 0.60, dt=5.0, duration_s=300.0)
    sec = r["SEC_kWh_per_Nm3"]
    assert_true(0.05 < sec < 1.0, f"SEC={sec:.3f} kWh/Nm3 in (0.05, 1.0)")


def test_edge_cases():
    print("\n[Test 10] Edge cases: tiny flow and high CH4 fraction")
    m, _ = make_model()
    r_small = m.simulate(1.0, 0.60, dt=5.0, duration_s=100.0)
    assert_true(0 <= r_small["purity_CH4_ss"] <= 1.0, "Tiny flow purity in [0,1]")
    r_rich = m.simulate(500.0, 0.75, dt=5.0, duration_s=200.0)
    assert_true(r_rich["purity_CH4_ss"] > 0.96, "CH4-rich feed still >0.96 purity")
    assert_true(r_rich["CH4_recovery_ss"] < 1.0, "Recovery < 1 for rich feed")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"biogas_flow_Nm3_per_h": 500.0, "CH4_fraction_in": 0.60,
                    "dt": 5.0, "duration_s": 100.0})
    for key in ["t", "purity_CH4", "CH4_recovery", "CH4_slip", "CO2_removal",
                "biomethane_Nm3_per_h", "SEC_kWh_per_Nm3"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["purity_CH4"]), "Time arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC142", "get_info component_id == EC142")


def test_benchmark():
    print("\n[Test 12] Benchmark: 300s transient sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(500.0, 0.60, dt=2.0, duration_s=300.0)
    elapsed = time.perf_counter() - t0
    print(f"  300s transient simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_mass_conservation,
        test_purity_above_threshold,
        test_recovery_below_one,
        test_slip_accounted,
        test_co2_preferentially_removed,
        test_henry_temperature,
        test_ntu_monotone_purity,
        test_transient_relaxes,
        test_sec_physical,
        test_edge_cases,
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

    print(f"\n{'='*64}")
    print(f"EC142 Biogas Upgrading F2a (Water Scrubbing) -- "
          f"Results: {passed} passed, {failed} failed")
    print(f"{'='*64}")
    sys.exit(0 if failed == 0 else 1)
