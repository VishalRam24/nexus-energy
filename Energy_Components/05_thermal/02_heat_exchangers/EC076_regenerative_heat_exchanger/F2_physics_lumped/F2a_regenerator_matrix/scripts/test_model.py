"""
EC076 -- Regenerative Heat Exchanger -- F2a Physics-Lumped Regenerator Matrix
Test suite: energy conservation, effectiveness bounds, monotonicity,
matrix-capacity dependence, ODE vs correlation cross-check, predict() interface.
NO pytest -- custom assert_true harness, run as __main__.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import RegeneratorF2a, AIR_CP, AIR_RHO
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
def test_air_properties():
    print("\n[Test 1] Hardcoded air properties (Cengel & Ghajar 2015)")
    assert_true(abs(AIR_CP - 1006.0) < 1.0, f"air cp={AIR_CP} J/(kg.K) ~ 1006")
    assert_true(1.1 < AIR_RHO < 1.3, f"air rho={AIR_RHO} kg/m3 ~ 1.18")


def test_effectiveness_bounds():
    print("\n[Test 2] 0 < effectiveness < 1 (both ODE and correlation)")
    _, cm = make_model()
    r = cm.predict({"T_h_in_K": 573.15, "T_c_in_K": 293.15, "n_cycles": 60})
    assert_true(0.0 < r["effectiveness_ode"] < 1.0,
                f"eps_ode={r['effectiveness_ode']:.4f} in (0,1)")
    assert_true(0.0 < r["effectiveness_correlation"] < 1.0,
                f"eps_corr={r['effectiveness_correlation']:.4f} in (0,1)")


def test_energy_conservation():
    print("\n[Test 3] Energy conservation: hot duty ~ cold duty at periodic SS")
    _, cm = make_model()
    r = cm.predict({"T_h_in_K": 573.15, "T_c_in_K": 293.15, "n_cycles": 80})
    q_c, q_h = r["Q_kW"], r["Q_hot_kW"]
    rel = abs(q_h - q_c) / max(abs(q_c), 1e-9)
    assert_true(rel < 0.02,
                f"|Q_hot-Q_cold|/Q = {rel*100:.2f}% < 2% (Q_c={q_c:.1f}, Q_h={q_h:.1f} kW)")


def test_outlet_temp_bounds():
    print("\n[Test 4] Outlet temps bracketed by inlet temps (2nd law)")
    _, cm = make_model()
    Th_in, Tc_in = 573.15, 293.15
    r = cm.predict({"T_h_in_K": Th_in, "T_c_in_K": Tc_in, "n_cycles": 60})
    assert_true(Tc_in <= r["T_c_out"] <= Th_in,
                f"Tc_in <= T_c_out({r['T_c_out']:.1f}) <= Th_in")
    assert_true(Tc_in <= r["T_h_out"] <= Th_in,
                f"Tc_in <= T_h_out({r['T_h_out']:.1f}) <= Th_in")
    assert_true(r["T_c_out"] > Tc_in and r["T_h_out"] < Th_in,
                "Cold stream heated, hot stream cooled")


def test_eps_rises_with_matrix_capacity():
    print("\n[Test 5] Effectiveness RISES with matrix capacity ratio Cr*")
    m, cm = make_model()
    eps_prev = -1.0
    crs_prev = -1.0
    for rpm in [2.0, 5.0, 10.0, 20.0, 40.0]:
        eps = m.effectiveness_correlation(rpm=rpm)
        crs = m.matrix_capacity_ratio(rpm=rpm)
        assert_true(crs > crs_prev, f"Cr*({rpm} rpm)={crs:.2f} increasing")
        assert_true(eps >= eps_prev - 1e-9,
                    f"eps({rpm} rpm)={eps:.4f} >= prev {eps_prev:.4f}")
        eps_prev, crs_prev = eps, crs


def test_eps_rises_with_matrix_capacity_ode():
    print("\n[Test 6] ODE effectiveness also rises with rpm (matrix capacity)")
    _, cm = make_model()
    r_slow = cm.predict({"rpm": 2.0, "n_cycles": 80})
    r_fast = cm.predict({"rpm": 40.0, "n_cycles": 80})
    assert_true(r_fast["effectiveness_ode"] > r_slow["effectiveness_ode"],
                f"eps_ode(40rpm)={r_fast['effectiveness_ode']:.4f} > "
                f"eps_ode(2rpm)={r_slow['effectiveness_ode']:.4f}")


def test_eps_rises_with_ntu():
    print("\n[Test 7] Effectiveness rises with NTU_o (hA)")
    m, _ = make_model()
    eps_prev = -1.0
    for hA in [2000.0, 6000.0, 20000.0]:
        m.hA_h = m.hA_c = hA
        eps = m.effectiveness_correlation()
        assert_true(eps > eps_prev, f"eps(hA={hA:.0f})={eps:.4f} > prev")
        eps_prev = eps


def test_ode_vs_correlation():
    print("\n[Test 8] ODE effectiveness agrees with Coppage-London correlation")
    _, cm = make_model()
    r = cm.predict({"T_h_in_K": 573.15, "T_c_in_K": 293.15, "n_cycles": 80})
    diff = abs(r["effectiveness_ode"] - r["effectiveness_correlation"])
    assert_true(diff < 0.05,
                f"|eps_ode - eps_corr| = {diff:.4f} < 0.05 "
                f"({r['effectiveness_ode']:.4f} vs {r['effectiveness_correlation']:.4f})")


def test_periodic_convergence():
    print("\n[Test 9] Matrix reaches periodic steady state (eps history settles)")
    _, cm = make_model()
    r = cm.predict({"T_h_in_K": 573.15, "T_c_in_K": 293.15, "n_cycles": 80})
    hist = r["eps_history"]
    assert_true(len(hist) >= 3, "Ran multiple cycles")
    settle = abs(hist[-1] - hist[-2])
    assert_true(settle < 1e-3, f"Cyclic equilibrium: |d eps| = {settle:.2e} < 1e-3")


def test_zero_dt_zero_duty():
    print("\n[Test 10] Equal inlet temps -> zero duty, zero rise")
    _, cm = make_model()
    r = cm.predict({"T_h_in_K": 400.0, "T_c_in_K": 400.0, "n_cycles": 30})
    assert_true(abs(r["Q_kW"]) < 1e-6, f"Q={r['Q_kW']:.2e} kW ~ 0")
    assert_true(abs(r["T_c_out"] - 400.0) < 1e-6, "T_c_out unchanged")


def test_high_ntu_approaches_unity():
    print("\n[Test 11] Very large NTU + large matrix capacity -> eps -> ~1")
    m, _ = make_model()
    m.hA_h = m.hA_c = 1.0e6
    eps = m.effectiveness_correlation(rpm=60.0)
    assert_true(eps > 0.9, f"eps={eps:.4f} > 0.9 (recuperator limit)")
    assert_true(eps < 1.0, f"eps={eps:.4f} < 1 (still bounded)")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() / get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC076", "component_id == EC076")
    r = cm.predict({"T_h_in_K": 500.0, "T_c_in_K": 300.0, "n_cycles": 40})
    for key in ["T_h_out", "T_c_out", "effectiveness_ode",
                "effectiveness_correlation", "Q_kW", "NTU_o", "Cr_star"]:
        assert_true(key in r, f"predict output has '{key}'")


def test_benchmark():
    print("\n[Test 13] Benchmark: full periodic-SS simulation timing")
    _, cm = make_model()
    t0 = time.perf_counter()
    cm.predict({"T_h_in_K": 573.15, "T_c_in_K": 293.15, "n_cycles": 60})
    elapsed = time.perf_counter() - t0
    print(f"  60-cycle simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_air_properties,
        test_effectiveness_bounds,
        test_energy_conservation,
        test_outlet_temp_bounds,
        test_eps_rises_with_matrix_capacity,
        test_eps_rises_with_matrix_capacity_ode,
        test_eps_rises_with_ntu,
        test_ode_vs_correlation,
        test_periodic_convergence,
        test_zero_dt_zero_duty,
        test_high_ntu_approaches_unity,
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
    print(f"EC076 Regenerator F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
