"""
EC128 -- Conventional Hydroelectric Dam -- F2a Dynamic
Test suite: physics sanity, governor response, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import ConventionalHydroDam_F2a
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


def test_power_positive():
    print("\n[Test 1] Power output is positive at rated conditions")
    m, _ = make_model()
    P = m.power_output(m.Q_rated, m.H_rated)
    assert_true(P > 0, f"P_rated = {P/1e6:.1f} MW > 0")
    assert_true(P < m.P_rated * 1.2, f"P = {P/1e6:.1f} MW < 1.2*P_rated")


def test_efficiency_range():
    print("\n[Test 2] Turbine efficiency in [0, 0.95]")
    m, _ = make_model()
    eta = m.turbine_efficiency(m.Q_rated, m.H_rated)
    assert_true(0.80 < eta <= 0.95, f"eta_rated = {eta:.4f}")
    eta_half = m.turbine_efficiency(m.Q_rated * 0.5, m.H_rated)
    assert_true(eta_half < eta, f"eta(50%) = {eta_half:.4f} < eta(100%) = {eta:.4f}")


def test_flow_from_gate():
    print("\n[Test 3] Turbine flow proportional to gate opening")
    m, _ = make_model()
    Q1 = m.turbine_flow(0.5, m.H_rated)
    Q2 = m.turbine_flow(1.0, m.H_rated)
    assert_true(Q2 > Q1, f"Q(G=1.0)={Q2:.1f} > Q(G=0.5)={Q1:.1f}")
    ratio = Q2 / Q1
    assert_true(abs(ratio - 2.0) < 0.1, f"Q ratio = {ratio:.3f} ~ 2.0")


def test_governor_step():
    print("\n[Test 4] Governor tracks step in G_ref")
    m, _ = make_model()
    def G_step(t):
        return 0.3 if t < 50 else 0.8
    r = m.simulate(G_step, dt=1.0, duration_s=200.0)
    # Gate should approach 0.8 in second half
    G_final = r["G_gate"][-1]
    assert_true(abs(G_final - 0.8) < 0.05,
                f"G_final={G_final:.3f} ~ 0.8")


def test_reservoir_drops_if_no_inflow():
    print("\n[Test 5] Reservoir drops when inflow < outflow")
    m, _ = make_model()
    r = m.simulate(0.8, Q_inflow=0.0, dt=1.0, duration_s=600.0)
    assert_true(r["H_reservoir"][-1] < r["H_reservoir"][0],
                f"H: {r['H_reservoir'][0]:.2f} -> {r['H_reservoir'][-1]:.2f}")


def test_more_gate_more_power():
    print("\n[Test 6] Larger gate opening -> more power")
    m, _ = make_model()
    r1 = m.simulate(0.3, dt=1.0, duration_s=120.0)
    r2 = m.simulate(0.8, dt=1.0, duration_s=120.0)
    P1 = np.mean(r1["P_output"][10:])
    P2 = np.mean(r2["P_output"][10:])
    assert_true(P2 > P1, f"P(G=0.8)={P2/1e6:.1f} MW > P(G=0.3)={P1/1e6:.1f} MW")


def test_load_rejection():
    print("\n[Test 7] Load rejection: gate closes, flow decreases")
    m, _ = make_model()
    def G_reject(t):
        return 0.7 if t < 60 else 0.05
    r = m.simulate(G_reject, dt=0.5, duration_s=120.0)
    idx_before = np.argmin(np.abs(r["t"] - 55))
    idx_after = np.argmin(np.abs(r["t"] - 115))
    assert_true(r["P_output"][idx_after] < r["P_output"][idx_before],
                "Power decreases after load rejection")


def test_predict_interface():
    print("\n[Test 8] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"G_ref": 0.5, "dt": 1.0, "duration_s": 10.0})
    for key in ["t", "H_reservoir", "Q_penstock", "Q_turbine", "G_gate",
                "P_output", "efficiency", "Q_inflow"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["P_output"]), "Arrays same length")


def test_benchmark():
    print("\n[Test 9] Benchmark: 3600s sim at dt=1.0")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.6, dt=1.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 30.0, "Completes in < 30 s")


if __name__ == "__main__":
    tests = [
        test_power_positive,
        test_efficiency_range,
        test_flow_from_gate,
        test_governor_step,
        test_reservoir_drops_if_no_inflow,
        test_more_gate_more_power,
        test_load_rejection,
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
    print(f"EC128 Hydro Dam F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
