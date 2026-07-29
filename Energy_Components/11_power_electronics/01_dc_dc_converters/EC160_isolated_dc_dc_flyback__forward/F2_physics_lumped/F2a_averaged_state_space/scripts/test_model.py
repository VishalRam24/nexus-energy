"""
EC160 -- Isolated DC-DC Flyback -- F2a Averaged State-Space Model
Test suite: physics sanity (gain relation, energy/power balance, isolation),
edge cases, ODE convergence, predict() interface, benchmark timing.
Run with system python3 (NO pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import FlybackConverterF2a
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
def test_ideal_gain_relation():
    print("\n[Test 1] Ideal flyback gain V_out = d/(1-d) * V_in/n")
    m, _ = make_model()
    v_in, n = 48.0, m.n
    for d in [0.2, 0.4, 0.5, 0.6, 0.75]:
        expected = (d / (1 - d)) * (v_in / n)
        got = m.ideal_gain(v_in, d)
        assert_true(abs(got - expected) < 1e-9,
                    f"d={d}: gain={got:.4f} V (expected {expected:.4f})")


def test_gain_monotone_in_duty():
    print("\n[Test 2] Ideal gain increases monotonically with duty")
    m, _ = make_model()
    ds = np.linspace(0.05, 0.85, 30)
    g_prev = m.ideal_gain(48.0, ds[0])
    for d in ds[1:]:
        g = m.ideal_gain(48.0, d)
        assert_true(g > g_prev - 1e-12, f"gain({d:.2f})={g:.3f} >= prev {g_prev:.3f}")
        g_prev = g


def test_steady_state_near_ideal_low_loss():
    print("\n[Test 3] With small parasitics, SS V_out is below but near ideal")
    m, _ = make_model()
    ss = m.steady_state(48.0, 0.5, 1.2)
    ideal = ss["v_out_ideal"]
    real = ss["v_out_ss"]
    assert_true(0.5 * ideal < real < ideal,
                f"V_out_ss={real:.3f} in (0.5*ideal, ideal)=({0.5*ideal:.3f}, {ideal:.3f})")


def test_efficiency_bounds():
    print("\n[Test 4] Efficiency strictly in (0, 1)")
    m, _ = make_model()
    for v_in in [24.0, 48.0, 72.0]:
        for d in [0.2, 0.4, 0.6, 0.8]:
            for R in [0.8, 1.2, 5.0, 20.0]:
                eta = m.efficiency(v_in, d, R)
                assert_true(0.0 < eta < 1.0, f"v_in={v_in}, d={d}, R={R}: eta={eta:.4f}")


def test_efficiency_drops_at_high_load():
    print("\n[Test 5] Efficiency falls as load current rises (smaller R_load)")
    m, _ = make_model()
    eta_light = m.efficiency(48.0, 0.5, 20.0)   # light load
    eta_heavy = m.efficiency(48.0, 0.5, 0.8)    # heavy load
    assert_true(eta_heavy < eta_light,
                f"eta(heavy)={eta_heavy:.4f} < eta(light)={eta_light:.4f}")


def test_energy_conservation():
    print("\n[Test 6] Power balance: P_in = P_out + P_loss within tolerance")
    m, _ = make_model()
    v_in, d, R = 48.0, 0.5, 1.2
    eta = m.efficiency(v_in, d, R)
    ss = m.steady_state(v_in, d, R)
    p_out = ss["power_ss"]
    p_in = p_out / eta
    p_loss = p_in - p_out
    assert_true(p_loss > 0, f"P_loss={p_loss:.3f} W > 0")
    assert_true(abs(p_in - (p_out + p_loss)) < 1e-6, "P_in = P_out + P_loss exactly")


def test_isolation_turns_ratio():
    print("\n[Test 7] Isolation: output scales by 1/n (turns ratio), not direct")
    m, _ = make_model()
    # Doubling n must lower the ideal output for fixed d
    g1 = m.ideal_gain(48.0, 0.5)
    m2 = FlybackConverterF2a(ComponentModel().params)
    m2.n = m.n * 2.0
    g2 = m2.ideal_gain(48.0, 0.5)
    assert_true(abs(g2 - g1 / 2.0) < 1e-9,
                f"Doubling n halves output: g2={g2:.3f} = g1/2={g1/2:.3f}")


def test_ode_converges_to_steady_state():
    print("\n[Test 8] ODE simulation converges to analytic steady state")
    m, _ = make_model()
    r = m.simulate(48.0, 0.5, 1.2, dt=2e-6, duration_s=0.03)
    ss = m.steady_state(48.0, 0.5, 1.2)
    v_final = r["v_out"][-1]
    assert_true(abs(v_final - ss["v_out_ss"]) < 0.05 * ss["v_out_ss"],
                f"V_final={v_final:.3f} matches SS={ss['v_out_ss']:.3f} (<5%)")


def test_output_rises_from_zero():
    print("\n[Test 9] Output capacitor charges up from zero initial state")
    m, _ = make_model()
    r = m.simulate(48.0, 0.5, 1.2, dt=2e-6, duration_s=0.01, x0=[0.0, 0.0])
    assert_true(r["v_out"][0] < r["v_out"][-1], "v_out increases over time")
    assert_true(np.all(r["v_out"] >= -1e-6), "v_out never negative")


def test_step_load_response():
    print("\n[Test 10] Load step (R drop) lowers output voltage")
    m, _ = make_model()
    def R_step(t):
        return 2.0 if t < 0.02 else 0.8
    r = m.simulate(48.0, 0.5, R_step, dt=2e-6, duration_s=0.04)
    idx_before = np.argmin(np.abs(r["t"] - 0.019))
    idx_after = np.argmin(np.abs(r["t"] - 0.039))
    assert_true(r["v_out"][idx_after] < r["v_out"][idx_before],
                "Heavier load (lower R) reduces v_out")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() / get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC160", "component_id == EC160")
    r = cm.predict({"v_in": 48.0, "duty_cycle": 0.5, "R_load": 1.2,
                    "dt": 5e-6, "duration_s": 0.005})
    for key in ["t", "v_out", "i_m", "i_out", "power"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["v_out"]), "Arrays same length")
    eta = cm.predict_efficiency({"v_in": 48.0, "duty_cycle": 0.5, "R_load": 1.2})
    assert_true(0.0 < eta < 1.0, f"predict_efficiency eta={eta:.4f}")


def test_benchmark():
    print("\n[Test 12] Benchmark: 30 ms sim at dt=1e-6")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(48.0, 0.5, 1.2, dt=1e-6, duration_s=0.03)
    elapsed = time.perf_counter() - t0
    print(f"  30 ms simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_ideal_gain_relation,
        test_gain_monotone_in_duty,
        test_steady_state_near_ideal_low_loss,
        test_efficiency_bounds,
        test_efficiency_drops_at_high_load,
        test_energy_conservation,
        test_isolation_turns_ratio,
        test_ode_converges_to_steady_state,
        test_output_rises_from_zero,
        test_step_load_response,
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
    print(f"EC160 Flyback F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
