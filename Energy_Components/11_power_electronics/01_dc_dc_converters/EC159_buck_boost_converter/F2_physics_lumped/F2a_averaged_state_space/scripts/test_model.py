"""
EC159 -- Buck-Boost Converter (Inverting) -- F2a State-Space Averaged Model
Test suite: physics sanity, known limits, steady-state, ODE convergence,
edge cases, predict() interface, benchmark timing. (NO pytest.)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import BuckBoostConverterF2a
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
def test_ideal_gain_formula():
    print("\n[Test 1] Ideal gain Vout/Vin = -d/(1-d)")
    m, _ = make_model()
    for d in [0.25, 0.5, 0.75]:
        g = m.ideal_gain(d)
        expect = -d / (1.0 - d)
        assert_true(abs(g - expect) < 1e-12, f"d={d}: gain={g:.4f} == {expect:.4f}")
    assert_true(abs(m.ideal_gain(0.5) - (-1.0)) < 1e-12, "d=0.5 -> gain = -1.0 (unity inverting)")


def test_buck_vs_boost_region():
    print("\n[Test 2] Buck region (d<0.5) vs boost region (d>0.5)")
    m, _ = make_model()
    g_buck = abs(m.ideal_gain(0.3))
    g_boost = abs(m.ideal_gain(0.7))
    assert_true(g_buck < 1.0, f"d=0.3 -> |gain|={g_buck:.4f} < 1 (step-down)")
    assert_true(g_boost > 1.0, f"d=0.7 -> |gain|={g_boost:.4f} > 1 (step-up)")


def test_lossless_steady_state_matches_ideal():
    print("\n[Test 3] Lossless steady state matches ideal gain")
    # Zero out parasitics
    m, _ = make_model()
    m.R_L = 0.0
    m.R_ds_on = 0.0
    m.V_f = 0.0
    for d in [0.3, 0.5, 0.7]:
        ss = m.steady_state(d, Vin=24.0, R_load=4.0)
        expect = m.ideal_gain(d) * 24.0
        assert_true(abs(ss["vout"] - expect) < 1e-6,
                    f"d={d}: Vout={ss['vout']:.4f} == ideal {expect:.4f}")


def test_lossy_gain_below_ideal():
    print("\n[Test 4] Parasitics reduce |Vout| below ideal")
    m, _ = make_model()
    for d in [0.5, 0.7]:
        ss = m.steady_state(d, Vin=24.0, R_load=4.0)
        ideal = abs(m.ideal_gain(d) * 24.0)
        assert_true(abs(ss["vout"]) < ideal,
                    f"d={d}: |Vout|={abs(ss['vout']):.3f} < ideal {ideal:.3f}")


def test_efficiency_range():
    print("\n[Test 5] Steady-state efficiency in (0,1)")
    m, _ = make_model()
    for d in [0.3, 0.5, 0.7]:
        ss = m.steady_state(d, Vin=24.0, R_load=4.0)
        assert_true(0.0 < ss["efficiency"] < 1.0,
                    f"d={d}: eta={ss['efficiency']*100:.2f}%")


def test_output_is_inverting():
    print("\n[Test 6] Output voltage is negative (inverting)")
    m, _ = make_model()
    ss = m.steady_state(0.5, Vin=24.0, R_load=4.0)
    assert_true(ss["vout"] < 0.0, f"Vout={ss['vout']:.3f} V < 0")
    assert_true(ss["vC"] > 0.0, f"vC magnitude={ss['vC']:.3f} V > 0")


def test_transient_converges_to_steady_state():
    print("\n[Test 7] Transient settles to steady-state operating point")
    m, _ = make_model()
    r = m.simulate(0.5, Vin=24.0, R_load=4.0, dt=2.0e-6, duration_s=6.0e-3)
    ss = m.steady_state(0.5, Vin=24.0, R_load=4.0)
    err = abs(r["vout"][-1] - ss["vout"])
    assert_true(err < 0.05 * abs(ss["vout"]) + 1e-3,
                f"Settled Vout={r['vout'][-1]:.3f} ~ SS {ss['vout']:.3f} (err={err:.4f})")


def test_capacitor_charges_monotone_at_start():
    print("\n[Test 8] Capacitor voltage rises from zero initial condition")
    m, _ = make_model()
    # Long enough for the underdamped LC transient to settle (ring decays).
    r = m.simulate(0.5, Vin=24.0, R_load=4.0, dt=4.0e-6, duration_s=2.0e-2,
                   iL0=0.0, vC0=0.0)
    assert_true(r["vC"][0] < 1e-6, "vC starts ~0")
    assert_true(r["vC"][-1] > r["vC"][0], "vC increases over the transient")
    # Once settled, the averaged inductor current must equal the positive
    # steady-state value (it can ring negative mid-transient -- that is real LC
    # physics, so check the settled tail mean, not an instantaneous sample).
    iL_settled = float(np.mean(r["iL"][-50:]))
    ss = m.steady_state(0.5, Vin=24.0, R_load=4.0)
    assert_true(iL_settled > 0.0, f"Settled inductor current positive ({iL_settled:.3f} A)")
    assert_true(abs(iL_settled - ss["iL"]) < 0.05 * ss["iL"] + 1e-3,
                f"Settled iL={iL_settled:.3f} ~ SS iL={ss['iL']:.3f} A")


def test_power_balance():
    print("\n[Test 9] Power balance: P_in = P_out + P_loss (>=)")
    m, _ = make_model()
    ss = m.steady_state(0.6, Vin=24.0, R_load=4.0)
    p_out = ss["vC"] ** 2 / 4.0
    p_in = 0.6 * 24.0 * ss["iL"]
    assert_true(p_in >= p_out - 1e-9, f"P_in={p_in:.3f} >= P_out={p_out:.3f}")
    assert_true(p_in > p_out, "Losses present: P_in strictly > P_out")


def test_higher_load_resistance_higher_efficiency():
    print("\n[Test 10] Lighter load (higher R) -> higher efficiency")
    m, _ = make_model()
    eta_heavy = m.steady_state(0.5, Vin=24.0, R_load=2.0)["efficiency"]
    eta_light = m.steady_state(0.5, Vin=24.0, R_load=20.0)["efficiency"]
    assert_true(eta_light > eta_heavy,
                f"eta(R=20)={eta_light*100:.2f}% > eta(R=2)={eta_heavy*100:.2f}%")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"duty": 0.5, "v_in": 24.0, "R_load": 4.0,
                    "dt": 2.0e-6, "duration_s": 2.0e-3})
    for key in ["t", "iL", "vC", "vout", "p_in", "p_out", "efficiency", "steady_state"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["vout"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC159", "get_info component_id == EC159")


def test_benchmark():
    print("\n[Test 12] Benchmark: 2 ms averaged sim at dt=1 us")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.5, Vin=24.0, R_load=4.0, dt=1.0e-6, duration_s=2.0e-3)
    elapsed = time.perf_counter() - t0
    print(f"  2 ms simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_ideal_gain_formula,
        test_buck_vs_boost_region,
        test_lossless_steady_state_matches_ideal,
        test_lossy_gain_below_ideal,
        test_efficiency_range,
        test_output_is_inverting,
        test_transient_converges_to_steady_state,
        test_capacitor_charges_monotone_at_start,
        test_power_balance,
        test_higher_load_resistance_higher_efficiency,
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
    print(f"EC159 Buck-Boost F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
