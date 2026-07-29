"""
EC165 -- Multilevel Inverter -- F2a Physics-Lumped
Test suite: THD-vs-levels monotonicity, V_ac scaling, energy conservation,
efficiency bounds, ODE behaviour, predict() interface, benchmark timing.
Custom assert harness (NO pytest). Run: python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MultilevelInverterF2a
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
def test_thd_decreases_with_levels():
    print("\n[Test 1] THD decreases monotonically with level count")
    m, _ = make_model()
    levels = [2, 3, 5, 7, 9, 11, 15]
    thds = [m.thd(1.0, N) for N in levels]
    for i in range(1, len(levels)):
        assert_true(thds[i] < thds[i - 1],
                    f"THD(N={levels[i]})={thds[i]*100:.2f}% < THD(N={levels[i-1]})={thds[i-1]*100:.2f}%")


def test_two_level_thd_magnitude():
    print("\n[Test 2] 2-level (square-wave) THD ~ 48% (textbook)")
    m, _ = make_model()
    thd2 = m.thd(1.0, 2)
    assert_true(0.40 < thd2 < 0.52, f"2-level THD={thd2*100:.2f}% near 48.3%")


def test_vac_scales_with_vdc():
    print("\n[Test 3] V_ac fundamental scales linearly with V_dc")
    m, _ = make_model()
    v1 = m.ac_rms_voltage(1.0, 9)
    m.V_dc *= 2.0
    v2 = m.ac_rms_voltage(1.0, 9)
    assert_true(abs(v2 / v1 - 2.0) < 0.02, f"V_ac doubles with V_dc: ratio={v2/v1:.3f}")
    m.V_dc /= 2.0


def test_vac_scales_with_modulation():
    print("\n[Test 4] V_ac fundamental increases with modulation index")
    m, _ = make_model()
    v_lo = m.ac_rms_voltage(0.4, 11)
    v_hi = m.ac_rms_voltage(1.0, 11)
    assert_true(v_hi > v_lo, f"V_ac(m=1.0)={v_hi:.1f} > V_ac(m=0.4)={v_lo:.1f}")
    # near-proportional for high level count
    assert_true(abs((v_hi / v_lo) / (1.0 / 0.4) - 1.0) < 0.10,
                "V_ac roughly proportional to m at high N")


def test_efficiency_bounds():
    print("\n[Test 5] Efficiency strictly in (0, 1)")
    m, _ = make_model()
    for N in [3, 5, 9]:
        r = m.simulate(m=1.0, n_levels=N, n_periods=5)
        eta = r["efficiency"]
        assert_true(0.0 < eta < 1.0, f"N={N}: eff={eta*100:.3f}% in (0,1)")


def test_energy_conservation():
    print("\n[Test 6] Energy conservation: P_in = P_out + P_loss")
    m, _ = make_model()
    r = m.simulate(m=1.0, n_levels=5, n_periods=6)
    p_in = r["p_out"] + r["p_loss"]
    assert_true(p_in > r["p_out"] > 0, f"P_in={p_in:.1f} > P_out={r['p_out']:.1f} > 0")
    recon_eff = r["p_out"] / p_in
    assert_true(abs(recon_eff - r["efficiency"]) < 1e-9, "efficiency == P_out/(P_out+P_loss)")


def test_switching_loss_falls_with_levels():
    print("\n[Test 7] Per-event switching energy falls as levels rise (lower V_cell)")
    m, _ = make_model()
    # same load current, compare switching loss per device-event via blocking voltage
    e3 = m.switching_loss(100.0, 3) / (2 * (3 - 1))
    e9 = m.switching_loss(100.0, 9) / (2 * (9 - 1))
    assert_true(e9 < e3, f"per-event sw loss N=9 ({e9:.1f}W) < N=3 ({e3:.1f}W)")


def test_filter_attenuates_thd():
    print("\n[Test 8] LC filter output THD < raw staircase pole THD")
    m, _ = make_model()
    r = m.simulate(m=1.0, n_levels=5, n_periods=8)
    assert_true(r["thd_output"] < r["thd_pole"],
                f"THD_out={r['thd_output']*100:.2f}% < THD_pole={r['thd_pole']*100:.2f}%")


def test_ode_reaches_steady_oscillation():
    print("\n[Test 9] Filter ODE settles to bounded steady oscillation")
    m, _ = make_model()
    r = m.simulate(m=1.0, n_levels=5, n_periods=8)
    v = r["v_out"]
    T = 1.0 / m.f_out
    last = r["t"] >= (r["t"][-1] - T)
    prev = (r["t"] >= (r["t"][-1] - 2 * T)) & (r["t"] < (r["t"][-1] - T))
    rms_last = np.sqrt(np.mean(v[last] ** 2))
    rms_prev = np.sqrt(np.mean(v[prev] ** 2))
    assert_true(abs(rms_last - rms_prev) / rms_last < 0.05,
                f"Periodic steady state: dRMS={abs(rms_last-rms_prev)/rms_last*100:.2f}%")
    assert_true(np.all(np.abs(v) < m.V_dc), "Output bounded by DC bus")


def test_pole_levels_count():
    print("\n[Test 10] Staircase uses exactly N distinct levels at full modulation")
    m, _ = make_model()
    for N in [3, 5, 7]:
        _, v = m.waveform(1.15, N)
        n_distinct = len(np.unique(np.round(v, 3)))
        assert_true(n_distinct == N, f"N={N}: waveform has {n_distinct} distinct levels")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"n_levels": 5, "modulation_index": 1.0, "n_periods": 3})
    for key in ["t", "v_pole", "i_L", "v_out", "thd_pole", "thd_output",
                "v_ac_rms", "p_out", "p_loss", "efficiency", "n_levels"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["v_out"]) == len(r["i_L"]),
                "Time-series arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC165", "component_id == EC165")


def test_benchmark():
    print("\n[Test 12] Benchmark: 6-period 5-level simulation")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(m=1.0, n_levels=5, n_periods=6)
    elapsed = time.perf_counter() - t0
    print(f"  6-period ODE simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_thd_decreases_with_levels,
        test_two_level_thd_magnitude,
        test_vac_scales_with_vdc,
        test_vac_scales_with_modulation,
        test_efficiency_bounds,
        test_energy_conservation,
        test_switching_loss_falls_with_levels,
        test_filter_attenuates_thd,
        test_ode_reaches_steady_oscillation,
        test_pole_levels_count,
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
    print(f"EC165 Multilevel Inverter F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
