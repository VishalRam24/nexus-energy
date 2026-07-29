"""
EC119 -- Molten Salt Reactor (MSR) -- F2a Point Kinetics
Test suite: flowing-fuel beta reduction, negative-feedback stability,
energy conservation, edge cases, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MSR_F2a, PCM
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
def test_flowing_fuel_reduces_beta():
    print("\n[Test 1] MSR signature: flowing fuel reduces effective beta")
    m, _ = make_model()
    b_static = m.beta_eff(0.0)        # stagnant: no loss
    b_flow = m.beta_eff(1.0)          # rated flow
    b_fast = m.beta_eff(1.5)          # higher flow
    assert_true(abs(b_static - m.beta) < 1e-9,
                f"Stagnant beta_eff={b_static:.5f} == static beta={m.beta:.5f}")
    assert_true(b_flow < b_static,
                f"Flowing beta_eff={b_flow:.5f} < stagnant {b_static:.5f}")
    assert_true(b_fast < b_flow,
                f"Faster flow loses more: {b_fast:.5f} < {b_flow:.5f}")
    loss = 100 * (1 - b_flow / b_static)
    print(f"  Precursor drift loss at rated flow = {loss:.1f}% of beta")
    assert_true(0.5 < loss < 50, f"Loss {loss:.1f}% physically reasonable")


def test_negative_temperature_feedback():
    print("\n[Test 2] Strong negative temperature reactivity feedback")
    m, _ = make_model()
    rho_cold = m.reactivity(m.T_core0, 0.0)
    rho_hot = m.reactivity(m.T_core0 + 50.0, 0.0)
    assert_true(rho_hot < rho_cold,
                f"Hotter core -> less reactive ({rho_hot/PCM:.1f} < {rho_cold/PCM:.1f} pcm)")
    assert_true((m.alpha_fuel + m.alpha_graph) < 0,
                "Combined temperature coefficient is negative")


def test_feedback_stability():
    print("\n[Test 3] Positive reactivity step -> bounded, self-limited power")
    m, _ = make_model()
    r = m.simulate(rho_ext_pcm=80.0, flow_fraction=1.0, dt=1.0, duration_s=400.0)
    pf = r["power_fraction"]
    assert_true(np.all(np.isfinite(pf)), "Power stays finite (no runaway)")
    assert_true(pf.max() < 20.0,
                f"Peak power fraction {pf.max():.2f} bounded by feedback")
    # settles: last 10% of trace nearly flat
    tail = pf[int(0.9 * len(pf)):]
    assert_true(tail.std() < 0.05 * tail.mean() + 1e-3,
                f"Power settles to steady state (tail std {tail.std():.3e})")
    assert_true(pf[-1] > 1.0, f"New steady power {pf[-1]:.3f} > 1 (rose then settled)")


def test_steady_state_holds():
    print("\n[Test 4] Zero external reactivity at equilibrium stays put")
    m, _ = make_model()
    r = m.simulate(rho_ext_pcm=0.0, flow_fraction=1.0, dt=1.0, duration_s=100.0)
    pf = r["power_fraction"]
    assert_true(np.allclose(pf, 1.0, atol=2e-2),
                f"Power stays ~1.0 (range {pf.min():.4f}-{pf.max():.4f})")


def test_negative_reactivity_drops_power():
    print("\n[Test 5] Negative reactivity (control rod) drops power")
    m, _ = make_model()
    r = m.simulate(rho_ext_pcm=-200.0, flow_fraction=1.0, dt=1.0, duration_s=200.0)
    assert_true(r["power_fraction"][-1] < 1.0,
                f"Power drops to {r['power_fraction'][-1]:.3f} < 1.0")
    assert_true(np.all(r["power_fraction"] >= 0),
                "Power fraction stays non-negative")


def test_energy_conservation():
    print("\n[Test 6] Energy conservation in lumped thermal model")
    m, _ = make_model()
    r = m.simulate(rho_ext_pcm=0.0, flow_fraction=1.0, dt=0.5, duration_s=300.0)
    t = r["t"]
    P = r["power_w"]
    Tc, Tl = r["T_core_K"], r["T_loop_K"]
    # Fission energy in over the window
    E_in = np.trapezoid(P, t)
    # Energy delivered to the secondary sink
    Q_sink = m.hA_ls * (Tl - m.T_sink)
    E_sink = np.trapezoid(Q_sink, t)
    # Internal energy change of the two thermal lumps
    dU = (m.M_core * m.cp_core * (Tc[-1] - Tc[0])
          + m.M_loop * m.cp_loop * (Tl[-1] - Tl[0]))
    # Balance: E_in = E_sink + dU
    residual = abs(E_in - E_sink - dU)
    rel = residual / max(E_in, 1.0)
    print(f"  E_in={E_in:.3e} J, E_sink={E_sink:.3e} J, dU={dU:.3e} J, rel={rel:.2e}")
    assert_true(rel < 1e-3, f"Energy balance closes (rel err {rel:.2e} < 1e-3)")


def test_thermal_settles_to_balance():
    print("\n[Test 7] At steady state, power out = power to sink")
    m, _ = make_model()
    r = m.simulate(rho_ext_pcm=0.0, flow_fraction=1.0, dt=1.0, duration_s=600.0)
    P_end = r["power_w"][-1]
    Q_sink_end = m.hA_ls * (r["T_loop_K"][-1] - m.T_sink)
    rel = abs(P_end - Q_sink_end) / P_end
    assert_true(rel < 0.02,
                f"P={P_end:.3e} ~ Q_sink={Q_sink_end:.3e} (rel {rel:.2e})")


def test_temperatures_physical():
    print("\n[Test 8] Temperatures stay in physical MSR range")
    m, _ = make_model()
    r = m.simulate(rho_ext_pcm=100.0, flow_fraction=1.0, dt=1.0, duration_s=300.0)
    Tc = r["T_core_K"]
    assert_true(np.all(Tc > m.T_sink), "Core hotter than sink (heat flows out)")
    assert_true(np.all(Tc < 1400.0), f"Core T_max={Tc.max():.1f} K below salt limit")
    assert_true(np.all(r["T_core_K"] >= r["T_loop_K"] - 1.0),
                "Core node >= loop node (heat source is the core)")


def test_stagnant_more_reactive_than_flowing():
    print("\n[Test 9] Same insertion: stagnant fuel responds milder (larger beta)")
    m, _ = make_model()
    # +100 pcm with flow loses delayed neutrons -> closer to prompt -> faster
    rf = m.simulate(rho_ext_pcm=100.0, flow_fraction=1.0, dt=0.2, duration_s=20.0)
    rs = m.simulate(rho_ext_pcm=100.0, flow_fraction=0.0, dt=0.2, duration_s=20.0)
    # early peak: flowing fuel rises faster because beta_eff smaller
    peak_flow = rf["power_fraction"][:50].max()
    peak_stag = rs["power_fraction"][:50].max()
    assert_true(peak_flow >= peak_stag - 1e-6,
                f"Flowing early peak {peak_flow:.3f} >= stagnant {peak_stag:.3f}")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"rho_ext_pcm": 50.0, "dt": 1.0, "duration_s": 20.0})
    for key in ["t", "n", "power_w", "power_fraction", "T_core_K",
                "T_loop_K", "reactivity_pcm", "beta_eff", "beta_static"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["power_fraction"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC119", "get_info id == EC119")


def test_callable_reactivity():
    print("\n[Test 11] Time-varying (callable) reactivity insertion")
    m, _ = make_model()
    step = lambda t: 0.0 if t < 50 else 60.0
    r = m.simulate(rho_ext_pcm=step, flow_fraction=1.0, dt=1.0, duration_s=200.0)
    before = r["power_fraction"][r["t"] < 50].max()
    after = r["power_fraction"][r["t"] > 150].mean()
    assert_true(after > before, f"Power rises after step ({after:.3f} > {before:.3f})")


def test_benchmark():
    print("\n[Test 12] Benchmark: 200 s transient")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(rho_ext_pcm=50.0, flow_fraction=1.0, dt=0.5, duration_s=200.0)
    elapsed = time.perf_counter() - t0
    print(f"  200 s transient in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_flowing_fuel_reduces_beta,
        test_negative_temperature_feedback,
        test_feedback_stability,
        test_steady_state_holds,
        test_negative_reactivity_drops_power,
        test_energy_conservation,
        test_thermal_settles_to_balance,
        test_temperatures_physical,
        test_stagnant_more_reactive_than_flowing,
        test_predict_interface,
        test_callable_reactivity,
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
    print(f"EC119 MSR F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
