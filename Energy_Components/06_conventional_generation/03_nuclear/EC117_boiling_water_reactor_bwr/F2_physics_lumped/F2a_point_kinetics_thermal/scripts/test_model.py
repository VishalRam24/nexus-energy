"""
EC117 -- Boiling Water Reactor (BWR) -- F2a Point Kinetics + Thermal-Hydraulics
Test suite: kinetics/feedback physics sanity, conservation, edge cases, interface.
Run with: python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import BWR_F2a
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
def test_steady_state_holds():
    print("\n[Test 1] Zero external reactivity -> reactor stays at equilibrium")
    m, _ = make_model()
    r = m.simulate(0.0, duration_s=100.0, dt=0.2, n0=1.0)
    P0 = r["power_mw"][0]
    drift = np.max(np.abs(r["power_mw"] - P0)) / P0
    assert_true(r["success"], "Integration succeeded")
    assert_true(drift < 0.02, f"Power drift {drift*100:.3f}% < 2% (self-consistent steady state)")
    assert_true(abs(r["reactivity"][-1]) < 1e-4, f"|rho_final|={abs(r['reactivity'][-1]):.2e} ~ 0")


def test_positive_reactivity_feedback_stabilises():
    print("\n[Test 2] +0.3$ step -> power rises then feedback arrests it (bounded)")
    m, _ = make_model()
    r = m.simulate(m.dollars_to_reactivity(0.3), duration_s=80.0, dt=0.01, n0=1.0)
    P0 = r["power_mw"][0]
    peak = r["power_mw"].max()
    final = r["power_mw"][-1]
    assert_true(peak > P0, f"Power overshoots: peak {peak:.0f} > P0 {P0:.0f}")
    assert_true(np.isfinite(peak) and peak < 5 * P0, f"Peak bounded (no runaway): {peak:.0f} MW")
    # Settles below the prompt peak -> feedback stabilised the transient.
    assert_true(final < peak, f"Settles below peak: final {final:.0f} < peak {peak:.0f}")
    # New equilibrium reached (last 5 s nearly flat).
    tail = r["power_mw"][r["t"] > r["t"][-1] - 5.0]
    assert_true((tail.max() - tail.min()) / final < 0.05, "Reaches a new steady level")


def test_negative_reactivity_drops_power():
    print("\n[Test 3] Negative reactivity (-1$) -> power and fuel temperature fall")
    m, _ = make_model()
    r = m.simulate(m.dollars_to_reactivity(-1.0), duration_s=100.0, dt=0.05, n0=1.0)
    assert_true(r["power_mw"][-1] < r["power_mw"][0], "Power decreases")
    assert_true(r["T_fuel"][-1] < r["T_fuel"][0], "Fuel temperature decreases")
    assert_true(r["void_fraction"][-1] < r["void_fraction"][0], "Void fraction decreases")
    assert_true(r["power_mw"][-1] > 0, "Power stays positive")


def test_void_feedback_is_negative():
    print("\n[Test 4] BWR void coefficient is negative & dominant")
    m, _ = make_model()
    # More void -> less reactivity.
    rho_more_void = m.reactivity(0.0, m.T_fuel_ref, m.T_cool_ref, m.void_ref + 0.1)
    rho_less_void = m.reactivity(0.0, m.T_fuel_ref, m.T_cool_ref, m.void_ref - 0.1)
    assert_true(rho_more_void < rho_less_void, "Higher void -> lower reactivity (alpha_void < 0)")
    assert_true(m.alpha_void < 0, f"alpha_void={m.alpha_void:.2e} < 0")


def test_doppler_feedback_is_negative():
    print("\n[Test 5] Doppler (fuel-temperature) coefficient is negative")
    m, _ = make_model()
    rho_hot = m.reactivity(0.0, m.T_fuel_ref + 100.0, m.T_cool_ref, m.void_ref)
    rho_cold = m.reactivity(0.0, m.T_fuel_ref - 100.0, m.T_cool_ref, m.void_ref)
    assert_true(rho_hot < rho_cold, "Hotter fuel -> lower reactivity (Doppler < 0)")
    assert_true(m.alpha_D < 0, f"alpha_doppler={m.alpha_D:.2e} < 0")


def test_prompt_jump():
    print("\n[Test 6] Sub-prompt-critical step shows fast prompt jump")
    m, _ = make_model()
    r = m.simulate(m.dollars_to_reactivity(0.5), duration_s=20.0, dt=0.005, n0=1.0)
    # Within the first ~0.2 s the power should rise sharply (prompt jump),
    # far faster than the delayed-neutron timescale.
    idx_fast = np.argmin(np.abs(r["t"] - 0.2))
    rise = r["power_mw"][idx_fast] / r["power_mw"][0]
    assert_true(rise > 1.1, f"Prompt jump within 0.2 s: power x{rise:.2f}")
    # And it stays finite/bounded (below prompt critical = 1$, no divergence).
    assert_true(np.all(np.isfinite(r["power_mw"])), "All power values finite")


def test_precursor_equilibrium():
    print("\n[Test 7] Delayed-neutron precursors at correct equilibrium")
    m, _ = make_model()
    y0 = m.steady_state(1.0)
    C0 = y0[1:1 + m.n_groups]
    # Equilibrium: lambda_i C_i = (beta_i/Lambda) n  ->  C_i = beta_i/(Lambda lambda_i)
    C_expected = m.beta_i / (m.Lambda * m.lam_i)
    err = np.max(np.abs(C0 - C_expected) / C_expected)
    assert_true(err < 1e-9, f"Precursor equilibrium exact (max rel err {err:.1e})")
    assert_true(m.n_groups == 6, "Six delayed-neutron groups")


def test_energy_balance():
    print("\n[Test 8] Steady-state energy conservation: fission power = heat removed")
    m, _ = make_model()
    r = m.simulate(0.0, duration_s=60.0, dt=0.5, n0=1.0)
    P_fis = r["power_mw"][-1] * 1e6                       # W
    Tf, Tc = r["T_fuel"][-1], r["T_coolant"][-1]
    Q_fc = m.hA * (Tf - Tc)                               # fuel -> coolant
    Q_cs = m.W_cool * (Tc - m.T_sink)                     # coolant -> sink
    assert_true(abs(P_fis - Q_fc) / P_fis < 0.02, f"P_fis~Q_fuel->cool (err {abs(P_fis-Q_fc)/P_fis*100:.2f}%)")
    assert_true(abs(P_fis - Q_cs) / P_fis < 0.02, f"P_fis~Q_cool->sink (err {abs(P_fis-Q_cs)/P_fis*100:.2f}%)")


def test_void_in_range():
    print("\n[Test 9] Void fraction stays within physical [0, 0.95]")
    m, _ = make_model()
    for d in [-2.0, -1.0, 0.0, 0.4]:
        r = m.simulate(m.dollars_to_reactivity(d), duration_s=60.0, dt=0.1, n0=1.0)
        v = r["void_fraction"]
        assert_true(np.all(v >= -1e-6) and np.all(v <= 0.95 + 1e-6),
                    f"d={d}$: void in [0,0.95], range [{v.min():.3f},{v.max():.3f}]")


def test_temperatures_physical():
    print("\n[Test 10] Fuel/coolant temperatures physical & T_fuel > T_coolant")
    m, _ = make_model()
    r = m.simulate(m.dollars_to_reactivity(0.2), duration_s=80.0, dt=0.1, n0=1.0)
    assert_true(np.all(r["T_fuel"] > r["T_coolant"]), "T_fuel > T_coolant (heat flows fuel->coolant)")
    assert_true(np.all(r["T_fuel"] < 2800.0), "T_fuel below UO2 melting (~3100 K) with margin")
    assert_true(np.all(r["T_coolant"] > 450.0), "T_coolant above 450 K")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + get_info()")
    _, cm = make_model()
    r = cm.predict({"reactivity_dollars": 0.1, "duration_s": 10.0, "dt": 0.1})
    for key in ["t", "n", "power_mw", "T_fuel", "T_coolant",
                "void_fraction", "reactivity", "precursors"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["power_mw"]), "Time-series arrays same length")
    assert_true(r["precursors"].shape[0] == 6, "Precursors array has 6 groups")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC117", "get_info component_id == EC117")
    assert_true("Duderstadt" in info["source"] and "Lamarsh" in info["source"],
                "Literature cited in source")


def test_benchmark():
    print("\n[Test 12] Benchmark: 100 s transient at dt=0.05")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(m.dollars_to_reactivity(0.3), duration_s=100.0, dt=0.05, n0=1.0)
    elapsed = time.perf_counter() - t0
    print(f"  100 s transient in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_steady_state_holds,
        test_positive_reactivity_feedback_stabilises,
        test_negative_reactivity_drops_power,
        test_void_feedback_is_negative,
        test_doppler_feedback_is_negative,
        test_prompt_jump,
        test_precursor_equilibrium,
        test_energy_balance,
        test_void_in_range,
        test_temperatures_physical,
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
    print(f"EC117 BWR F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
