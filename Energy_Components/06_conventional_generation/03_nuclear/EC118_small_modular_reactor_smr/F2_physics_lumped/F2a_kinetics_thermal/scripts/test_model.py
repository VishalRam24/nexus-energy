"""
EC118 -- Small Modular Reactor (SMR) -- F2a Point Kinetics + Lumped Thermal
Test suite: physics sanity (feedback stability, energy conservation,
prompt-jump, natural circulation), load-following, edge cases, interface,
and a benchmark timing test. Custom harness (NO pytest).

Run:  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SMRPointKineticsThermalF2a
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
def test_equilibrium_steady():
    print("\n[Test 1] Rated-power equilibrium is stationary (rho ~ 0)")
    m, _ = make_model()
    r = m.simulate(0.0, 1.0, 60.0)
    assert_true(abs(r["n"][-1] - 1.0) < 1e-3,
                f"n stays at rated: n_final={r['n'][-1]:.5f}")
    assert_true(abs(r["rho"][-1]) < 1e-5,
                f"feedback reactivity ~0 at rated: rho={r['rho'][-1]:.2e}")
    assert_true(abs(r["T_m"][-1] - m.T_m0) < 0.5 and abs(r["T_f"][-1] - m.T_f0) < 1.0,
                "fuel/coolant temps hold at reference equilibrium")


def test_feedback_stabilizes_positive():
    print("\n[Test 2] Positive reactivity insertion is arrested (no runaway)")
    m, _ = make_model()
    r = m.step_reactivity_insertion(0.001, dt=0.5, duration_s=600.0, t_insert=1.0)
    n_peak = r["n"].max()
    n_final = r["n"][-1]
    assert_true(np.isfinite(n_peak) and n_peak < 2.0,
                f"power bounded after +100 pcm: peak={n_peak:.3f}")
    assert_true(n_final > 1.0,
                f"settles above rated: n_final={n_final:.3f}")
    assert_true(n_final <= n_peak + 1e-6,
                "power settles at/below the prompt-jump peak (feedback acted)")
    assert_true(r["rho"][-1] < 0.0010 + 1e-6,
                f"total reactivity driven back toward 0: rho={r['rho'][-1]:.2e}")


def test_feedback_negative_drops_power():
    print("\n[Test 3] Negative reactivity lowers power (monotone direction)")
    m, _ = make_model()
    r = m.step_reactivity_insertion(-0.001, dt=0.5, duration_s=600.0)
    assert_true(r["n"][-1] < 1.0, f"power drops: n_final={r['n'][-1]:.3f}")
    assert_true(r["T_f"][-1] < m.T_f0, "fuel temperature falls at lower power")


def test_prompt_jump():
    print("\n[Test 4] Prompt-jump: small step gives near-instant n bump")
    m, _ = make_model()
    # Sample immediately after the step (within ~tens of ms) before the delayed
    # source and thermal feedback fill in, where the prompt-jump approximation
    # n2/n1 ~ beta/(beta - rho) holds (Duderstadt & Hamilton 1976, sec. 6).
    r = m.step_reactivity_insertion(0.0005, dt=0.005, duration_s=2.0, t_insert=1.0)
    i0 = np.argmin(np.abs(r["t"] - 0.999))
    i1 = np.argmin(np.abs(r["t"] - 1.02))
    expected = m.beta_total / (m.beta_total - 0.0005)
    ratio = r["n"][i1] / r["n"][i0]
    assert_true(ratio > 1.0, f"power jumps up promptly: ratio={ratio:.4f}")
    assert_true(1.0 < ratio <= expected + 1e-3,
                f"bounded by prompt-jump beta/(beta-rho)={expected:.4f}: got {ratio:.4f}")
    assert_true(abs(ratio - expected) < 0.02,
                f"near prompt-jump value: |{ratio:.4f}-{expected:.4f}|<0.02")


def test_precursor_equilibrium():
    print("\n[Test 5] Delayed-precursor groups start in equilibrium")
    m, _ = make_model()
    x0 = m.initial_conditions(1.0)
    d = m.derivatives(0.0, x0, lambda t: 0.0)
    # dC_i/dt should be ~0 at equilibrium
    dC = d[1:1 + m.n_groups]
    assert_true(np.all(np.abs(dC) < 1e-9),
                f"all 6 dC_i/dt ~ 0: max|dC|={np.max(np.abs(dC)):.2e}")
    assert_true(abs(d[0]) < 1e-9, f"dn/dt ~ 0 at equilibrium: {d[0]:.2e}")


def test_energy_conservation():
    print("\n[Test 6] Lumped energy balance closes at steady state")
    m, _ = make_model()
    r = m.simulate(0.0, 1.0, 300.0)
    T_f, T_m = r["T_f"][-1], r["T_m"][-1]
    P = r["n"][-1] * m.P_th
    W = m._flow_capacity(r["n"][-1])
    q_fuel_to_cool = m.hA_fg * (T_f - T_m)
    q_removed = W * (T_m - m.T_in)
    # At steady state fission power = fuel->coolant heat = coolant->sink heat
    assert_true(abs(q_fuel_to_cool - P) / P < 1e-3,
                f"P_fission = h*A*(T_f-T_m): {q_fuel_to_cool/1e6:.2f} vs {P/1e6:.2f} MW")
    assert_true(abs(q_removed - P) / P < 1e-3,
                f"P = W*(T_m-T_in) coolant removal: {q_removed/1e6:.2f} MW")


def test_natural_circulation():
    print("\n[Test 7] Natural-circulation flow scales ~ sqrt(power)")
    m, _ = make_model()
    f_full = m._flow_capacity(1.0) / m.mdot_cp
    f_half = m._flow_capacity(0.5) / m.mdot_cp
    assert_true(abs(f_full - 1.0) < 1e-9, "flow = rated at full power")
    assert_true(abs(f_half - 0.5 ** 0.5) < 1e-9,
                f"flow(0.5)=sqrt(0.5)={0.5**0.5:.3f}: got {f_half:.3f}")
    assert_true(f_half < f_full, "flow decreases with power (buoyancy-driven)")


def test_load_following():
    print("\n[Test 8] Deep load-following 100% -> 60% -> 100% tracks demand")
    _, cm = make_model()
    def sched(t):
        if t < 2000: return 1.0
        if t < 5000: return 0.6
        return 1.0
    r = cm.predict({"mode": "load_follow", "power_schedule": sched,
                    "duration_s": 8000.0, "dt": 10.0})
    for lo, hi, d in [(1500, 2000, 1.0), (4500, 5000, 0.6), (7500, 8000, 1.0)]:
        msk = (r["t"] > lo) & (r["t"] <= hi)
        err = abs(r["n"][msk].mean() - d)
        assert_true(err < 0.03, f"plateau demand={d:.1f}: |error|={err:.3f}")
    # fuel temperature lower at reduced power
    msk_low = (r["t"] > 4500) & (r["t"] <= 5000)
    msk_hi = (r["t"] > 1500) & (r["t"] <= 2000)
    assert_true(r["T_f"][msk_low].mean() < r["T_f"][msk_hi].mean(),
                "fuel hotter at full power than at 60%")


def test_power_fraction_consistency():
    print("\n[Test 9] Equilibrium power-fraction map matches dynamic settle")
    m, _ = make_model()
    rho = -0.002
    n_pred = m.equilibrium_power_fraction(rho)
    r = m.step_reactivity_insertion(rho, dt=1.0, duration_s=1500.0)
    n_dyn = r["n"][-1]
    assert_true(0.2 < n_pred < 1.0, f"predicted eq power frac={n_pred:.3f} in range")
    assert_true(abs(n_pred - n_dyn) < 0.02,
                f"map vs dynamics agree: {n_pred:.3f} vs {n_dyn:.3f}")


def test_temperatures_physical():
    print("\n[Test 10] Temperatures stay physical (T_f > T_m > T_in)")
    m, _ = make_model()
    r = m.step_reactivity_insertion(0.0008, dt=0.5, duration_s=400.0)
    assert_true(np.all(r["T_f"] > r["T_m"]),
                "fuel always hotter than coolant")
    assert_true(np.all(r["T_m"] >= m.T_in - 1.0),
                "coolant never below inlet temperature")
    assert_true(np.all(r["T_f"] < 1600.0),
                f"fuel below melting margin: max T_f={r['T_f'].max():.0f} K")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(cm.component_id == "EC118", "component_id == EC118")
    r = cm.predict({"rho_step": 0.0005, "dt": 1.0, "duration_s": 20.0})
    for key in ["t", "n", "T_f", "T_m", "P_thermal_W", "P_elec_W", "rho"]:
        assert_true(key in r, f"output has '{key}'")
    assert_true(len(r["t"]) == len(r["n"]) == len(r["T_f"]),
                "all time series equal length")
    assert_true(np.all(r["P_elec_W"] <= r["P_thermal_W"] + 1.0),
                "electrical power <= thermal power")


def test_benchmark():
    print("\n[Test 12] Benchmark: 600 s stiff transient")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.step_reactivity_insertion(0.001, dt=0.5, duration_s=600.0)
    elapsed = time.perf_counter() - t0
    print(f"  600 s transient (stiff BDF) in {elapsed*1000:.0f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_equilibrium_steady,
        test_feedback_stabilizes_positive,
        test_feedback_negative_drops_power,
        test_prompt_jump,
        test_precursor_equilibrium,
        test_energy_conservation,
        test_natural_circulation,
        test_load_following,
        test_power_fraction_consistency,
        test_temperatures_physical,
        test_predict_interface,
        test_benchmark,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")

    print(f"\n{'='*64}")
    print(f"EC118 SMR F2a (kinetics+thermal) -- Results: {passed} passed, {failed} failed")
    print(f"{'='*64}")
    sys.exit(0 if failed == 0 else 1)
