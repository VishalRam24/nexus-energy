"""
EC082 -- Ice Thermal Storage -- F2a Stefan-Problem Enthalpy Model
Test suite: enthalpy-closure correctness, energy conservation, Stefan UA
monotonicity, phase-change temperature pinning, edge cases, predict() interface,
and a benchmark timing test.  Custom harness (NO pytest).
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import IceTES_F2a
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
def test_enthalpy_closure():
    print("\n[Test 1] Enthalpy closure: ice fraction and T at known states")
    m, _ = make_model()
    # All liquid at T_f
    assert_true(abs(m.ice_fraction(0.0) - 0.0) < 1e-12, "H=0 -> f_ice=0")
    assert_true(abs(m.temperature(0.0) - m.T_f) < 1e-9, "H=0 -> T=0 C")
    # All ice at T_f
    assert_true(abs(m.ice_fraction(-m.L) - 1.0) < 1e-12, "H=-L -> f_ice=1")
    assert_true(abs(m.temperature(-m.L) - m.T_f) < 1e-9, "H=-L -> T=0 C")
    # Half ice
    assert_true(abs(m.ice_fraction(-0.5 * m.L) - 0.5) < 1e-12, "H=-L/2 -> f_ice=0.5")


def test_ice_fraction_bounds():
    print("\n[Test 2] Ice fraction always in [0, 1]")
    m, _ = make_model()
    for H in [5e9, 0.0, -0.3 * m.L, -m.L, -2.0 * m.L, 1e10]:
        f = float(m.ice_fraction(H))
        assert_true(0.0 <= f <= 1.0, f"H={H:.2e} -> f_ice={f:.4f} in [0,1]")


def test_phase_change_temp_pinned():
    print("\n[Test 3] Temperature pinned at 0 C throughout phase change")
    m, _ = make_model()
    for f in np.linspace(0.01, 0.99, 20):
        H = -f * m.L
        T = float(m.temperature(H))
        assert_true(abs(T - m.T_f) < 1e-9, f"f_ice={f:.2f} -> T={T:.2e} C (pinned)")


def test_sensible_regions():
    print("\n[Test 4] Sensible branches: liquid warms above 0, ice cools below 0")
    m, _ = make_model()
    # liquid: H>0 gives T>0
    T_liq = float(m.temperature(m.m_water * m.cp_w * 5.0))  # +5 C worth
    assert_true(abs(T_liq - 5.0) < 1e-6, f"liquid sensible T={T_liq:.3f} ~ 5 C")
    # solid: H<-L gives T<0
    T_sol = float(m.temperature(-m.L - m.m_water * m.cp_i * 8.0))  # -8 C
    assert_true(abs(T_sol + 8.0) < 1e-6, f"solid sensible T={T_sol:.3f} ~ -8 C")


def test_stefan_UA_decreases():
    print("\n[Test 5] Stefan effect: UA decreases monotonically as ice grows")
    m, _ = make_model()
    f_vals = np.linspace(0.0, 1.0, 30)
    UA = [m.UA_effective(f) for f in f_vals]
    assert_true(abs(UA[0] - m.UA_clean) < 1e-6, "UA(f=0) = clean-coil UA")
    for i in range(1, len(UA)):
        assert_true(UA[i] <= UA[i - 1] + 1e-9,
                    f"UA({f_vals[i]:.2f})={UA[i]:.1f} <= UA_prev={UA[i-1]:.1f}")
    assert_true(UA[-1] < UA[0], f"UA(full ice)={UA[-1]:.1f} < UA(clean)={UA[0]:.1f}")


def test_charging_builds_ice():
    print("\n[Test 6] Charging with cold brine builds ice (SOC increases)")
    m, _ = make_model()
    r = m.simulate(T_brine=-6.0, T_amb=20.0, ice_fraction0=0.0,
                   dt=600.0, duration_s=8 * 3600.0)
    assert_true(r["success"], "integrator succeeded")
    assert_true(r["ice_fraction"][-1] > r["ice_fraction"][0], "ice fraction grew")
    assert_true(r["ice_fraction"][-1] > 0.9, f"nearly full: {r['ice_fraction'][-1]:.3f}")
    # charge power must fall as ice grows (Stefan)
    qc_early = r["q_coil_W"][2]
    qc_late = r["q_coil_W"][-2]
    assert_true(qc_late < qc_early, f"charge power decays {qc_early/1e3:.1f}->{qc_late/1e3:.1f} kW")


def test_discharge_melts_ice():
    print("\n[Test 7] Discharge with warm brine melts ice and delivers cooling")
    m, _ = make_model()
    r = m.simulate(T_brine=10.0, T_amb=20.0, ice_fraction0=1.0,
                   dt=600.0, duration_s=8 * 3600.0)
    assert_true(r["ice_fraction"][-1] < r["ice_fraction"][0], "ice fraction shrank")
    assert_true(np.all(r["cooling_power_W"] >= -1e-9), "cooling power >= 0")
    cooling_kwh = np.trapezoid(r["cooling_power_W"], r["t"]) / 3.6e6
    assert_true(cooling_kwh > 400.0, f"delivered {cooling_kwh:.1f} kWh cooling (> latent floor)")


def test_energy_conservation():
    print("\n[Test 8] Energy conservation: dH = -(integral of q_coil + q_loss)")
    m, _ = make_model()
    r = m.simulate(T_brine=-6.0, T_amb=15.0, ice_fraction0=0.2,
                   dt=300.0, duration_s=4 * 3600.0)
    dH = r["enthalpy_J"][-1] - r["enthalpy_J"][0]
    flux_integral = -np.trapezoid(r["q_coil_W"] + r["q_loss_W"], r["t"])
    rel_err = abs(dH - flux_integral) / (abs(dH) + 1.0)
    assert_true(rel_err < 0.02,
                f"dH={dH/3.6e6:.2f} kWh vs flux={flux_integral/3.6e6:.2f} kWh, rel_err={rel_err:.4f}")


def test_idle_ambient_self_discharge():
    print("\n[Test 9] Idle at T_f: ambient ingress slowly melts ice (self-discharge)")
    m, _ = make_model()
    # brine at 0 C so coil is inert; ambient warm -> q_loss>0 melts ice.
    r = m.simulate(T_brine=0.0, T_amb=25.0, ice_fraction0=0.8,
                   dt=600.0, duration_s=12 * 3600.0)
    assert_true(r["ice_fraction"][-1] < 0.8, "ice melts from ambient ingress")
    assert_true(r["ice_fraction"][-1] > 0.0, "but does not vanish in 12 h (slow loss)")
    # Convention q_loss = UA_amb*(T_water - T_amb): warm ambient (T_w<T_amb)
    # gives q_loss<0, i.e. heat flows INTO the tank and -q_loss melts ice.
    assert_true(np.all(r["q_loss_W"] < 0), "q_loss < 0 (heat ingress from warm ambient)")


def test_no_charge_past_full():
    print("\n[Test 10] Cannot freeze past full ice (f_ice clamped at 1)")
    m, _ = make_model()
    r = m.simulate(T_brine=-10.0, T_amb=0.0, ice_fraction0=0.95,
                   dt=300.0, duration_s=12 * 3600.0)
    assert_true(np.all(r["ice_fraction"] <= 1.0 + 1e-9), "ice fraction never exceeds 1")
    assert_true(r["ice_fraction"][-1] >= 0.99, f"reaches full: {r['ice_fraction'][-1]:.4f}")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + get_info()")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC082", "component_id is EC082")
    r = cm.predict({"T_brine_C": -6.0, "dt": 600.0, "duration_s": 3600.0})
    for key in ["t", "ice_fraction", "temperature_C", "UA_eff_W_per_K",
                "q_coil_W", "cooling_power_W", "energy_stored_kwh"]:
        assert_true(key in r, f"predict output has '{key}'")
    assert_true(len(r["t"]) == len(r["ice_fraction"]), "arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 24 h charge+discharge sim at dt=60 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(T_brine=lambda t: -6.0 if t < 12 * 3600 else 10.0,
               T_amb=20.0, ice_fraction0=0.0, dt=60.0, duration_s=24 * 3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  24 h simulation (1440 steps) in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_enthalpy_closure,
        test_ice_fraction_bounds,
        test_phase_change_temp_pinned,
        test_sensible_regions,
        test_stefan_UA_decreases,
        test_charging_builds_ice,
        test_discharge_melts_ice,
        test_energy_conservation,
        test_idle_ambient_self_discharge,
        test_no_charge_past_full,
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
    print(f"EC082 Ice TES F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
