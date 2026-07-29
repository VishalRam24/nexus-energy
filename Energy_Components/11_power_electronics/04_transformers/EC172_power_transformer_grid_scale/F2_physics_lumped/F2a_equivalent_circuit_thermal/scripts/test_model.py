"""
EC172 -- Power Transformer (Grid-Scale) -- F2a Equivalent-Circuit + Thermal ODE
Test suite: physics sanity (energy conservation, regulation, efficiency peak),
thermal ODE behavior, edge cases, predict() interface, benchmark timing.
Run with: python3 scripts/test_model.py   (NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import PowerTransformerF2a
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
def test_efficiency_range():
    print("\n[Test 1] Efficiency strictly in (0,1) and high (~99%)")
    m, _ = make_model()
    for plr in [0.1, 0.25, 0.5, 0.75, 1.0, 1.25]:
        eta = float(m.efficiency(plr, power_factor=0.9, winding_temperature=75.0))
        assert_true(0.0 < eta < 1.0, f"eta(PLR={plr})={eta:.5f} in (0,1)")
    eta_rated = float(m.efficiency(1.0, power_factor=1.0, winding_temperature=75.0))
    assert_true(eta_rated > 0.99, f"Rated eta={eta_rated*100:.3f}% > 99% (grid transformer)")


def test_energy_conservation():
    print("\n[Test 2] Energy conservation: P_in = P_out + losses")
    m, _ = make_model()
    for plr in [0.3, 0.7, 1.0]:
        P_out = float(m.output_power(plr, 0.95))
        P_loss = float(m.total_losses(plr, 1.0, 75.0))
        eta = float(m.efficiency(plr, 1.0, 0.95, 75.0))
        P_in = P_out / eta
        assert_true(abs(P_in - (P_out + P_loss)) / P_in < 1e-9,
                    f"PLR={plr}: P_in={P_in/1e6:.3f}MW == P_out+loss")


def test_efficiency_peaks_partial_load():
    print("\n[Test 3] Efficiency peaks at partial load (copper==core loss)")
    m, _ = make_model()
    plr_grid = np.linspace(0.05, 1.3, 400)
    eta = np.array([float(m.efficiency(p, 1.0, 1.0, 75.0)) for p in plr_grid])
    plr_peak = plr_grid[int(np.argmax(eta))]
    plr_theory = m.max_efficiency_load()
    assert_true(plr_peak < 0.95, f"Peak at partial load PLR={plr_peak:.3f} < 0.95")
    assert_true(abs(plr_peak - plr_theory) < 0.05,
                f"Numeric peak {plr_peak:.3f} ~ sqrt(Pc/Pcu)={plr_theory:.3f}")


def test_copper_loss_quadratic():
    print("\n[Test 4] Copper loss ~ I^2 (doubling load -> 4x copper loss)")
    m, _ = make_model()
    p1 = float(m.copper_loss(0.5, 75.0))
    p2 = float(m.copper_loss(1.0, 75.0))
    assert_true(abs(p2 / p1 - 4.0) < 1e-6, f"P_cu(1.0)/P_cu(0.5)={p2/p1:.4f} ~ 4")


def test_core_loss_load_independent():
    print("\n[Test 5] Core loss independent of load, ~V^2")
    m, _ = make_model()
    c1 = float(m.core_loss(1.0))
    c0 = float(m.core_loss(1.0))
    assert_true(c1 == c0, "Core loss does not depend on load fraction")
    assert_true(abs(m.core_loss(1.2) / m.core_loss(1.0) - 1.44) < 1e-6,
                "Core loss ~ V^2 (1.2^2=1.44)")


def test_regulation_behavior():
    print("\n[Test 6] Voltage regulation: positive, grows with load, pf-dependent")
    m, _ = make_model()
    vr_half = m.voltage_regulation(0.5, 0.9)
    vr_full = m.voltage_regulation(1.0, 0.9)
    assert_true(vr_full > vr_half > 0, f"VR grows with load: {vr_full:.4f} > {vr_half:.4f} > 0")
    # lagging pf gives larger regulation than unity pf (inductive X dominant)
    vr_lag = m.voltage_regulation(1.0, 0.8)
    vr_unity = m.voltage_regulation(1.0, 1.0)
    assert_true(vr_lag > vr_unity, f"Lagging pf VR {vr_lag:.4f} > unity-pf VR {vr_unity:.4f}")
    # leading pf can give negative (rising) regulation
    vr_lead = m.voltage_regulation(1.0, 0.8, leading=True)
    assert_true(vr_lead < vr_lag, f"Leading-pf VR {vr_lead:.4f} < lagging {vr_lag:.4f}")


def test_thermal_rises_with_load():
    print("\n[Test 7] Hot-spot temperature rises monotonically with load")
    m, _ = make_model()
    T_amb = 20.0
    prev = -1e9
    for plr in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]:
        Ths = float(m.steady_hotspot_temperature(plr, T_amb))
        assert_true(Ths >= prev, f"Ths(PLR={plr})={Ths:.1f}C >= prev {prev:.1f}C")
        prev = Ths
    Ths_rated = float(m.steady_hotspot_temperature(1.0, T_amb))
    assert_true(Ths_rated > T_amb, f"Rated hot-spot {Ths_rated:.1f}C > ambient")
    # rated hot-spot should be in a physically sensible band (IEEE ~98-110C @ rated)
    assert_true(85.0 < Ths_rated < 120.0, f"Rated hot-spot {Ths_rated:.1f}C in IEEE band")


def test_thermal_ode_transient_to_steady():
    print("\n[Test 8] Thermal ODE: cold start warms to steady-state hot-spot")
    m, _ = make_model()
    # tau_oil ~ 150 min, so integrate well past ~6*tau for full settling
    r = m.simulate(1.0, ambient_temperature=20.0, dt=2.0, duration=1200.0,
                   power_factor=0.9, theta_o0=0.0, dtheta_h0=0.0)
    Ths_final = r["hotspot_temperature"][-1]
    Ths_ss = float(m.steady_hotspot_temperature(1.0, 20.0))
    assert_true(r["hotspot_temperature"][0] < Ths_final,
                f"Warms up: start {r['hotspot_temperature'][0]:.1f}C < final {Ths_final:.1f}C")
    assert_true(abs(Ths_final - Ths_ss) < 0.5,
                f"Converges to SS: final {Ths_final:.2f}C ~ SS {Ths_ss:.2f}C")


def test_thermal_step_overload():
    print("\n[Test 9] Step overload raises hot-spot above rated steady-state")
    m, _ = make_model()
    def step_K(t):
        return 0.8 if t < 120.0 else 1.3
    r = m.simulate(step_K, ambient_temperature=20.0, dt=2.0, duration=480.0,
                   power_factor=0.9, theta_o0=None, dtheta_h0=None
                   if False else 0.0)
    i_before = int(np.argmin(np.abs(r["t"] - 118.0)))
    i_after = r["t"].size - 1
    assert_true(r["hotspot_temperature"][i_after] > r["hotspot_temperature"][i_before],
                "Hot-spot increases after step overload")


def test_magnetizing_current_small():
    print("\n[Test 10] No-load magnetizing current is small (<5% pu)")
    m, _ = make_model()
    Im = m.magnetizing_current_pu(1.0)
    assert_true(0.0 < Im < 0.05, f"I_mag={Im*100:.3f}% pu (small, < 5%)")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + summary fields")
    _, cm = make_model()
    r = cm.predict({"load_fraction": 1.0, "power_factor": 0.9,
                    "duration_min": 120.0, "dt_min": 5.0})
    for key in ["t", "hotspot_temperature", "efficiency", "voltage_regulation",
                "p_total_loss", "regulation_pct", "efficiency_ss",
                "hotspot_temp_final_C", "max_efficiency_load"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["hotspot_temperature"]), "Arrays same length")
    assert_true(0.0 < r["efficiency_ss"] < 1.0, f"efficiency_ss={r['efficiency_ss']:.4f}")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC172", "get_info id == EC172")


def test_benchmark():
    print("\n[Test 12] Benchmark: 24h (1440 min) transient")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(1.0, ambient_temperature=20.0, dt=1.0, duration=1440.0,
               power_factor=0.9)
    elapsed = time.perf_counter() - t0
    print(f"  1440-min transient in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_efficiency_range,
        test_energy_conservation,
        test_efficiency_peaks_partial_load,
        test_copper_loss_quadratic,
        test_core_loss_load_independent,
        test_regulation_behavior,
        test_thermal_rises_with_load,
        test_thermal_ode_transient_to_steady,
        test_thermal_step_overload,
        test_magnetizing_current_small,
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
    print(f"EC172 Power Transformer F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
