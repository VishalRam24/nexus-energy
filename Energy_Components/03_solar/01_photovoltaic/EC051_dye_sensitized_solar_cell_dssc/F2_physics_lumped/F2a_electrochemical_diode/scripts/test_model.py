"""
EC051 -- Dye-Sensitized Solar Cell (DSSC) -- F2a Physics-Lumped
Test suite: physics sanity (P=0 at G=0, Isc ~ G, monotone P-V, eta bounds),
thermal ODE behaviour, edge cases, predict() interface, and a benchmark.

Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import DSSC_F2a
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


# --------------------------------------------------------------------------- #
def test_zero_irradiance_zero_power():
    print("\n[Test 1] P = 0 at G = 0 (dark cell)")
    m, _ = make_model()
    r = m.iv_curve(0.0, 298.15)
    assert_true(r["Pmp_W"] == 0.0, f"Pmp={r['Pmp_W']:.3e} W == 0")
    assert_true(r["Isc_A"] == 0.0, f"Isc={r['Isc_A']:.3e} A == 0")
    assert_true(r["eta"] == 0.0, f"eta={r['eta']:.3e} == 0")


def test_voc_around_0p7():
    print("\n[Test 2] Voc ~ 0.7 V at 1 sun (DSSC signature)")
    m, _ = make_model()
    r = m.iv_curve(1000.0, 298.15)
    assert_true(0.55 < r["Voc_V"] < 0.85, f"Voc={r['Voc_V']:.3f} V in (0.55, 0.85)")


def test_efficiency_bounds():
    print("\n[Test 3] 0 < eff < 0.13 across irradiance")
    m, _ = make_model()
    for G in [50.0, 200.0, 600.0, 1000.0, 1200.0]:
        r = m.iv_curve(G, 298.15)
        assert_true(0.0 < r["eta"] < 0.13, f"G={G:.0f}: eta={r['eta']*100:.2f}% in (0, 13)%")


def test_isc_proportional_to_G():
    print("\n[Test 4] Isc increases ~monotonically with G (Isc ~ G)")
    m, _ = make_model()
    Gs = [100.0, 300.0, 600.0, 1000.0]
    Iscs = [m.iv_curve(G, 298.15)["Isc_A"] for G in Gs]
    for a, b in zip(Iscs, Iscs[1:]):
        assert_true(b > a, f"Isc rises: {b*1e3:.2f} mA > {a*1e3:.2f} mA")
    # Near-linear at low/moderate G (below diffusion saturation).
    ratio = m.iv_curve(500.0, 298.15)["Isc_A"] / m.iv_curve(250.0, 298.15)["Isc_A"]
    assert_true(1.7 < ratio < 2.3, f"Isc(500)/Isc(250)={ratio:.2f} ~ 2 (near-linear)")


def test_pv_monotone_to_mpp():
    print("\n[Test 5] P-V curve monotonically rises to MPP then falls")
    m, _ = make_model()
    r = m.iv_curve(1000.0, 298.15)
    P = r["P"]
    idx = int(np.argmax(P))
    rising = np.all(np.diff(P[:idx + 1]) >= -1e-9)
    falling = np.all(np.diff(P[idx:]) <= 1e-9)
    assert_true(rising, "P rises monotonically up to MPP")
    assert_true(falling, "P falls monotonically after MPP")
    assert_true(0.0 < r["Vmp_V"] < r["Voc_V"], f"0 < Vmp({r['Vmp_V']:.3f}) < Voc({r['Voc_V']:.3f})")


def test_fill_factor_physical():
    print("\n[Test 6] Fill factor in physical (0, 1) range")
    m, _ = make_model()
    r = m.iv_curve(1000.0, 298.15)
    assert_true(0.4 < r["FF"] < 0.85, f"FF={r['FF']:.3f} in (0.4, 0.85)")


def test_voc_low_light_robust():
    print("\n[Test 7] Good low-light: Voc falls only mildly from 1000 -> 100 W/m2")
    m, _ = make_model()
    voc_full = m.iv_curve(1000.0, 298.15)["Voc_V"]
    voc_low = m.iv_curve(100.0, 298.15)["Voc_V"]
    drop = voc_full - voc_low
    assert_true(voc_low > 0.45, f"Voc(100 W/m2)={voc_low:.3f} V still high")
    assert_true(0.0 < drop < 0.20, f"Voc drop {drop*1e3:.0f} mV is small (log-like)")


def test_voc_decreases_with_temperature():
    print("\n[Test 8] Voc decreases with cell temperature (recombination up)")
    m, _ = make_model()
    voc_cold = m.iv_curve(1000.0, 283.15)["Voc_V"]
    voc_hot = m.iv_curve(1000.0, 323.15)["Voc_V"]
    assert_true(voc_hot < voc_cold, f"Voc(50C)={voc_hot:.3f} < Voc(10C)={voc_cold:.3f}")


def test_thermal_ode_warms_up():
    print("\n[Test 9] Thermal ODE: illuminated cell warms above ambient and settles")
    m, _ = make_model()
    # Thermal time constant tau = m*cp/hA ~ 450 s; integrate to ~8 tau for true SS.
    r = m.simulate(1000.0, T0=298.15, T_amb=298.15, dt=10.0, duration_s=4000.0)
    assert_true(r["temperature"][-1] > 298.15, f"T_final={r['temperature'][-1]:.2f} > T_amb")
    assert_true(r["temperature"][-1] < 360.0, f"T_final={r['temperature'][-1]:.2f} < 360 K (bounded)")
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.5, f"Near steady state: dT={dT:.4f} K between last steps")
    # Energy conservation at SS: |Q_abs - P_elec - Q_loss| ~ 0.
    Tf = r["temperature"][-1]
    iv = m.iv_curve(1000.0, Tf)
    Q_abs = m.absorptance * 1000.0 * m.A * 1e-4
    Q_loss = m.hA_loss * (Tf - 298.15)
    resid = abs(Q_abs - iv["Pmp_W"] - Q_loss)
    # Residual = m*cp*dT/dt still unbalanced; require << absorbed power.
    assert_true(resid < 0.05 * Q_abs, f"SS energy balance residual={resid:.2e} W << Q_abs={Q_abs:.3e} W")


def test_dark_simulation_stays_at_ambient():
    print("\n[Test 10] G=0 over time: no heating, P stays 0")
    m, _ = make_model()
    r = m.simulate(0.0, T0=298.15, T_amb=298.15, dt=10.0, duration_s=300.0)
    assert_true(np.allclose(r["temperature"], 298.15, atol=1e-3), "T stays at ambient in dark")
    assert_true(np.all(r["Pmp"] == 0.0), "Power stays 0 in dark")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + get_info()")
    _, cm = make_model()
    r = cm.predict({"irradiance_W_m2": 800.0, "dt": 10.0, "duration_s": 100.0})
    for key in ["t", "temperature", "Voc", "Isc", "Pmp", "efficiency", "iv_curve",
                "Voc_V", "Isc_A", "Pmp_W", "FF"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["temperature"]) == len(r["Pmp"]), "Arrays same length")
    ss = cm.predict({"steady_state": True, "irradiance_W_m2": 1000.0})
    assert_true(0.0 < ss["efficiency"] < 0.13, f"steady_state eta={ss['efficiency']*100:.2f}%")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC051" and info["version"] == "1.0.0", "get_info metadata")


def test_benchmark():
    print("\n[Test 12] Benchmark: 600s thermal sim at dt=2")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(1000.0, T0=298.15, T_amb=298.15, dt=2.0, duration_s=600.0)
    elapsed = time.perf_counter() - t0
    print(f"  600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_zero_irradiance_zero_power,
        test_voc_around_0p7,
        test_efficiency_bounds,
        test_isc_proportional_to_G,
        test_pv_monotone_to_mpp,
        test_fill_factor_physical,
        test_voc_low_light_robust,
        test_voc_decreases_with_temperature,
        test_thermal_ode_warms_up,
        test_dark_simulation_stays_at_ambient,
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

    print(f"\n{'=' * 62}")
    print(f"EC051 DSSC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'=' * 62}")
    sys.exit(0 if failed == 0 else 1)
