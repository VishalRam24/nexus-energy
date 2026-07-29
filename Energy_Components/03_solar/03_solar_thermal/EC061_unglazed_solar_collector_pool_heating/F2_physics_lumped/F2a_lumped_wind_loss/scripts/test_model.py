"""
EC061 — Unglazed Solar Collector (Pool Heating) — F2a Physics-Lumped
Test suite: energy conservation, wind sensitivity, efficiency curve, edge cases.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import UnglazedCollectorF2a
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
def test_loss_increases_with_wind():
    print("\n[Test 1] Loss coefficient U_L rises with wind speed")
    m, _ = make_model()
    U0 = m.loss_coefficient(0.0)
    U5 = m.loss_coefficient(5.0)
    U10 = m.loss_coefficient(10.0)
    assert_true(U5 > U0, f"U_L(5)={U5:.2f} > U_L(0)={U0:.2f}")
    assert_true(U10 > U5, f"U_L(10)={U10:.2f} > U_L(5)={U5:.2f}")
    # unglazed has a high loss coefficient
    assert_true(U10 > 20.0, f"Unglazed loss is large at wind: U_L(10)={U10:.2f}")


def test_optical_efficiency_high_and_wind_degraded():
    print("\n[Test 2] eta0 high (no glazing) and degraded by wind")
    m, _ = make_model()
    e0 = m.optical_efficiency(0.0)
    e5 = m.optical_efficiency(5.0)
    assert_true(e0 >= 0.85, f"eta0_eff(0)={e0:.3f} high (no glass reflection)")
    assert_true(e5 < e0, f"eta0_eff(5)={e5:.3f} < eta0_eff(0)={e0:.3f}")


def test_zero_irradiance_no_useful_heat():
    print("\n[Test 3] Q = 0 at night (G = 0)")
    m, _ = make_model()
    r = m.simulate(G=0.0, Ta=18.0, u_wind=2.0, Tf_in=22.0,
                   Tp0=22.0, dt=60.0, duration_s=1800.0)
    assert_true(np.all(r["q_use"] >= -1e-9), "q_use never negative")
    assert_true(r["Q_use_W"][-1] < 1e-6, f"No useful heat at night: Q={r['Q_use_W'][-1]:.3e} W")
    assert_true(r["eta"][-1] == 0.0, "Efficiency = 0 with no sun")


def test_plate_cools_below_ambient_at_night():
    print("\n[Test 4] At night plate cools toward/below ambient (radiative loss)")
    m, _ = make_model()
    r = m.simulate(G=0.0, Ta=18.0, Tsky=8.0, u_wind=0.5, Tf_in=18.0,
                   Tp0=30.0, dt=60.0, duration_s=3600.0)
    assert_true(r["T_plate"][-1] < r["T_plate"][0],
                f"Plate cools: {r['T_plate'][-1]:.2f} < {r['T_plate'][0]:.2f} degC")
    # radiative sky loss can push slightly below ambient
    assert_true(r["T_plate"][-1] <= 18.0 + 0.5,
                f"Plate near/below ambient at night: {r['T_plate'][-1]:.2f} degC")


def test_steady_state_reached():
    print("\n[Test 5] Dynamic ODE converges to steady state")
    m, _ = make_model()
    r = m.simulate(G=800.0, Ta=22.0, u_wind=1.0, Tf_in=24.0,
                   Tp0=22.0, dt=30.0, duration_s=3600.0)
    dT = abs(r["T_plate"][-1] - r["T_plate"][-2])
    assert_true(dT < 0.05, f"Near SS: dT={dT:.4f} K between last two steps")
    # matches the analytic steady-state root
    Tss = m.steady_state(800.0, Ta_C=22.0, u_wind=1.0, Tf_in_C=24.0)
    assert_true(abs(r["T_plate"][-1] - Tss) < 0.5,
                f"ODE final {r['T_plate'][-1]:.2f} ~ root {Tss:.2f} degC")


def test_energy_conservation_steady_state():
    print("\n[Test 6] Energy balance closes at steady state (net flux = 0)")
    m, _ = make_model()
    Tss = m.steady_state(700.0, Ta_C=20.0, u_wind=2.0, Tf_in_C=22.0)
    flux = m._net_flux(Tss, 700.0, 20.0, 14.0, 2.0, 22.0)
    assert_true(abs(flux) < 1e-3, f"Net flux at SS ~ 0: {flux:.3e} W/m2")


def test_efficiency_drops_with_reduced_temperature():
    print("\n[Test 7] eta falls sharply with (Tm-Ta)/G")
    m, _ = make_model()
    eta = []
    for Tfin in [20.0, 26.0, 32.0, 38.0]:
        r = m.simulate(G=800.0, Ta=20.0, u_wind=1.0, Tf_in=Tfin,
                       Tp0=20.0, dt=60.0, duration_s=3600.0)
        eta.append(r["eta"][-1])
    for i in range(1, len(eta)):
        assert_true(eta[i] <= eta[i - 1] + 1e-6,
                    f"eta decreasing as Tf_in rises: {eta[i]:.3f} <= {eta[i-1]:.3f}")
    assert_true(eta[0] > eta[-1] + 0.05, f"Significant drop: {eta[0]:.3f} -> {eta[-1]:.3f}")


def test_efficiency_drops_with_wind():
    print("\n[Test 8] eta falls with wind speed")
    m, _ = make_model()
    r_calm = m.simulate(G=800.0, Ta=20.0, u_wind=0.0, Tf_in=28.0,
                        Tp0=20.0, dt=60.0, duration_s=3600.0)
    r_windy = m.simulate(G=800.0, Ta=20.0, u_wind=8.0, Tf_in=28.0,
                         Tp0=20.0, dt=60.0, duration_s=3600.0)
    assert_true(r_windy["eta"][-1] < r_calm["eta"][-1],
                f"Windy eta={r_windy['eta'][-1]:.3f} < calm eta={r_calm['eta'][-1]:.3f}")


def test_efficiency_bounds():
    print("\n[Test 9] Efficiency stays in [0, 1]")
    m, _ = make_model()
    for u in [0.0, 3.0, 8.0]:
        for Tfin in [18.0, 25.0, 35.0]:
            r = m.simulate(G=900.0, Ta=20.0, u_wind=u, Tf_in=Tfin,
                           Tp0=20.0, dt=120.0, duration_s=2400.0)
            assert_true(np.all((r["eta"] >= 0.0) & (r["eta"] <= 1.0)),
                        f"eta in [0,1] for u={u}, Tf_in={Tfin}")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(cm.component_id == "EC061", "component_id == EC061")
    r = cm.predict({"G": 800.0, "Ta": 22.0, "u_wind": 1.0, "Tf_in": 24.0,
                    "dt": 60.0, "duration_s": 1200.0})
    for key in ["t", "T_plate", "q_use", "Q_use_W", "eta", "U_L", "eta0_eff"]:
        assert_true(key in r, f"Output key '{key}' present")
    assert_true(len(r["t"]) == len(r["T_plate"]) == len(r["eta"]),
                "Output arrays same length")


def test_time_varying_inputs():
    print("\n[Test 11] Callable (time-varying) irradiance — diurnal cycle")
    m, _ = make_model()
    def G_day(t):
        # half-sine day, dark at start/end
        frac = t / (12 * 3600.0)
        return max(0.0, 1000.0 * np.sin(np.pi * frac))
    r = m.simulate(G=G_day, Ta=20.0, u_wind=1.0, Tf_in=24.0,
                   Tp0=20.0, dt=600.0, duration_s=12 * 3600.0)
    assert_true(r["q_use"][0] < r["q_use"][len(r["t"]) // 2],
                "Useful heat rises from dawn toward midday")
    assert_true(r["q_use"][-1] < r["q_use"][len(r["t"]) // 2],
                "Useful heat falls toward dusk")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h sim at dt=10 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(G=800.0, Ta=22.0, u_wind=2.0, Tf_in=24.0,
               Tp0=22.0, dt=10.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  1 h simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_loss_increases_with_wind,
        test_optical_efficiency_high_and_wind_degraded,
        test_zero_irradiance_no_useful_heat,
        test_plate_cools_below_ambient_at_night,
        test_steady_state_reached,
        test_energy_conservation_steady_state,
        test_efficiency_drops_with_reduced_temperature,
        test_efficiency_drops_with_wind,
        test_efficiency_bounds,
        test_predict_interface,
        test_time_varying_inputs,
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
    print(f"EC061 Unglazed Collector F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
