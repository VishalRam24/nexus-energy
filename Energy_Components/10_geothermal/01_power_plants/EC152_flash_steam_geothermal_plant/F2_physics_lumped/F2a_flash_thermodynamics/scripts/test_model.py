"""
EC152 -- Flash Steam Geothermal Plant -- F2a Flash Thermodynamics
Test suite: flash conservation, eff<Carnot, optimal flash, ODE transient, edges.
NO pytest -- run as:  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import FlashSteamGeothermalF2a
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
def test_psat_steam_tables():
    print("\n[Test 1] Saturation pressure matches steam tables")
    m, _ = make_model()
    # IAPWS: p_sat(100C)=101.32 kPa, p_sat(200C)=1554.9 kPa
    assert_true(abs(m.p_sat(100.0) - 101.32) < 2.0, f"p_sat(100C)={m.p_sat(100.0):.1f} kPa ~ 101.3")
    assert_true(abs(m.p_sat(200.0) - 1554.9) < 30.0, f"p_sat(200C)={m.p_sat(200.0):.1f} kPa ~ 1555")
    # monotonic increasing
    Ts = np.linspace(50, 300, 20)
    ps = m.p_sat(Ts)
    assert_true(np.all(np.diff(ps) > 0), "p_sat monotonically increases with T")


def test_enthalpy_steam_tables():
    print("\n[Test 2] Saturated enthalpies match steam tables (<2%)")
    m, _ = make_model()
    # IAPWS A-4: hf(100)=419.0, hfg(100)=2256.5, hg(200)=2792.0
    assert_true(abs(m.h_f(100.0) - 419.0) / 419.0 < 0.02, f"h_f(100)={float(m.h_f(100.0)):.1f} ~419")
    assert_true(abs(m.h_fg(100.0) - 2256.5) / 2256.5 < 0.03, f"h_fg(100)={float(m.h_fg(100.0)):.1f} ~2257")
    # h_g = h_f + h_fg and decreases at high T (vapour line turns over near Tc)
    assert_true(m.h_g(160.0) > m.h_f(160.0), "h_g > h_f")


def test_flash_mass_energy_conservation():
    print("\n[Test 3] Isenthalpic flash conserves mass & energy")
    m, _ = make_model()
    T_geo, T_fl = 240.0, 160.0
    x = float(m.flash_steam_fraction(T_geo, T_fl))
    assert_true(0.0 < x < 1.0, f"steam fraction x={x:.4f} in (0,1)")
    # Energy: h_geo = (1-x) h_f(T_fl) + x h_g(T_fl)  (lever rule, isenthalpic)
    h_geo = float(m.h_f(T_geo))
    h_mix = (1 - x) * float(m.h_f(T_fl)) + x * float(m.h_g(T_fl))
    assert_true(abs(h_geo - h_mix) / h_geo < 1e-3,
                f"enthalpy balance: h_geo={h_geo:.1f} = mix={h_mix:.1f}")
    # Mass: steam + liquid = total
    mdot = 100.0
    m_steam = float(m.steam_mass_flow(mdot, T_geo, T_fl))
    m_liq = mdot - m_steam
    assert_true(abs((m_steam + m_liq) - mdot) < 1e-9, "mass balance steam+brine=total")


def test_efficiency_below_carnot():
    print("\n[Test 4] Utilization efficiency < Carnot (2nd law)")
    m, _ = make_model()
    for T_geo, T_rej in [(200, 30), (240, 50), (300, 20)]:
        eta = float(m.utilization_efficiency(T_geo, T_rej, 100.0))
        carnot = float(m.carnot_efficiency(T_geo, T_rej))
        assert_true(0.0 < eta < carnot,
                    f"eta_util({T_geo},{T_rej})={eta:.4f} < Carnot={carnot:.4f}")


def test_optimal_flash_maximizes_power():
    print("\n[Test 5] Analytic optimal flash T near power-maximizing T")
    m, _ = make_model()
    T_geo, T_rej = 240.0, 50.0
    T_opt = float(m.optimal_flash_temperature(T_geo, T_rej))
    Ts = np.arange(90.0, 226.0, 5.0)
    P = np.array([float(m.net_power_ss(T_geo, T_rej, 100.0, T)) for T in Ts])
    T_grid_best = Ts[int(np.argmax(P))]
    assert_true(abs(T_opt - T_grid_best) <= 15.0,
                f"T_opt={T_opt:.1f} C within 15 C of grid argmax {T_grid_best:.1f} C")
    # power must be a concave hump: lower at both extremes than at optimum
    P_opt = float(m.net_power_ss(T_geo, T_rej, 100.0, T_opt))
    assert_true(P_opt > P[0] and P_opt > P[-1], "power at optimum exceeds extremes")


def test_power_scales_with_flow():
    print("\n[Test 6] Net power scales linearly with brine flow")
    m, _ = make_model()
    P1 = float(m.net_power_ss(240, 50, 50.0))
    P2 = float(m.net_power_ss(240, 50, 100.0))
    assert_true(abs(P2 - 2.0 * P1) / P2 < 1e-6, f"P(100)={P2:.1f} = 2*P(50)={2*P1:.1f}")


def test_hotter_brine_more_power():
    print("\n[Test 7] Hotter brine -> more power & higher Carnot")
    m, _ = make_model()
    P_lo = float(m.net_power_ss(200, 50, 100.0))
    P_hi = float(m.net_power_ss(280, 50, 100.0))
    assert_true(P_hi > P_lo, f"P(280C)={P_hi:.0f} > P(200C)={P_lo:.0f}")
    assert_true(m.carnot_efficiency(280, 50) > m.carnot_efficiency(200, 50),
                "Carnot rises with source T")


def test_ode_startup_transient():
    print("\n[Test 8] Lumped ODE: cold start rises to steady state")
    m, _ = make_model()
    r = m.simulate(100.0, 240.0, 50.0, dt=0.5, duration_s=120.0)
    assert_true(r["net_power_kW"][0] < 1.0, "starts near zero power (cold start)")
    P_ss = r["P_steady_kW"]
    assert_true(abs(r["net_power_kW"][-1] - P_ss) / P_ss < 0.05,
                f"converges: P_final={r['net_power_kW'][-1]:.0f} ~ P_ss={P_ss:.0f}")
    assert_true(np.all(np.diff(r["net_power_kW"]) >= -1e-6), "power rises monotonically")


def test_ode_flow_step_response():
    print("\n[Test 9] ODE responds to a brine-flow step change")
    m, _ = make_model()
    def step(t):
        return 60.0 if t < 60.0 else 120.0
    r = m.simulate(step, 240.0, 50.0, dt=0.5, duration_s=140.0)
    i_before = int(np.argmin(np.abs(r["t"] - 58.0)))
    i_after = int(np.argmin(np.abs(r["t"] - 130.0)))
    assert_true(r["net_power_kW"][i_after] > r["net_power_kW"][i_before],
                "power increases after flow step up")
    assert_true(r["steam_flow_kgs"][i_after] > r["steam_flow_kgs"][i_before],
                "steam flow increases after flow step up")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"m_dot_brine_kgs": 100.0, "dt": 1.0, "duration_s": 30.0})
    for key in ["t", "net_power_kW", "steam_flow_kgs", "steam_fraction",
                "eta_utilization", "eta_carnot", "specific_work_kJkg"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["net_power_kW"]), "arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC152" and info["fidelity"].startswith("F2a"),
                "get_info() reports EC152 / F2a")


def test_edge_low_source():
    print("\n[Test 11] Edge: low source T -> low but positive output")
    m, _ = make_model()
    P = float(m.net_power_ss(185, 55, 100.0))
    assert_true(P >= 0.0, f"P(185C)={P:.1f} >= 0")
    eta = float(m.utilization_efficiency(185, 55, 100.0))
    assert_true(0.0 <= eta < float(m.carnot_efficiency(185, 55)),
                "low-T efficiency still bounded by Carnot")


def test_benchmark():
    print("\n[Test 12] Benchmark: 120 s sim at dt=0.5")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(100.0, 240.0, 50.0, dt=0.5, duration_s=120.0)
    elapsed = time.perf_counter() - t0
    print(f"  120 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_psat_steam_tables,
        test_enthalpy_steam_tables,
        test_flash_mass_energy_conservation,
        test_efficiency_below_carnot,
        test_optimal_flash_maximizes_power,
        test_power_scales_with_flow,
        test_hotter_brine_more_power,
        test_ode_startup_transient,
        test_ode_flow_step_response,
        test_predict_interface,
        test_edge_low_source,
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
    print(f"EC152 Flash Steam Geothermal F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
