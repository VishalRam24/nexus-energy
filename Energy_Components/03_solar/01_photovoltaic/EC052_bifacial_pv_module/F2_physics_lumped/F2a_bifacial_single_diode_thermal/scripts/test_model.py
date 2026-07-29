"""
EC052 -- Bifacial PV Module -- F2a Physics-Lumped
Test suite: physics sanity, Lambert-W single-diode, bifacial gain, thermal ODE.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import BifacialPV_F2a
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
def test_zero_irradiance_zero_power():
    print("\n[Test 1] P = 0 when total irradiance is 0")
    m, _ = make_model()
    r = m.mpp(0.0, 25.0, G_rear=0.0)
    assert_true(abs(r["p_mp"]) < 1e-9, f"P_mp={r['p_mp']:.3e} ~ 0 at G=0")
    assert_true(abs(r["i_sc"]) < 1e-9, f"I_sc={r['i_sc']:.3e} ~ 0 at G=0")
    assert_true(abs(r["v_oc"]) < 1e-9, f"V_oc={r['v_oc']:.3e} ~ 0 at G=0")


def test_bifacial_gain_positive():
    print("\n[Test 2] Bifacial gain > 0 with rear illumination")
    m, _ = make_model()
    g = m.bifacial_gain(800.0, 25.0, albedo=0.5)
    assert_true(g > 0.0, f"bifacial_gain={g*100:.2f}% > 0")
    # no rear -> zero gain
    g0 = m.bifacial_gain(800.0, 25.0, albedo=0.0)
    assert_true(abs(g0) < 1e-6, f"gain={g0:.3e} ~ 0 with no rear light")


def test_gain_increases_with_albedo():
    print("\n[Test 3] Bifacial gain monotonically increases with albedo")
    m, _ = make_model()
    albedos = [0.1, 0.2, 0.4, 0.6, 0.8]
    gains = [m.bifacial_gain(800.0, 25.0, albedo=a) for a in albedos]
    for i in range(1, len(gains)):
        assert_true(gains[i] > gains[i - 1],
                    f"gain({albedos[i]})={gains[i]*100:.2f}% > "
                    f"gain({albedos[i-1]})={gains[i-1]*100:.2f}%")


def test_efficiency_range():
    print("\n[Test 4] Module efficiency in (0, 1) and physically reasonable")
    m, _ = make_model()
    for G in [200.0, 600.0, 1000.0]:
        eta = m.efficiency(G, 25.0, albedo=0.3)
        assert_true(0.0 < eta < 1.0, f"eff(G={G})={eta*100:.2f}% in (0,1)")
        assert_true(0.10 < eta < 0.30,
                    f"eff(G={G})={eta*100:.2f}% in reasonable PV band 10-30%")


def test_isc_proportional_to_Geff():
    print("\n[Test 5] I_sc proportional to effective irradiance")
    m, _ = make_model()
    r1 = m.mpp(500.0, 25.0, G_rear=0.0)
    r2 = m.mpp(1000.0, 25.0, G_rear=0.0)
    ratio = r2["i_sc"] / r1["i_sc"]
    assert_true(abs(ratio - 2.0) < 0.05,
                f"I_sc(1000)/I_sc(500)={ratio:.3f} ~ 2.0 (G doubled)")
    # adding rear raises G_eff and thus I_sc
    r_rear = m.mpp(500.0, 25.0, albedo=0.5)
    assert_true(r_rear["i_sc"] > r1["i_sc"],
                f"I_sc with rear ({r_rear['i_sc']:.3f}) > front-only ({r1['i_sc']:.3f})")


def test_pv_monotone_to_mpp():
    print("\n[Test 6] P-V curve monotonically rises to MPP then falls")
    m, _ = make_model()
    V, I, P = m.iv_curve(1000.0, 25.0, n=300)
    imax = int(np.argmax(P))
    # rising up to MPP
    diffs_up = np.diff(P[: imax + 1])
    assert_true(np.all(diffs_up >= -1e-6),
                f"P monotone increasing up to MPP (idx {imax})")
    # falling after MPP
    diffs_down = np.diff(P[imax:])
    assert_true(np.all(diffs_down <= 1e-6), "P monotone decreasing after MPP")
    # MPP from root-find agrees with curve peak
    r = m.mpp(1000.0, 25.0, G_rear=0.0)
    assert_true(abs(r["p_mp"] - P[imax]) / P[imax] < 0.02,
                f"root-find P_mp={r['p_mp']:.1f}W ~ curve peak {P[imax]:.1f}W")


def test_lambertw_vs_residual():
    print("\n[Test 7] Lambert-W I(V) satisfies the diode equation residual")
    m, _ = make_model()
    I_L, I_o, R_sh, a = m._calc_params(1000.0, 298.15)
    for V in [0.0, 10.0, 20.0, 30.0]:
        I = m.current_from_voltage(V, I_L, I_o, R_sh, a)
        resid = (I_L - I_o * (np.exp((V + I * m.R_s) / a) - 1.0)
                 - (V + I * m.R_s) / R_sh - I)
        assert_true(abs(resid) < 1e-6,
                    f"residual at V={V}: {resid:.2e} ~ 0")


def test_voltage_drops_with_temperature():
    print("\n[Test 8] V_oc decreases as cell temperature rises")
    m, _ = make_model()
    r_cold = m.mpp(1000.0, 15.0, G_rear=0.0)
    r_hot = m.mpp(1000.0, 60.0, G_rear=0.0)
    assert_true(r_hot["v_oc"] < r_cold["v_oc"],
                f"V_oc hot({r_hot['v_oc']:.2f}) < cold({r_cold['v_oc']:.2f})")
    assert_true(r_hot["p_mp"] < r_cold["p_mp"],
                f"P_mp hot({r_hot['p_mp']:.1f}) < cold({r_cold['p_mp']:.1f})")


def test_thermal_ode_heats_up():
    print("\n[Test 9] Thermal ODE: module heats above ambient under sun")
    m, _ = make_model()
    r = m.simulate(900.0, T_amb_C=25.0, v_wind=1.0, albedo=0.3,
                   T_cell0_C=25.0, dt=60.0, duration_s=3600.0)
    Tf = r["temperature_C"][-1]
    assert_true(Tf > 25.0, f"T_cell_final={Tf:.2f}C > ambient 25C")
    assert_true(Tf < 90.0, f"T_cell_final={Tf:.2f}C < 90C (physically reasonable)")
    # approximate steady state reached
    dT = abs(r["temperature_C"][-1] - r["temperature_C"][-2])
    assert_true(dT < 0.5, f"near steady state: dT={dT:.4f}C between last steps")


def test_thermal_dark_cools_to_ambient():
    print("\n[Test 10] No sun: module relaxes to ambient temperature")
    m, _ = make_model()
    r = m.simulate(0.0, T_amb_C=10.0, v_wind=2.0, G_rear=0.0,
                   T_cell0_C=40.0, dt=60.0, duration_s=7200.0)
    Tf = r["temperature_C"][-1]
    assert_true(abs(Tf - 10.0) < 1.0, f"T_cell_final={Tf:.2f}C -> ambient 10C")
    assert_true(np.all(np.abs(r["p_mp"]) < 1e-6), "P_mp = 0 in the dark")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"G_front_W_m2": 800.0, "albedo": 0.4,
                    "duration_s": 600.0, "dt": 60.0})
    for key in ["t", "temperature_C", "v_mp", "i_mp", "p_mp",
                "v_oc", "i_sc", "efficiency", "G_effective", "bifacial_gain"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["p_mp"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC052", "get_info id == EC052")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1-hour sim at dt=60s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(900.0, T_amb_C=25.0, v_wind=1.0, albedo=0.3,
               dt=60.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  1-hour simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_zero_irradiance_zero_power,
        test_bifacial_gain_positive,
        test_gain_increases_with_albedo,
        test_efficiency_range,
        test_isc_proportional_to_Geff,
        test_pv_monotone_to_mpp,
        test_lambertw_vs_residual,
        test_voltage_drops_with_temperature,
        test_thermal_ode_heats_up,
        test_thermal_dark_cools_to_ambient,
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
    print(f"EC052 Bifacial PV F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
