"""
EC049 -- Multi-Junction CPV -- F2a Physics-Lumped
Test suite: current matching, concentration scaling, thermal ODE, edge cases.
Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MultiJunctionCPV_F2a
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
def test_zero_dni_zero_power():
    print("\n[Test 1] P = 0 at DNI = 0")
    m, _ = make_model()
    r = m.mpp(0.0, 298.15)
    assert_true(r["p_mp"] == 0.0, f"P_mp={r['p_mp']} at DNI=0")
    assert_true(r["v_oc"] == 0.0, f"Voc={r['v_oc']} at DNI=0")
    assert_true(r["concentration"] == 0.0, "Concentration=0 at DNI=0")


def test_current_matching():
    print("\n[Test 2] Series current matched: i_sc == limiting subcell Jsc")
    m, _ = make_model()
    C = m.concentration(900.0)
    Jsc = m.subcell_Jsc(C, 298.15)
    J_lim = m.limiting_current(C, 298.15)
    assert_true(abs(J_lim - np.min(Jsc)) < 1e-12,
                f"limiting current = min subcell Jsc ({J_lim:.3f} A/cm2)")
    # the limiting subcell should be one of the configured ones
    idx = m.limiting_index(C, 298.15)
    assert_true(0 <= idx < m.n_junctions,
                f"limiting subcell index {idx} ({m.names[idx]})")


def test_voltage_is_sum_of_subcells():
    print("\n[Test 3] Cell voltage = sum of subcell voltages (minus Rs drop)")
    m, _ = make_model()
    C = m.concentration(900.0)
    T = 320.0
    J = 0.0  # at open circuit Rs drop = 0
    Jsc = m.subcell_Jsc(C, T)
    J0 = m.subcell_J0(T)
    Vsub = m.subcell_voltage(J, Jsc, J0, T)
    V_total = m.cell_voltage(J, C, T)
    assert_true(abs(V_total - np.sum(Vsub)) < 1e-9,
                f"V_cell={V_total:.4f} = sum(Vsub)={np.sum(Vsub):.4f}")
    assert_true(m.n_junctions == 3 and V_total > 2.5,
                f"3-junction Voc {V_total:.3f} V >> single-junction (~0.7-1 V)")


def test_voc_rises_with_concentration():
    print("\n[Test 4] Voc rises ~logarithmically with concentration")
    m, _ = make_model()
    T = 298.15
    Voc_low = m.Voc(m.concentration(100.0), T)
    Voc_mid = m.Voc(m.concentration(500.0), T)
    Voc_hi = m.Voc(m.concentration(1000.0), T)
    assert_true(Voc_low < Voc_mid < Voc_hi,
                f"Voc(100)={Voc_low:.3f} < Voc(500)={Voc_mid:.3f} < Voc(1000)={Voc_hi:.3f}")
    # logarithmic: dVoc for 10x ~ n*Vt*ln(10)*n_junctions; check it's modest (< 0.6 V)
    dVoc = Voc_hi - Voc_low  # factor ~10 in concentration ratio
    assert_true(0.05 < dVoc < 0.8,
                f"dVoc over 10x C = {dVoc:.3f} V (logarithmic, not linear)")


def test_jsc_scales_with_concentration():
    print("\n[Test 5] Photocurrent scales linearly with concentration")
    m, _ = make_model()
    J1 = m.limiting_current(m.concentration(100.0), 298.15)
    J2 = m.limiting_current(m.concentration(200.0), 298.15)
    assert_true(abs(J2 / J1 - 2.0) < 1e-6,
                f"Jsc doubles when C doubles: {J2/J1:.4f}")


def test_efficiency_bound():
    print("\n[Test 6] Efficiency in (0, 0.45], peaks near ~0.40")
    m, _ = make_model()
    etas = []
    for dni in [200, 500, 800, 1000]:
        r = m.mpp(dni, 298.15)
        etas.append(r["efficiency"])
        assert_true(0.0 < r["efficiency"] < 0.45,
                    f"eta(DNI={dni})={r['efficiency']*100:.2f}% in (0,45%)")
    assert_true(max(etas) > 0.30,
                f"peak eta {max(etas)*100:.1f}% reaches high-CPV regime (>30%)")


def test_fill_factor_range():
    print("\n[Test 7] Fill factor physically reasonable (0.7-0.95)")
    m, _ = make_model()
    r = m.mpp(900.0, 298.15)
    assert_true(0.7 < r["fill_factor"] < 0.95,
                f"FF={r['fill_factor']:.3f} in (0.7, 0.95)")
    assert_true(r["v_mp"] < r["v_oc"], f"v_mp={r['v_mp']:.3f} < v_oc={r['v_oc']:.3f}")


def test_thermal_ode_heats_up():
    print("\n[Test 8] Thermal ODE: junction heats above coolant under load")
    m, _ = make_model()
    r = m.simulate(900.0, T0=298.15, dt=2.0, duration_s=120.0)
    assert_true(r["temperature"][-1] > 298.15,
                f"T_final={r['temperature'][-1]:.2f} K > coolant 298.15 K")
    assert_true(r["temperature"][-1] < 420.0,
                f"T_final={r['temperature'][-1]:.2f} K < 420 K (active cooling holds it)")


def test_thermal_steady_state():
    print("\n[Test 9] Thermal ODE reaches steady state")
    m, _ = make_model()
    r = m.simulate(900.0, T0=298.15, dt=5.0, duration_s=1500.0)
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.05, f"Near SS: dT={dT:.5f} K between last two steps")


def test_temperature_lowers_voc():
    print("\n[Test 10] Higher junction temperature lowers Voc")
    m, _ = make_model()
    C = m.concentration(900.0)
    Voc_cold = m.Voc(C, 280.0)
    Voc_hot = m.Voc(C, 350.0)
    assert_true(Voc_hot < Voc_cold,
                f"Voc(350K)={Voc_hot:.3f} < Voc(280K)={Voc_cold:.3f}")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"DNI": 900.0, "dt": 5.0, "duration_s": 30.0})
    for key in ["t", "temperature", "p_mp", "v_mp", "i_mp", "v_oc",
                "efficiency", "concentration"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["temperature"]) == len(r["p_mp"]),
                "Output arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC049", "get_info id = EC049")


def test_benchmark():
    print("\n[Test 12] Benchmark: 120 s transient sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(900.0, T0=298.15, dt=2.0, duration_s=120.0)
    elapsed = time.perf_counter() - t0
    print(f"  120 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_zero_dni_zero_power,
        test_current_matching,
        test_voltage_is_sum_of_subcells,
        test_voc_rises_with_concentration,
        test_jsc_scales_with_concentration,
        test_efficiency_bound,
        test_fill_factor_range,
        test_thermal_ode_heats_up,
        test_thermal_steady_state,
        test_temperature_lowers_voc,
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
    print(f"EC049 Multi-Junction CPV F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
