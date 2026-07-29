"""
EC050 -- Organic Photovoltaic (OPV) -- F2a Physics-Lumped + Thermal ODE
Test suite: physics sanity, mandated OPV constraints, edge cases, interface, benchmark.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import OPV_F2a
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


T25 = 298.15  # K


# ---------------------------------------------------------------------------
def test_zero_power_at_dark():
    print("\n[Test 1] P = 0 at G = 0 (enforced)")
    m, _ = make_model()
    res = m.mpp(0.0, T25)
    assert_true(res["Pmp"] == 0.0, f"Pmp(G=0)={res['Pmp']:.6e} == 0")
    assert_true(res["Isc"] == 0.0, f"Isc(G=0)={res['Isc']:.6e} == 0")
    assert_true(res["eta"] == 0.0, f"eta(G=0)={res['eta']:.6e} == 0")


def test_efficiency_bounds():
    print("\n[Test 2] Efficiency in (0, 0.12) across irradiance range")
    m, _ = make_model()
    for G in [50.0, 100.0, 300.0, 600.0, 1000.0, 1200.0]:
        eta = m.mpp(G, T25)["eta"]
        assert_true(0.0 < eta < 0.12, f"G={G:6.0f}: eta={eta*100:.2f}% in (0, 12%)")


def test_isc_proportional_to_G():
    print("\n[Test 3] Isc proportional to G (Isc/G ~ const)")
    m, _ = make_model()
    ratios = []
    for G in [100.0, 300.0, 600.0, 1000.0]:
        isc = m.mpp(G, T25)["Isc"]
        ratios.append(isc / G)
    r = np.array(ratios)
    rel_spread = (r.max() - r.min()) / r.mean()
    assert_true(rel_spread < 0.02, f"Isc/G spread={rel_spread*100:.3f}% < 2% (linear in G)")


def test_pv_monotone_to_mpp():
    print("\n[Test 4] P-V curve monotonically rises to MPP then falls")
    m, _ = make_model()
    curve = m.iv_curve(1000.0, T25, n_points=300)
    P = curve["P"]
    imax = int(np.argmax(P))
    rising = np.all(np.diff(P[:imax + 1]) >= -1e-9)
    falling = np.all(np.diff(P[imax:]) <= 1e-9)
    assert_true(rising, f"P rises monotonically up to MPP (idx {imax})")
    assert_true(falling, "P falls monotonically after MPP")
    assert_true(0 < imax < len(P) - 1, f"MPP interior at idx {imax}/{len(P)}")


def test_low_fill_factor():
    print("\n[Test 5] Low fill factor (OPV: 0.4 < FF < 0.7)")
    m, _ = make_model()
    ff = m.mpp(1000.0, T25)["FF"]
    assert_true(0.40 < ff < 0.70, f"FF={ff:.3f} in OPV range (0.40, 0.70)")


def test_high_ideality_and_resistance():
    print("\n[Test 6] High ideality factor and significant Rs/Rsh present")
    m, _ = make_model()
    assert_true(m.n >= 1.5, f"ideality n={m.n} >= 1.5 (recombination-dominated OPV)")
    assert_true(m.Rs > 0.0, f"Rs={m.Rs} Ohm > 0 (lowers FF)")
    assert_true(np.isfinite(m.Rsh_ref) and m.Rsh_ref > 0,
                f"finite Rsh={m.Rsh_ref} Ohm (lowers FF / low-light)")


def test_iv_consistency():
    print("\n[Test 7] I(V) root-find consistency: I=Isc at V=0, I=0 at Voc")
    m, _ = make_model()
    params = m.diode_params(1000.0, T25)
    Voc = m.open_circuit_voltage(1000.0, T25, params)
    I_at_voc = float(m.current_from_voltage(Voc, 1000.0, T25, params))
    I_at_0 = float(m.current_from_voltage(0.0, 1000.0, T25, params))
    Isc = m.mpp(1000.0, T25)["Isc"]
    assert_true(abs(I_at_voc) < 1e-4, f"I(Voc)={I_at_voc:.2e} A ~ 0")
    assert_true(abs(I_at_0 - Isc) < 1e-6, f"I(0)={I_at_0:.4f} == Isc={Isc:.4f}")


def test_temperature_dependence():
    print("\n[Test 8] Temperature dependence: Voc falls, Isc rises with T")
    m, _ = make_model()
    cold = m.mpp(1000.0, 273.15 + 10.0)
    hot = m.mpp(1000.0, 273.15 + 60.0)
    assert_true(hot["Voc"] < cold["Voc"], f"Voc: hot {hot['Voc']:.3f} < cold {cold['Voc']:.3f}")
    assert_true(hot["Isc"] > cold["Isc"], f"Isc: hot {hot['Isc']:.4f} > cold {cold['Isc']:.4f}")
    assert_true(hot["Pmp"] < cold["Pmp"], f"Pmp: hot {hot['Pmp']:.3f} < cold {cold['Pmp']:.3f}")


def test_low_light_strength():
    print("\n[Test 9] Excellent low-light: relative eff retained at 100 W/m2")
    m, _ = make_model()
    eta_stc = m.mpp(1000.0, T25)["eta"]
    eta_low = m.mpp(100.0, T25)["eta"]
    rel = eta_low / eta_stc
    assert_true(rel > 0.75, f"eta(100)/eta(1000)={rel:.2f} > 0.75 (good low-light)")


def test_thermal_ode_warms_and_settles():
    print("\n[Test 10] Thermal ODE: cell warms above ambient, reaches steady state")
    m, _ = make_model()
    r = m.simulate(1000.0, 25.0, T0_C=25.0, dt=5.0, duration_s=1200.0)
    T_final = r["T_cell_C"][-1]
    assert_true(T_final > 25.0, f"T_cell_final={T_final:.2f}C > T_amb=25C (absorbed heat)")
    assert_true(T_final < 90.0, f"T_cell_final={T_final:.2f}C < 90C (physical)")
    dT = abs(r["T_cell_C"][-1] - r["T_cell_C"][-2])
    assert_true(dT < 0.05, f"near steady state: dT={dT:.4f}C between last steps")
    T_ss = m.steady_state_temperature(1000.0, 25.0 + 273.15) - 273.15
    assert_true(abs(T_ss - T_final) < 1.0,
                f"ODE final {T_final:.2f}C matches fixed-point SS {T_ss:.2f}C")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"irradiance": 800.0, "T_ambient_C": 20.0,
                    "dt": 10.0, "duration_s": 100.0})
    for key in ["t", "T_cell_C", "Voc", "Isc", "Vmp", "Imp",
                "power", "efficiency", "FF", "final"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["power"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC050", "get_info component_id == EC050")


def test_benchmark():
    print("\n[Test 12] Benchmark: 600s thermal sim at dt=5s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(1000.0, 25.0, dt=5.0, duration_s=600.0)
    elapsed = time.perf_counter() - t0
    print(f"  600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_zero_power_at_dark,
        test_efficiency_bounds,
        test_isc_proportional_to_G,
        test_pv_monotone_to_mpp,
        test_low_fill_factor,
        test_high_ideality_and_resistance,
        test_iv_consistency,
        test_temperature_dependence,
        test_low_light_strength,
        test_thermal_ode_warms_and_settles,
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
    print(f"EC050 OPV F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
