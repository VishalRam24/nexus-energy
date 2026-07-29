"""
EC220 -- Triboelectric Nanogenerator (TENG) -- F2a Physics-Lumped V-Q-x Model
Test suite: charge conservation, V-Q-x governing-equation sanity, power-vs-load
optimum, scaling with surface charge & motion, ODE convergence, edge cases,
predict() interface, benchmark timing. Custom harness (NO pytest).
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import TENG_F2a, eps_0
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
def test_governing_relation():
    print("\n[Test 1] V-Q-x governing eq: V_oc & C(x) match Niu 2013 forms")
    m, _ = make_model()
    x = m.x_max
    # V_oc = sigma*x/eps_0
    assert_true(abs(m.v_oc(x) - m.sigma * x / eps_0) < 1e-9,
                f"V_oc(x_max)={m.v_oc(x):.2f} V = sigma*x/eps_0")
    # C(x) = eps_0*A/(d0+x), decreasing with gap
    C0 = m.capacitance(0.0)
    Cmax = m.capacitance(m.x_max)
    assert_true(C0 > Cmax, f"C decreases with gap: C(0)={C0:.3e} > C(x_max)={Cmax:.3e}")
    assert_true(abs(Cmax - eps_0 * m.A / (m.d0 + m.x_max)) < 1e-30, "C(x_max) formula")


def test_charge_conservation():
    print("\n[Test 2] Charge conservation: net charge per cycle ~ 0")
    m, _ = make_model()
    r = m.simulate(3.0, 1e7, n_cycles=8)
    Q_sat = m.q_saturation()
    rel = abs(r["net_charge_per_cycle"]) / Q_sat
    assert_true(rel < 1e-2,
                f"net charge/cycle = {r['net_charge_per_cycle']:.3e} C "
                f"<< Q_sat={Q_sat:.3e} C (rel={rel:.2e})")


def test_charge_bounded_by_saturation():
    print("\n[Test 3] |Q(t)| bounded by short-circuit saturation charge")
    m, _ = make_model()
    r = m.simulate(3.0, 1e7, n_cycles=6)
    Q_sat = m.q_saturation()
    assert_true(np.max(np.abs(r["charge"])) <= Q_sat * 1.05,
                f"max|Q|={np.max(np.abs(r['charge'])):.3e} <= Q_sat={Q_sat:.3e}")
    assert_true(Q_sat > 0, "Q_sat > 0")


def test_energy_and_power_positive():
    print("\n[Test 4] Energy per cycle and average power are positive")
    m, _ = make_model()
    r = m.simulate(3.0, 1e7, n_cycles=6)
    assert_true(r["energy_per_cycle"] > 0, f"E/cycle={r['energy_per_cycle']:.3e} J > 0")
    assert_true(r["power_avg"] > 0, f"P_avg={r['power_avg']:.3e} W > 0")
    # P_avg should equal E/cycle * f
    assert_true(abs(r["power_avg"] - r["energy_per_cycle"] * 3.0) / r["power_avg"] < 1e-6,
                "P_avg = E_cycle * f (consistency)")


def test_power_vs_load_optimum():
    print("\n[Test 5] Power vs load has an interior optimum (impedance match)")
    m, _ = make_model()
    R_arr, P_arr, R_opt = m.power_vs_load(3.0)
    imax = int(np.argmax(P_arr))
    assert_true(0 < imax < len(R_arr) - 1,
                f"optimum interior at R_opt={R_opt:.3e} ohm (idx {imax}/{len(R_arr)-1})")
    # Power lower at both extremes than at optimum
    assert_true(P_arr[imax] > P_arr[0] and P_arr[imax] > P_arr[-1],
                "P(R_opt) exceeds P at both load extremes")
    # TENG is a high-impedance source: optimum in the MOhm-GOhm band
    assert_true(1e5 < R_opt < 1e10, f"R_opt={R_opt:.3e} ohm in high-impedance band")


def test_scaling_with_surface_charge():
    print("\n[Test 6] Average power scales ~ sigma^2")
    m, _ = make_model()
    P1 = m.simulate(3.0, 1e7, 6)["power_avg"]
    m.sigma *= 2.0
    P2 = m.simulate(3.0, 1e7, 6)["power_avg"]
    ratio = P2 / P1
    assert_true(3.5 < ratio < 4.5, f"doubling sigma -> P x{ratio:.2f} (~4 expected)")


def test_scaling_with_motion():
    print("\n[Test 7] Output increases with gap amplitude (motion)")
    m, _ = make_model()
    P_small = m.simulate(3.0, 1e7, 6)["power_avg"]
    m.x_max *= 2.0
    P_big = m.simulate(3.0, 1e7, 6)["power_avg"]
    assert_true(P_big > P_small,
                f"larger stroke raises power: {P_big:.3e} > {P_small:.3e}")


def test_power_increases_with_frequency():
    print("\n[Test 8] Average power increases with frequency (more cycles/s)")
    m, _ = make_model()
    f_arr, P_arr = m.power_vs_frequency(R_load_ohm=1e7, n_cycles=6)
    assert_true(P_arr[-1] > P_arr[0],
                f"P({f_arr[-1]:.1f}Hz)={P_arr[-1]:.3e} > P({f_arr[0]:.1f}Hz)={P_arr[0]:.3e}")


def test_voltage_consistency():
    print("\n[Test 9] V = R*dQ/dt = -Q/C(x) + V_oc(x) consistent with ODE")
    m, _ = make_model()
    r = m.simulate(3.0, 1e7, n_cycles=4)
    # Recompute V from charge & gap, compare to stored
    x = r["gap"]
    V_check = -r["charge"] * (m.d0 + x) / (eps_0 * m.A) + m.sigma * x / eps_0
    assert_true(np.allclose(V_check, r["voltage"], rtol=1e-9, atol=1e-9),
                "Stored V matches V-Q-x relation")
    # Open-circuit limit: very large R -> V_peak approaches V_oc(x_max)
    r_oc = m.simulate(3.0, 1e13, n_cycles=4)
    Voc_max = m.v_oc(m.x_max)
    assert_true(r_oc["V_peak"] > 0.5 * Voc_max,
                f"large-R V_peak={r_oc['V_peak']:.1f} -> toward V_oc_max={Voc_max:.1f} V")


def test_edge_low_resistance():
    print("\n[Test 10] Low load (near short circuit): small voltage, finite charge")
    m, _ = make_model()
    r = m.simulate(3.0, 1e4, n_cycles=4)
    assert_true(np.all(np.isfinite(r["voltage"])), "voltage finite at low R")
    assert_true(r["V_peak"] < m.v_oc(m.x_max),
                f"V_peak={r['V_peak']:.3f} < V_oc_max at low R (clamped by load)")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + sweeps")
    _, cm = make_model()
    r = cm.predict({"frequency_hz": 3.0, "R_load_ohm": 1e7, "n_cycles": 4,
                    "sweep_load": True, "sweep_freq": True})
    for key in ["t", "charge", "voltage", "current", "power",
                "energy_per_cycle", "power_avg", "R_optimal_ohm",
                "sweep_power_vs_load", "sweep_power_vs_freq"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["voltage"]) == len(r["charge"]),
                "Time-series arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC220" and info["version"] == "1.0.0",
                "get_info metadata correct")


def test_benchmark():
    print("\n[Test 12] Benchmark: 5-cycle simulation timing")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(3.0, 1e7, n_cycles=5)
    elapsed = time.perf_counter() - t0
    print(f"  5-cycle simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Single simulate() completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_governing_relation,
        test_charge_conservation,
        test_charge_bounded_by_saturation,
        test_energy_and_power_positive,
        test_power_vs_load_optimum,
        test_scaling_with_surface_charge,
        test_scaling_with_motion,
        test_power_increases_with_frequency,
        test_voltage_consistency,
        test_edge_low_resistance,
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
    print(f"EC220 TENG F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
