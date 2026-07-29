"""
EC218 -- Thermionic Converter -- F2a Physics-Lumped Emitter-Thermal
Test suite: thermionic-emission physics, energy balance, Carnot limit,
ODE convergence, edge cases, predict() interface, benchmark timing.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import ThermionicF2a, k_B, q_e
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
def test_richardson_law():
    print("\n[Test 1] Richardson-Dushman law: J = A T^2 exp(-q phi / kT)")
    m, _ = make_model()
    phi, T = 2.0, 1800.0
    J = m.richardson_current(phi, T)
    J_ref = m.A_r * T ** 2 * np.exp(-phi * q_e / (k_B * T))
    assert_true(abs(J - J_ref) / J_ref < 1e-9, f"J={J:.4e} matches closed form")
    assert_true(J > 0, "Emission current strictly positive")


def test_emission_monotone_in_T():
    print("\n[Test 2] Emission current rises with emitter temperature")
    m, _ = make_model()
    Ts = np.linspace(1300, 2100, 40)
    J = [m.richardson_current(2.0, T) for T in Ts]
    assert_true(all(J[i + 1] > J[i] for i in range(len(J) - 1)),
                "J strictly increasing in T")
    ratio = J[-1] / J[0]
    assert_true(ratio > 100, f"Strong T-sensitivity: J grows x{ratio:.0f}")


def test_work_function_dependence():
    print("\n[Test 3] Emission falls with higher work function")
    m, _ = make_model()
    J_low = m.richardson_current(1.5, 1800.0)
    J_high = m.richardson_current(2.5, 1800.0)
    assert_true(J_high < J_low, f"J(phi=2.5)={J_high:.3e} < J(phi=1.5)={J_low:.3e}")


def test_voltage_is_work_function_difference():
    print("\n[Test 4] Open-circuit voltage = phi_E - phi_C")
    m, _ = make_model()
    T_E, T_C = 1800.0, 900.0
    V_oc = float(m.open_circuit_voltage(T_E, T_C))
    expect = float(m.phi_emitter(T_E) - m.phi_collector(T_C))
    assert_true(abs(V_oc - expect) < 1e-9, f"V_oc={V_oc:.4f} = phi_E-phi_C")
    assert_true(V_oc > 0, "Positive net EMF when phi_E > phi_C")


def test_efficiency_below_carnot():
    print("\n[Test 5] Efficiency strictly below Carnot limit")
    m, _ = make_model()
    for (T_E, T_C) in [(1600, 800), (1800, 900), (2000, 1000)]:
        op = m.operating_point(T_E, T_C)
        eta_c = 1.0 - T_C / T_E
        assert_true(op["efficiency"] <= eta_c + 1e-12,
                    f"eta={op['efficiency']:.4f} <= Carnot={eta_c:.4f} at T_E={T_E}")
        assert_true(0.0 <= op["efficiency"] < 1.0, "Efficiency in [0,1)")


def test_energy_balance_radiation():
    print("\n[Test 6] Heat input = electron cooling + radiation (T^4)")
    m, _ = make_model()
    T_E, T_C = 1800.0, 900.0
    Q_e = float(m.electron_cooling_power(T_E, T_C))
    Q_r = float(m.radiation_power(T_E, T_C))
    Q_tot = float(m.heat_input(T_E, T_C))
    assert_true(abs(Q_tot - (Q_e + Q_r)) < 1e-9, "Q_in = Q_electron + Q_rad")
    # Radiation scales as T^4
    Q_r2 = float(m.radiation_power(2.0 * T_E, T_C))
    # (2 T_E)^4 dominates; ratio ~ 16 ignoring T_C^4 term
    assert_true(Q_r2 / Q_r > 14, f"Radiation ~T^4: ratio={Q_r2/Q_r:.1f}")
    assert_true(Q_r > 0, "Net radiation emitter->collector positive")


def test_steady_state_energy_balance():
    print("\n[Test 7] ODE steady state: Q_external == Q_input(T_ss)")
    m, _ = make_model()
    r = m.simulate(80.0, 1500.0, 900.0, dt=0.05, duration_s=60.0)
    T_ss = r["T_emitter"][-1]
    Q_in_ss = float(m.heat_input(T_ss, 900.0))
    assert_true(abs(Q_in_ss - 80.0) / 80.0 < 0.02,
                f"At SS Q_in={Q_in_ss:.2f} W ~ Q_ext=80 W")
    dT = abs(r["T_emitter"][-1] - r["T_emitter"][-2])
    assert_true(dT < 0.5, f"Near steady state: dT={dT:.4f} K")


def test_emitter_heats_up():
    print("\n[Test 8] Cold emitter heats up under external power")
    m, _ = make_model()
    r = m.simulate(100.0, 1300.0, 900.0, dt=0.05, duration_s=30.0)
    assert_true(r["T_emitter"][-1] > 1300.0,
                f"T_final={r['T_emitter'][-1]:.1f} > 1300 K")
    assert_true(r["T_emitter"][-1] < 2300.0,
                f"T_final={r['T_emitter'][-1]:.1f} stays bounded")


def test_current_from_emission():
    print("\n[Test 9] Output current driven by emission (drops if T_E falls)")
    m, _ = make_model()
    op_hot = m.operating_point(1900.0, 900.0)
    op_cool = m.operating_point(1500.0, 900.0)
    assert_true(op_hot["J_net_Am2"] > op_cool["J_net_Am2"],
                f"J(1900K)={op_hot['J_net_Am2']:.2e} > J(1500K)={op_cool['J_net_Am2']:.2e}")
    assert_true(op_hot["power_w"] >= op_cool["power_w"],
                "Hotter emitter delivers >= power")


def test_no_emf_when_phi_equal():
    print("\n[Test 10] Edge: no net output when collector hotter / phi_E<=phi_C")
    m, _ = make_model()
    # Force collector work function above emitter via equal-ish temps not enough;
    # use operating point where phi_E - phi_C clamps to >=0 voltage.
    op = m.operating_point(1300.0, 1200.0)
    assert_true(op["V_terminal_V"] >= 0.0, "Terminal voltage never negative")
    assert_true(op["power_w"] >= 0.0, "Power never negative")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"Q_external_w": 60.0, "dt": 0.5, "duration_s": 10.0})
    for key in ["t", "T_emitter", "J_net_Am2", "V_terminal_V", "power_w",
                "heat_input_w", "efficiency", "carnot_efficiency"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_emitter"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC218", "get_info reports EC218")


def test_benchmark():
    print("\n[Test 12] Benchmark: 60 s emitter-thermal sim at dt=0.05")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(60.0, 1800.0, 900.0, dt=0.05, duration_s=60.0)
    elapsed = time.perf_counter() - t0
    print(f"  60 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_richardson_law,
        test_emission_monotone_in_T,
        test_work_function_dependence,
        test_voltage_is_work_function_difference,
        test_efficiency_below_carnot,
        test_energy_balance_radiation,
        test_steady_state_energy_balance,
        test_emitter_heats_up,
        test_current_from_emission,
        test_no_emf_when_phi_equal,
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
    print(f"EC218 Thermionic Converter F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
