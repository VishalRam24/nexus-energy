"""
EC095 — Thermoelectric Cooler (Peltier) — F2a Physics-Lumped Transient
Test suite: thermoelectric physics sanity, energy conservation, COP bounds,
optimal-current behaviour, transient ODE, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import PeltierTEC_F2a
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
def test_energy_conservation():
    print("\n[Test 1] First-law energy conservation: Q_h = Q_c + W_in")
    m, _ = make_model()
    for I in [1.0, 3.0, 5.0]:
        for Tc, Th in [(280.0, 300.0), (270.0, 320.0)]:
            Qc = float(m.cooling_power(I, Tc, Th))
            W = float(m.electrical_input(I, Tc, Th))
            Qh = float(m.heat_rejection(I, Tc, Th))
            assert_true(abs(Qh - (Qc + W)) < 1e-6,
                        f"I={I},dT={Th-Tc}: Qh={Qh:.3f}=Qc+W={Qc+W:.3f}")


def test_cop_positive_and_below_carnot():
    print("\n[Test 2] 0 < COP < Carnot in the cooling regime")
    m, _ = make_model()
    Tc, Th = 285.0, 300.0
    checked = 0
    for I in np.linspace(0.5, 5.0, 20):
        Qc = float(m.cooling_power(I, Tc, Th))
        if Qc <= 0:
            continue
        cop = float(m.cop(I, Tc, Th))
        carnot = float(m.carnot_cop(Tc, Th))
        assert_true(cop > 0, f"I={I:.2f}: COP={cop:.3f} > 0")
        assert_true(cop < carnot, f"I={I:.2f}: COP={cop:.3f} < Carnot={carnot:.3f}")
        checked += 1
    assert_true(checked >= 5, f"{checked} cooling points exercised")


def test_seebeck_joule_fourier_terms():
    print("\n[Test 3] Q_c decomposition: Peltier up, Joule/Fourier down")
    m, _ = make_model()
    Tc, Th = 290.0, 300.0
    # Peltier pumping grows linearly with I; Joule grows quadratically.
    I = 2.0
    peltier = m.N * m.alpha * I * Tc
    joule_half = m.N * 0.5 * I * I * m.R
    fourier = m.N * m.K * (Th - Tc)
    Qc = float(m.cooling_power(I, Tc, Th))
    assert_true(abs(Qc - (peltier - joule_half - fourier)) < 1e-6,
                "Q_c = alpha*I*Tc - 0.5*I^2*R - K*dT")
    # Zero current: only Fourier leak, Q_c must be negative (heat leaks in).
    assert_true(float(m.cooling_power(0.0, Tc, Th)) < 0,
                "I=0: cold side gains heat (Fourier leak, Q_c<0)")


def test_qc_max_optimal_current():
    print("\n[Test 4] Q_c maximised at I_opt = alpha*Tc/R")
    m, _ = make_model()
    Tc, Th = 280.0, 310.0
    I_opt = m.alpha * Tc / m.R
    Qc_opt = float(m.cooling_power(I_opt, Tc, Th))
    for dI in [-1.0, -0.3, 0.3, 1.0]:
        Qc = float(m.cooling_power(I_opt + dI, Tc, Th))
        assert_true(Qc <= Qc_opt + 1e-6,
                    f"Qc(I_opt{dI:+.1f})={Qc:.3f} <= Qc_opt={Qc_opt:.3f}")


def test_cop_optimal_current_distinct():
    print("\n[Test 5] COP-optimal current maximises COP and differs from Q_c-opt")
    m, _ = make_model()
    Tc, Th = 285.0, 305.0
    I_cop = m.optimum_current_cop(Tc, Th)
    cop_opt = float(m.cop(I_cop, Tc, Th))
    for dI in [-0.5, -0.2, 0.2, 0.5]:
        Ip = I_cop + dI
        if Ip <= 0:
            continue
        if float(m.cooling_power(Ip, Tc, Th)) <= 0:
            continue
        cop = float(m.cop(Ip, Tc, Th))
        assert_true(cop <= cop_opt + 1e-6,
                    f"COP(I_cop{dI:+.1f})={cop:.4f} <= COP_opt={cop_opt:.4f}")
    I_qc = float(m.optimum_current_qc(Tc))
    assert_true(I_cop < I_qc, f"I_optCOP={I_cop:.2f} < I_optQc={I_qc:.2f}")


def test_zt_reasonable():
    print("\n[Test 6] Device figure of merit ZT in Bi2Te3 module range")
    m, _ = make_model()
    # Module-level (device) ZT includes lead/contact losses, so it is lower
    # than the intrinsic material ZT (~0.7-1.0 for Bi2Te3, Rowe 2006).
    zt = float(m.zt(300.0))
    assert_true(0.3 < zt < 1.0, f"device ZT(300K)={zt:.3f} in [0.3,1.0]")
    # And it must scale linearly with temperature.
    assert_true(abs(float(m.zt(600.0)) - 2.0 * zt) < 1e-9, "ZT linear in T")


def test_transient_cools_down():
    print("\n[Test 7] Transient ODE: cold plate cools below start, hot plate warms")
    m, _ = make_model()
    T0 = 298.15
    r = m.simulate(4.0, T0, T0, T0, T0, Q_load_W=0.0, dt=2.0, duration_s=600.0)
    assert_true(r["T_cold"][-1] < T0 - 2.0,
                f"T_cold {r['T_cold_C'][-1]:.2f}C dropped from {T0-273.15:.2f}C")
    assert_true(r["T_hot"][-1] > T0,
                f"T_hot {r['T_hot_C'][-1]:.2f}C rose above ambient")
    assert_true(r["T_cold"][-1] < r["T_hot"][-1], "Cold side colder than hot side")


def test_transient_steady_state():
    print("\n[Test 8] Transient reaches approximate steady state")
    m, _ = make_model()
    T0 = 298.15
    r = m.simulate(3.5, T0, T0, T0, T0, Q_load_W=15.0, dt=5.0, duration_s=3000.0)
    dTc = abs(r["T_cold"][-1] - r["T_cold"][-2])
    assert_true(dTc < 0.02, f"Near SS: dT_cold={dTc:.5f} K between last two steps")


def test_global_energy_balance_steady():
    print("\n[Test 9] Steady-state node balances hold (in - out ~ 0)")
    m, _ = make_model()
    T0 = 298.15
    r = m.simulate(3.5, T0, T0, T0, T0, Q_load_W=15.0, dt=5.0, duration_s=3000.0)
    Tc = r["T_cold"][-1]; Th = r["T_hot"][-1]
    Qc = float(m.cooling_power(3.5, Tc, Th))
    Qh = float(m.heat_rejection(3.5, Tc, Th))
    # cold node: -Qc + Qload + hA_c*(T_load - Tc) ~ 0
    cold_bal = -Qc + 15.0 + m.hA_cold * (T0 - Tc)
    hot_bal = Qh - m.hA_hot * (Th - T0)
    assert_true(abs(cold_bal) < 0.5, f"cold-node residual={cold_bal:.4f} W ~ 0")
    assert_true(abs(hot_bal) < 0.5, f"hot-node residual={hot_bal:.4f} W ~ 0")


def test_higher_current_lifts_more_then_overheats():
    print("\n[Test 10] Q_c rises to I_opt then falls as Joule dominates")
    m, _ = make_model()
    Tc, Th = 290.0, 300.0
    I_opt = m.alpha * Tc / m.R
    # Below the optimum Q_c increases with current...
    Qc_lo = float(m.cooling_power(0.5 * I_opt, Tc, Th))
    Qc_opt = float(m.cooling_power(I_opt, Tc, Th))
    # ...above it (Joule term ~I^2) Q_c falls back down.
    Qc_hi = float(m.cooling_power(1.5 * I_opt, Tc, Th))
    assert_true(Qc_opt > Qc_lo, f"Qc(I_opt)={Qc_opt:.2f} > Qc(0.5 I_opt)={Qc_lo:.2f}")
    assert_true(Qc_hi < Qc_opt, f"Qc(1.5 I_opt)={Qc_hi:.2f} < Qc(I_opt)={Qc_opt:.2f} (Joule)")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"current_A": 4.0, "T_ambient_C": 25.0, "T_load_C": 10.0,
                    "Q_load_W": 20.0, "dt": 5.0, "duration_s": 120.0})
    for key in ["t", "T_cold", "T_hot", "current", "Q_cold", "Q_hot",
                "W_elec", "cop", "cop_carnot", "dT"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_cold"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC095", "get_info component_id")


def test_benchmark():
    print("\n[Test 12] Benchmark: 600s sim at dt=1")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(4.0, 298.15, 298.15, 298.15, 298.15, Q_load_W=10.0,
               dt=1.0, duration_s=600.0)
    elapsed = time.perf_counter() - t0
    print(f"  600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_energy_conservation,
        test_cop_positive_and_below_carnot,
        test_seebeck_joule_fourier_terms,
        test_qc_max_optimal_current,
        test_cop_optimal_current_distinct,
        test_zt_reasonable,
        test_transient_cools_down,
        test_transient_steady_state,
        test_global_energy_balance_steady,
        test_higher_current_lifts_more_then_overheats,
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
    print(f"EC095 Peltier TEC F2a — Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
