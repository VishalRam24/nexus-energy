"""
EC071 -- Absorption Heat Pump (LiBr-H2O) -- F2a Physics-Lumped
Test suite: cycle conservation, COP band, mass balance, ODE convergence, edges.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import AbsorptionHeatPumpF2a
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
def test_circulation_ratio():
    print("\n[Test 1] Circulation ratio f = x_strong/(x_strong-x_weak)")
    m, _ = make_model()
    f = m.circulation_ratio()
    f_expected = m.x_strong / (m.x_strong - m.x_weak)
    assert_true(abs(f - f_expected) < 1e-9, f"f={f:.4f} matches species balance")
    assert_true(f > 1.0, f"f={f:.4f} > 1 (more solution than refrigerant)")


def test_mass_conservation():
    print("\n[Test 2] LiBr (salt) conserved in solution loop")
    m, _ = make_model()
    salt_in, salt_out = m.check_mass_balance()
    assert_true(abs(salt_in - salt_out) < 1e-9,
                f"salt_in={salt_in:.5f} == salt_out={salt_out:.5f}")


def test_energy_balance_closes():
    print("\n[Test 3] First law closes: Q_gen+Q_evap+W = Q_cond+Q_abs")
    m, _ = make_model()
    c = m.steady_cycle()
    rel = abs(c["energy_residual"]) / c["q_gen"]
    assert_true(rel < 0.01, f"residual/Q_gen = {rel*100:.3f}% < 1%")


def test_cop_heating_band():
    print("\n[Test 4] Heating thermal COP in physical band (1.0, 1.9)")
    m, _ = make_model()
    c = m.steady_cycle()
    assert_true(1.0 < c["cop_heating"] < 1.9,
                f"COP_heat={c['cop_heating']:.3f} in (1.0, 1.9)")


def test_cop_cooling_band():
    print("\n[Test 5] Cooling COP in physical band (0.6, 0.9) and = COP_h - 1")
    m, _ = make_model()
    c = m.steady_cycle()
    assert_true(0.6 < c["cop_cooling"] < 0.9,
                f"COP_cool={c['cop_cooling']:.3f} in (0.6, 0.9)")
    # Type-I identity: Q_heat = Q_cond+Q_abs, Q_cooling=Q_evap;
    # COP_h ~ COP_c + 1 (each driving heat unit also leaves as useful heat)
    assert_true(abs((c["cop_heating"] - c["cop_cooling"]) - 1.0) < 0.05,
                f"COP_h - COP_c = {c['cop_heating']-c['cop_cooling']:.3f} ~ 1")


def test_all_duties_positive():
    print("\n[Test 6] All four component duties > 0")
    m, _ = make_model()
    c = m.steady_cycle()
    for k in ["q_gen", "q_cond", "q_evap", "q_abs"]:
        assert_true(c[k] > 0, f"{k}={c[k]:.1f} kJ/kg > 0")


def test_property_correlations():
    print("\n[Test 7] LiBr-H2O property correlations are physical")
    m, _ = make_model()
    # solution enthalpy rises with temperature
    h_lo = m.solution_enthalpy(40.0, 0.58)
    h_hi = m.solution_enthalpy(90.0, 0.58)
    assert_true(h_hi > h_lo, f"h(90C)={h_hi:.1f} > h(40C)={h_lo:.1f}")
    # vapour enthalpy >> liquid (latent heat present)
    assert_true(m.water_vapor_enthalpy(40.0) - m.water_liquid_enthalpy(40.0) > 2000,
                "h_fg > 2000 kJ/kg at 40 C")
    # Duhring: higher LiBr fraction raises solution boiling temp
    T1 = m.equilibrium_temperature(40.0, 0.55)
    T2 = m.equilibrium_temperature(40.0, 0.62)
    assert_true(T2 > T1, f"T_eq(x=0.62)={T2:.1f} > T_eq(x=0.55)={T1:.1f}")


def test_transient_warms_and_settles():
    print("\n[Test 8] Transient ODE warms from cold start and reaches steady state")
    m, _ = make_model()
    r = m.simulate(T_gen0_c=40.0, dt=10.0, duration_s=2400.0)
    assert_true(r["T_gen_C"][-1] > r["T_gen_C"][0],
                f"T_gen rises {r['T_gen_C'][0]:.1f}->{r['T_gen_C'][-1]:.1f} C")
    assert_true(r["T_gen_C"][-1] < m.T_drive + 1e-6,
                f"T_gen_final={r['T_gen_C'][-1]:.2f} <= T_drive={m.T_drive} C")
    dT = abs(r["T_gen_C"][-1] - r["T_gen_C"][-2])
    assert_true(dT < 0.05, f"near steady state: dT={dT:.5f} C between last steps")


def test_transient_monotone():
    print("\n[Test 9] Generator temperature monotonically increasing (heating up)")
    m, _ = make_model()
    r = m.simulate(T_gen0_c=40.0, dt=20.0, duration_s=2000.0)
    diffs = np.diff(r["T_gen_C"])
    assert_true(np.all(diffs >= -1e-6), "T_gen non-decreasing throughout warm-up")


def test_higher_drive_raises_gen():
    print("\n[Test 10] Higher driving temperature -> higher steady generator temp")
    m, _ = make_model()
    r_lo = m.simulate(T_drive_c=85.0, T_gen0_c=40.0, dt=20.0, duration_s=3000.0)
    r_hi = m.simulate(T_drive_c=100.0, T_gen0_c=40.0, dt=20.0, duration_s=3000.0)
    assert_true(r_hi["T_gen_C"][-1] > r_lo["T_gen_C"][-1],
                f"T_gen(drive=100)={r_hi['T_gen_C'][-1]:.1f} > "
                f"T_gen(drive=85)={r_lo['T_gen_C'][-1]:.1f}")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"T_gen0_C": 40.0, "duration_s": 600.0, "dt": 60.0})
    for key in ["t", "T_gen_C", "Q_gen_kW", "Q_evap_kW", "Q_heat_kW",
                "cop_heating", "cop_cooling", "cop_heating_design",
                "f_circulation"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_gen_C"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC071", "get_info() reports EC071")


def test_benchmark():
    print("\n[Test 12] Benchmark: 30-min transient at dt=5 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(T_gen0_c=40.0, dt=5.0, duration_s=1800.0)
    elapsed = time.perf_counter() - t0
    print(f"  1800 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_circulation_ratio,
        test_mass_conservation,
        test_energy_balance_closes,
        test_cop_heating_band,
        test_cop_cooling_band,
        test_all_duties_positive,
        test_property_correlations,
        test_transient_warms_and_settles,
        test_transient_monotone,
        test_higher_drive_raises_gen,
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
    print(f"EC071 Absorption HP F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
