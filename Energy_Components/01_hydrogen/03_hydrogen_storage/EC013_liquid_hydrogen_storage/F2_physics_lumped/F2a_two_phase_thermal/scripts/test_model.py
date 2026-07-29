"""
EC013 -- Liquid Hydrogen (LH2) Storage -- F2a Two-Phase Cryogenic Tank
Test suite: conservation laws, monotonic self-pressurization, saturation-line
sanity, boil-off magnitude, predict() interface, benchmark timing.

Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import LH2TwoPhaseTank
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
def test_saturation_line():
    print("\n[Test 1] Antoine saturation line anchored at NBP (NIST)")
    m, _ = make_model()
    P_nbp = float(m.p_sat(20.28))
    assert_true(abs(P_nbp - 1.013) < 0.06,
                f"P_sat(20.28 K)={P_nbp:.4f} bar ~ 1 atm")
    # round-trip inverse
    T_back = float(m.t_sat(1.01325))
    assert_true(abs(T_back - 20.28) < 0.3, f"t_sat(1 atm)={T_back:.3f} K ~ 20.28")
    # monotone increasing P with T
    assert_true(m.p_sat(24.0) > m.p_sat(20.28), "P_sat rises with T")


def test_property_signs():
    print("\n[Test 2] Saturated-property slopes physical toward critical point")
    m, _ = make_model()
    assert_true(m.h_vap(24.0) < m.h_vap(20.28), "latent heat declines with T")
    assert_true(m.rho_liquid(24.0) < m.rho_liquid(20.28),
                "liquid density falls with T")
    assert_true(m.rho_vapor_ideal(24.0, float(m.p_sat(24.0))) >
                m.rho_vapor_ideal(20.28, 1.01325), "vapor density rises with T")


def test_heat_leak_direction():
    print("\n[Test 3] Heat leaks INWARD (T_amb >> T_sat) and grows with T_amb")
    m, _ = make_model()
    q1 = float(m.heat_leak(298.15, 20.28))
    q2 = float(m.heat_leak(333.15, 20.28))
    assert_true(q1 > 0, f"Q_in={q1:.3f} W > 0 (heat enters cold tank)")
    assert_true(q2 > q1, f"Q_in rises with ambient: {q2:.3f} > {q1:.3f} W")
    assert_true(abs(float(m.heat_leak(20.28, 20.28))) < 1e-9,
                "Q_in = 0 when no temperature difference")


def test_mass_conservation_sealed():
    print("\n[Test 4] Sealed tank conserves total mass (no vent)")
    m, _ = make_model()
    r = m.simulate(0.90, 298.15, 1.01325, duration_s=7 * 86400.0, sealed=True)
    dm = abs(r["m_total"][-1] - r["m_total"][0])
    assert_true(dm < 1e-6 * r["m_total"][0],
                f"|dm_tot|={dm:.2e} kg ~ 0 over 7 days sealed")
    assert_true(r["success"], "integrator converged")


def test_volume_constraint():
    print("\n[Test 5] Fixed-volume constraint V_L+V_V = V_tank held exactly")
    m, _ = make_model()
    r = m.simulate(0.90, 298.15, 1.01325, duration_s=7 * 86400.0, sealed=True)
    rhoL = m.rho_liquid(r["temperature"])
    rhoV = m.rho_vapor_ideal(r["temperature"], r["pressure"])
    V = r["m_liquid"] / rhoL + r["m_vapor"] / rhoV
    err = np.max(np.abs(V - m.V_tank))
    assert_true(err < 1e-4, f"max |V - V_tank|={err:.2e} m3 (constraint satisfied)")


def test_monotonic_self_pressurization():
    print("\n[Test 6] Sealed tank self-pressurizes monotonically (Van Dresar 1993)")
    m, _ = make_model()
    r = m.simulate(0.90, 298.15, 1.01325, duration_s=7 * 86400.0, sealed=True)
    P = r["pressure"]
    T = r["temperature"]
    assert_true(np.all(np.diff(P) >= -1e-9), "pressure non-decreasing while sealed")
    assert_true(np.all(np.diff(T) >= -1e-9), "temperature non-decreasing while sealed")
    assert_true(P[-1] > P[0] + 0.05, f"P rose {P[0]:.3f} -> {P[-1]:.3f} bar")


def test_energy_conservation_sealed():
    print("\n[Test 7] Sealed energy balance: dU = integral(Q_in dt)")
    m, _ = make_model()
    r = m.simulate(0.90, 298.15, 1.01325, duration_s=5 * 86400.0,
                   n_steps=600, sealed=True)
    # U_tot reconstructed from state
    U = np.array([m.total_internal_energy(ml, mv, T)
                  for ml, mv, T in zip(r["m_liquid"], r["m_vapor"], r["temperature"])])
    dU = U[-1] - U[0]
    Q_int = np.trapezoid(r["heat_leak_W"], r["t"])   # J
    rel = abs(dU - Q_int) / abs(Q_int)
    assert_true(rel < 0.05, f"dU={dU:.3e} J vs integral Q_in={Q_int:.3e} J (rel {rel:.3%})")


def test_open_vent_constant_pressure():
    print("\n[Test 8] Open-vent mode holds pressure ~ constant (NBP boil-off)")
    m, _ = make_model()
    r = m.simulate(0.90, 298.15, 1.01325, duration_s=86400.0, sealed=False)
    dP = abs(r["pressure"][-1] - r["pressure"][0])
    assert_true(dP < 0.05, f"open vent dP={dP:.4f} bar ~ 0")
    # liquid is lost to venting
    assert_true(r["m_total"][-1] < r["m_total"][0],
                "mass is vented in open-vent mode")


def test_bor_magnitude():
    print("\n[Test 9] Boil-off rate in realistic band for MLI dewar")
    m, _ = make_model()
    s = m.steady_boiloff(0.90, 298.15, 1.01325)
    bor = s["BOR_pct_day"]
    assert_true(0.05 < bor < 5.0, f"BOR={bor:.4f} %/day in [0.05, 5] (MLI dewar band)")
    assert_true(s["heat_leak_W"] > 0, f"Q_in={s['heat_leak_W']:.3f} W > 0")
    # hotter ambient -> more boil-off
    bor_hot = m.steady_boiloff(0.90, 333.15, 1.01325)["BOR_pct_day"]
    assert_true(bor_hot > bor, f"BOR rises with T_amb: {bor_hot:.4f} > {bor:.4f}")


def test_dormancy_time_to_vent():
    print("\n[Test 10] Self-pressurization reaches vent set-point in finite time")
    m, _ = make_model()
    r = m.simulate(0.90, 298.15, 1.01325, duration_s=60 * 86400.0,
                   n_steps=800, sealed=True)
    reached = np.any(r["pressure"] >= m.P_vent - 0.01)
    assert_true(reached, f"P reached vent set-point {m.P_vent} bar within 60 days")
    assert_true(np.all(r["pressure"] <= m.P_max + 0.2),
                "pressure never exceeds design MAWP by margin")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface keys + shapes")
    _, cm = make_model()
    r = cm.predict({"fill_fraction": 0.85, "duration_s": 3600.0,
                    "n_steps": 50, "sealed": True})
    for key in ["t", "m_liquid", "m_vapor", "m_total", "temperature",
                "pressure", "heat_leak_W", "boiloff_rate_kg_s", "BOR_pct_day",
                "fill_fraction", "energy_stored_MJ"]:
        assert_true(key in r, f"Key '{key}' in output")
    n = len(r["t"])
    assert_true(all(len(r[k]) == n for k in
                    ["temperature", "pressure", "m_liquid", "BOR_pct_day"]),
                "all time-series arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC013" and info["version"] == "1.0.0",
                "get_info metadata correct")


def test_benchmark():
    print("\n[Test 12] Benchmark: 7-day sealed dormancy sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.90, 298.15, 1.01325, duration_s=7 * 86400.0, n_steps=400, sealed=True)
    elapsed = time.perf_counter() - t0
    print(f"  7-day sim ({400} samples) in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_saturation_line,
        test_property_signs,
        test_heat_leak_direction,
        test_mass_conservation_sealed,
        test_volume_constraint,
        test_monotonic_self_pressurization,
        test_energy_conservation_sealed,
        test_open_vent_constant_pressure,
        test_bor_magnitude,
        test_dormancy_time_to_vent,
        test_predict_interface,
        test_benchmark,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")

    print(f"\n{'='*64}")
    print(f"EC013 LH2 Storage F2a (two-phase) -- Results: {passed} passed, {failed} failed")
    print(f"{'='*64}")
    sys.exit(0 if failed == 0 else 1)
