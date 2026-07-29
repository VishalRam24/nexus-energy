"""
EC033 -- Iron-Air Battery (Fe-Air) -- F2a Physics-Lumped Electrochemical
Test suite: physics sanity, conservation, efficiency bounds, ODE, edge cases.
Run: python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import IronAirF2a
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
def test_ocv_range():
    print("\n[Test 1] OCV in physical Fe-air range ~1.0-1.3 V")
    m, _ = make_model()
    E = m.ocv(298.15)
    assert_true(1.0 < E < 1.3, f"OCV={E:.4f} V in (1.0, 1.3)")
    # entropic term raises OCV slightly with T (dOCV/dT > 0)
    assert_true(m.ocv(320.0) > E, "OCV rises with T (positive entropic coeff)")


def test_voltage_gap_ordering():
    print("\n[Test 2] V_discharge < OCV < V_charge (large gap)")
    m, _ = make_model()
    for j in [0.005, 0.02, 0.04]:
        E = m.ocv(298.15)
        Vd = m.cell_voltage(+j, 298.15)
        Vc = m.cell_voltage(-j, 298.15)
        assert_true(Vd < E < Vc, f"j={j}: Vd={Vd:.3f} < E={E:.3f} < Vc={Vc:.3f}")


def test_voltage_monotone():
    print("\n[Test 3] Discharge V decreases, charge V increases with |j|")
    m, _ = make_model()
    js = np.linspace(0.001, 0.05, 40)
    Vd = [m.cell_voltage(+j, 298.15) for j in js]
    Vc = [m.cell_voltage(-j, 298.15) for j in js]
    assert_true(all(Vd[i + 1] <= Vd[i] + 1e-9 for i in range(len(Vd) - 1)),
                "Discharge voltage monotonically decreasing")
    assert_true(all(Vc[i + 1] >= Vc[i] - 1e-9 for i in range(len(Vc) - 1)),
                "Charge voltage monotonically increasing")


def test_efficiencies_below_one():
    print("\n[Test 4] Voltaic, coulombic, round-trip efficiency all in (0,1)")
    m, _ = make_model()
    for j in [0.005, 0.02, 0.04]:
        ev = m.voltaic_efficiency(j, 298.15)
        ce, _, _ = m.coulombic_efficiency_charge(j, 298.15)
        ert = m.round_trip_efficiency(j, 298.15)
        assert_true(0.0 < ev < 1.0, f"voltaic={ev:.4f} in (0,1)")
        assert_true(0.0 < ce <= 1.0, f"coulombic={ce:.4f} in (0,1]")
        assert_true(0.0 < ert < 1.0, f"round-trip={ert:.4f} in (0,1)")
    # Characteristic LOW round-trip for Fe-air
    ert_mid = m.round_trip_efficiency(0.02, 298.15)
    assert_true(ert_mid < 0.75, f"Fe-air RTE is low: {ert_mid:.3f} < 0.75")


def test_her_reduces_coulombic():
    print("\n[Test 5] Parasitic HER reduces coulombic eff (more at higher charge rate)")
    m, _ = make_model()
    ce_lo, _, her_lo = m.coulombic_efficiency_charge(0.005, 298.15)
    ce_hi, _, her_hi = m.coulombic_efficiency_charge(0.05, 298.15)
    assert_true(her_hi > her_lo > 0.0, f"HER grows with charge rate: {her_lo:.2e} -> {her_hi:.2e}")
    assert_true(ce_hi < ce_lo, f"CE drops at high rate: {ce_hi:.4f} < {ce_lo:.4f}")
    assert_true(ce_hi < 1.0, "CE strictly below 1 (H2 loss)")


def test_concentration_air_transport():
    print("\n[Test 6] Air-cathode O2 transport overpotential diverges near j_L")
    m, _ = make_model()
    jL = m.j_L_O2
    v1 = m.concentration_overpotential(0.5 * jL, 298.15, charging=False)
    v2 = m.concentration_overpotential(0.95 * jL, 298.15, charging=False)
    assert_true(v2 > v1 > 0.0, f"eta_conc(0.95 jL)={v2:.4f} >> eta_conc(0.5 jL)={v1:.4f}")
    # OER charge side is not O2-supply limited
    assert_true(m.concentration_overpotential(0.5 * jL, 298.15, charging=True) == 0.0,
                "No O2 transport limit on charge (OER evolves O2)")


def test_coulomb_conservation():
    print("\n[Test 7] Coulomb conservation: stored = charge_in * CE")
    m, _ = make_model()
    j = -0.005  # charging at a low rate so SOC does not saturate at 1.0
    dt, dur, soc0 = 60.0, 1200.0, 0.3
    r = m.simulate(j, 298.15, dt, dur, soc_init=soc0)
    dsoc = r["soc"][-1] - soc0
    I = abs(j) * m.A_cell
    ce = r["coulombic_eff"][0]
    Q_cap_C = m.capacity_ref * 3600.0
    dsoc_expected = (I * ce * dur) / Q_cap_C
    assert_true(r["soc"][-1] < 1.0 - 1e-6, "SOC stays below saturation for clean check")
    assert_true(abs(dsoc - dsoc_expected) < 1e-3,
                f"dSOC={dsoc:.5f} matches charge*CE={dsoc_expected:.5f}")
    assert_true(dsoc < (I * dur) / Q_cap_C,
                "Stored charge < raw charge passed (H2 loss accounted)")


def test_thermal_balance_heats():
    print("\n[Test 8] Thermal ODE: irreversible heat warms cell; bounded")
    m, _ = make_model()
    r = m.simulate(0.04, 298.15, 30.0, 3600.0, soc_init=0.8)
    assert_true(r["temperature"][-1] > 298.15, f"T rises: {r['temperature'][-1]:.2f} > 298.15")
    assert_true(r["temperature"][-1] < 360.0, f"T bounded: {r['temperature'][-1]:.2f} < 360 K")


def test_thermal_steady_state():
    print("\n[Test 9] Thermal approaches steady state (Q_gen ~ Q_loss)")
    m, _ = make_model()
    r = m.simulate(0.02, 298.15, 60.0, 36000.0, soc_init=0.9)
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.05, f"Near steady state: dT={dT:.5f} K between last steps")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface + SOC discharge drops")
    _, cm = make_model()
    r = cm.predict({"current_density_A_cm2": 0.02, "dt": 60.0, "duration_s": 600.0,
                    "soc_init": 0.6})
    for key in ["t", "voltage", "ocv", "power_density", "soc", "temperature",
                "coulombic_eff", "her_current", "overpotentials"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["voltage"]), "Arrays same length")
    assert_true(r["soc"][-1] < r["soc"][0], "Discharge lowers SOC")


def test_zero_current_at_ocv():
    print("\n[Test 11] Zero current => terminal voltage equals OCV, no heating drive")
    m, _ = make_model()
    V = m.cell_voltage(0.0, 298.15)
    assert_true(abs(V - m.ocv(298.15)) < 1e-12, f"V(0)={V:.5f} == OCV")
    r = m.simulate(0.0, 320.0, 60.0, 3600.0)
    assert_true(r["temperature"][-1] < 320.0 + 1e-6,
                "No self-heating at I=0 (cools toward ambient)")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h sim at dt=10 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.02, 298.15, 10.0, 3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  1 h simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_ocv_range,
        test_voltage_gap_ordering,
        test_voltage_monotone,
        test_efficiencies_below_one,
        test_her_reduces_coulombic,
        test_concentration_air_transport,
        test_coulomb_conservation,
        test_thermal_balance_heats,
        test_thermal_steady_state,
        test_predict_interface,
        test_zero_current_at_ocv,
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
    print(f"EC033 Iron-Air F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
