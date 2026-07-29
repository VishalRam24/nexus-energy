"""
EC012 -- Compressed Gas H2 Storage -- F2a Thermodynamic Tank
Test suite: EOS sanity, mass & energy conservation, first-law fill/discharge
physics, wall heat transfer, predict() interface, benchmark timing.
"""

import sys
import os
import time
import numpy as np
from scipy.integrate import trapezoid

sys.path.insert(0, os.path.dirname(__file__))
from model import CompressedGasH2F2a
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
def test_compressibility_real_gas():
    print("\n[Test 1] Real-gas Z(T,P): H2 has Z>1, matches NIST refit anchors")
    m, _ = make_model()
    Z_700_300 = float(m.compressibility_factor(700.0, 300.0))
    Z_700_240 = float(m.compressibility_factor(700.0, 240.0))
    Z_100_300 = float(m.compressibility_factor(100.0, 300.0))
    assert_true(Z_700_300 > 1.0, f"Z(700bar,300K)={Z_700_300:.3f} > 1")
    assert_true(abs(Z_700_300 - 1.40) < 0.06, f"Z(700bar,300K)={Z_700_300:.3f} ~ 1.40 (NIST)")
    assert_true(Z_700_240 > Z_700_300, f"colder gas less ideal: {Z_700_240:.3f} > {Z_700_300:.3f}")
    assert_true(abs(Z_100_300 - 1.065) < 0.02, f"Z(100bar,300K)={Z_100_300:.3f} ~ 1.065 (NIST)")


def test_eos_roundtrip():
    print("\n[Test 2] EOS round-trip: mass(P,T) -> pressure(m,T) is self-consistent")
    m, _ = make_model()
    for P in [50.0, 350.0, 700.0]:
        for T in [250.0, 298.15, 350.0]:
            mass = float(m.mass_from_PT(P, T))
            P_back = m.pressure(mass, T)
            assert_true(abs(P_back - P) < 1e-3 * P,
                        f"P={P:.0f},T={T:.0f}: round-trip P={P_back:.3f} bar")


def test_mass_conservation_fill():
    print("\n[Test 3] Mass conservation: dm = mdot * duration on fill")
    m, _ = make_model()
    mdot = 0.008
    dur = 100.0
    r = m.simulate(mdot, 298.15, P0_bar=20.0, dt=1.0, duration_s=dur)
    dm = r["mass"][-1] - r["mass"][0]
    assert_true(abs(dm - mdot * dur) < 1e-3 * mdot * dur,
                f"dm={dm:.4f} kg vs mdot*t={mdot*dur:.4f} kg")


def test_mass_conservation_discharge():
    print("\n[Test 4] Mass conservation: discharge removes mdot*t")
    m, _ = make_model()
    m0 = float(m.mass_from_PT(700.0, 298.15))
    mdot = -0.005
    dur = 100.0
    r = m.simulate(mdot, 298.15, m0_kg=m0, dt=1.0, duration_s=dur)
    dm = r["mass"][-1] - r["mass"][0]
    assert_true(abs(dm - mdot * dur) < 1e-3 * abs(mdot * dur),
                f"dm={dm:.4f} kg vs mdot*t={mdot*dur:.4f} kg")
    assert_true(r["pressure"][-1] < r["pressure"][0], "Pressure falls during discharge")


def test_fill_heats_gas():
    print("\n[Test 5] Fast fill heats the gas (flow-work / heat of compression)")
    m, _ = make_model()
    # adiabatic-ish fast fill: high flow, short time so wall loss is small
    r = m.simulate(0.02, 298.15, T_in_K=298.15, P0_bar=20.0, dt=0.5, duration_s=60.0)
    dT = r["temperature"][-1] - r["temperature"][0]
    assert_true(dT > 10.0, f"Gas warms on fast fill: dT={dT:.1f} K")
    assert_true(dT < 120.0, f"dT={dT:.1f} K physically bounded (<120 K)")


def test_pressure_rises_on_fill():
    print("\n[Test 6] Pressure & SOC rise monotonically during fill")
    m, _ = make_model()
    r = m.simulate(0.006, 298.15, P0_bar=20.0, dt=2.0, duration_s=120.0)
    assert_true(np.all(np.diff(r["pressure"]) > 0), "P strictly increasing")
    assert_true(np.all(np.diff(r["soc"]) >= -1e-9), "SOC non-decreasing")
    assert_true(r["pressure"][-1] > r["pressure"][0], "Net pressure rise")


def test_wall_cooldown_to_ambient():
    print("\n[Test 7] No-flow post-fill: gas cools toward ambient via wall")
    m, _ = make_model()
    m0 = float(m.mass_from_PT(700.0, 298.15))
    # start hot (post fast-fill), no flow, long relaxation. The combined
    # gas+wall thermal mass with the small wall->ambient UA gives a time
    # constant of order 1e4 s, so integrate several constants out.
    r = m.simulate(0.0, 360.0, T_amb_K=298.15, m0_kg=m0, dt=50.0, duration_s=80000.0)
    assert_true(r["temperature"][-1] < r["temperature"][0], "Gas cools when hot")
    assert_true(abs(r["temperature"][-1] - 298.15) < 5.0,
                f"Approaches ambient: T_end={r['temperature'][-1]:.1f} K -> 298.15")
    # cooling at fixed mass lowers pressure (density-temperature coupling)
    assert_true(r["pressure"][-1] < r["pressure"][0], "Pressure drops as gas cools")


def test_energy_balance_no_flow():
    print("\n[Test 8] First-law check: closed tank energy change = -integral(Q_wall)")
    m, _ = make_model()
    m0 = float(m.mass_from_PT(700.0, 298.15))
    T_amb = 298.15
    r = m.simulate(0.0, 360.0, T_amb_K=T_amb, m0_kg=m0, dt=2.0, duration_s=2000.0)
    # gas internal energy change
    dU_gas = m0 * m.cv_H2 * (r["temperature"][-1] - r["temperature"][0])
    # integral of heat leaving the GAS to the wall: Q_gw = UA_gw*(T_gas - T_wall)
    Q_gw = m.UA_gw * (r["temperature"] - r["T_wall"])
    Q_integral = trapezoid(Q_gw, r["t"])
    # closed system: dU_gas = -integral Q_gw  (no flow work)
    rel_err = abs(dU_gas + Q_integral) / max(abs(dU_gas), 1.0)
    assert_true(rel_err < 0.02, f"dU_gas={dU_gas:.0f} J, -intQ={-Q_integral:.0f} J, rel_err={rel_err:.4f}")


def test_soc_bounds():
    print("\n[Test 9] SOC stays in [0,1]; full tank ~ SOC 1, empty ~ SOC 0")
    m, _ = make_model()
    m_full = float(m.mass_from_PT(700.0, 298.15))
    m_empty = float(m.mass_from_PT(20.0, 298.15))
    assert_true(abs(float(m.soc(m_full)) - 1.0) < 1e-6, "SOC(P_max)=1")
    assert_true(abs(float(m.soc(m_empty)) - 0.0) < 1e-6, "SOC(P_min)=0")
    assert_true(float(m.soc(m_full * 2.0)) == 1.0, "SOC clipped at 1")
    assert_true(float(m.soc(0.0)) == 0.0, "SOC clipped at 0")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface & metadata")
    _, cm = make_model()
    r = cm.predict({"mdot_kg_s": 0.005, "P0_bar": 30.0, "dt": 2.0, "duration_s": 20.0})
    for key in ["t", "mass", "temperature", "T_wall", "pressure", "density",
                "soc", "energy_MJ", "mdot"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["pressure"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC012", "component_id EC012")
    assert_true(info["version"] == "1.0.0", "version 1.0.0")


def test_energy_stored_matches_lhv():
    print("\n[Test 11] Stored chemical energy = mass * LHV")
    m, _ = make_model()
    mass = float(m.mass_from_PT(700.0, 298.15))
    E = float(m.energy_stored(mass))
    assert_true(abs(E - mass * 120.0) < 1e-6, f"E={E:.1f} MJ = m*LHV")
    assert_true(E > 0, "Positive stored energy")


def test_benchmark():
    print("\n[Test 12] Benchmark: 600 s fill at dt=1.0")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.004, 298.15, P0_bar=20.0, dt=1.0, duration_s=600.0)
    elapsed = time.perf_counter() - t0
    print(f"  600 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_compressibility_real_gas,
        test_eos_roundtrip,
        test_mass_conservation_fill,
        test_mass_conservation_discharge,
        test_fill_heats_gas,
        test_pressure_rises_on_fill,
        test_wall_cooldown_to_ambient,
        test_energy_balance_no_flow,
        test_soc_bounds,
        test_predict_interface,
        test_energy_stored_matches_lhv,
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
    print(f"EC012 H2 Storage F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
