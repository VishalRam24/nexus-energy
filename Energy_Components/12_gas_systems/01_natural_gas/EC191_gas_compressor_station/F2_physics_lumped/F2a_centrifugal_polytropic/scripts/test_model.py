"""
EC191 -- Gas Compressor Station -- F2a Centrifugal Polytropic
Test suite: physics sanity (energy/head/temperature), surge/choke limits,
real-gas Z, ODE transient, predict() interface, benchmark timing.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import NGCompressorF2a
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
def test_z_factor_physical():
    print("\n[Test 1] Real-gas Z in (0,1] and decreases with pressure")
    m, _ = make_model()
    Z_lo = m.z_factor(50.0, 288.15)
    Z_hi = m.z_factor(150.0, 288.15)
    assert_true(0.25 < Z_lo <= 1.05, f"Z(50bar)={Z_lo:.3f} in (0.25,1.05]")
    assert_true(Z_hi < Z_lo, f"Z(150bar)={Z_hi:.3f} < Z(50bar)={Z_lo:.3f}")
    assert_true(abs(Z_lo - m.Z_inlet) < 0.03,
                f"Z(suction)={Z_lo:.3f} ~ cited Z_inlet={m.Z_inlet}")


def test_head_curve_peak_at_design():
    print("\n[Test 2] Head coefficient psi maximal at design phi")
    m, _ = make_model()
    psi_design = m.head_coefficient(m.phi_design)
    for phi in [m.phi_design * 0.6, m.phi_design * 1.4]:
        assert_true(m.head_coefficient(phi) < psi_design,
                    f"psi({phi:.3f})={m.head_coefficient(phi):.3f} < psi_design={psi_design:.3f}")


def test_pressure_ratio_above_one():
    print("\n[Test 3] Polytropic compression gives PR > 1")
    m, _ = make_model()
    for md in [60.0, 100.0, 120.0]:
        op = m.operating_point(md)
        assert_true(op["pressure_ratio"] > 1.0,
                    f"PR({md} kg/s)={op['pressure_ratio']:.3f} > 1")


def test_discharge_temp_above_inlet():
    print("\n[Test 4] T_discharge > T_inlet (compression heats gas)")
    m, _ = make_model()
    for md in [60.0, 100.0]:
        op = m.operating_point(md)
        assert_true(op["T_discharge_K"] > m.T_inlet + 1.0,
                    f"T_disch({md})={op['T_discharge_K']:.1f} K > T_in={m.T_inlet:.1f} K")


def test_power_positive_and_increases_with_flow():
    print("\n[Test 5] Shaft power > 0 and rises with mass flow")
    m, _ = make_model()
    W_prev = -1.0
    for md in [40.0, 60.0, 80.0, 100.0]:
        op = m.operating_point(md)
        assert_true(op["shaft_power_W"] > 0, f"W({md})={op['shaft_power_MW']:.2f} MW > 0")
        assert_true(op["shaft_power_W"] > W_prev, f"W rises: {op['shaft_power_MW']:.2f} MW")
        W_prev = op["shaft_power_W"]


def test_fuel_power_exceeds_shaft():
    print("\n[Test 6] Driver fuel power > shaft power (eta_driver < 1)")
    m, _ = make_model()
    op = m.operating_point(100.0)
    assert_true(op["fuel_power_MW"] > op["shaft_power_MW"],
                f"fuel={op['fuel_power_MW']:.2f} > shaft={op['shaft_power_MW']:.2f} MW")


def test_surge_limit():
    print("\n[Test 7] Surge flagged at low flow, clear at design flow")
    m, _ = make_model()
    assert_true(m.operating_point(30.0)["in_surge"], "30 kg/s in surge")
    assert_true(not m.operating_point(100.0)["in_surge"], "100 kg/s out of surge")


def test_choke_limit():
    print("\n[Test 8] Choke flagged at very high flow, clear at design flow")
    m, _ = make_model()
    assert_true(m.operating_point(180.0)["in_choke"], "180 kg/s in choke")
    assert_true(not m.operating_point(100.0)["in_choke"], "100 kg/s out of choke")


def test_energy_conservation():
    print("\n[Test 9] Energy balance: shaft work bounds sensible enthalpy rise")
    m, _ = make_model()
    op = m.operating_point(100.0)
    w_spec = op["H_poly_J_per_kg"] / m.eta_mech       # J/kg shaft
    dh = m.cp * (op["T_discharge_K"] - m.T_inlet)      # J/kg sensible
    # polytropic head < sensible enthalpy rise (efficiency < 1), both same order
    assert_true(0 < w_spec < dh,
                f"0 < shaft {w_spec:.0f} < dh {dh:.0f} J/kg (eta loss accounted)")
    assert_true(w_spec / dh > 0.5,
                f"shaft/dh={w_spec/dh:.2f} > 0.5 (consistent magnitude)")


def test_transient_pressure_settles():
    print("\n[Test 10] Plenum pressure ODE settles toward PR*P_in")
    m, _ = make_model()
    r = m.simulate(100.0, 1.0, None, None, 0.5, 180.0)
    P_final = r["P_discharge_bar"][-1]
    op = m.operating_point(100.0)
    P_target = op["pressure_ratio"] * m.P_inlet
    assert_true(abs(P_final - P_target) / P_target < 0.05,
                f"P_final={P_final:.1f} ~ target {P_target:.1f} bar (<5%)")
    dP = abs(r["P_discharge_bar"][-1] - r["P_discharge_bar"][-2])
    assert_true(dP < 0.05, f"Near steady: dP={dP:.4f} bar between last steps")


def test_transient_temp_above_inlet():
    print("\n[Test 11] Discharge temperature stays above inlet through transient")
    m, _ = make_model()
    r = m.simulate(80.0, 1.0, None, None, 0.5, 120.0)
    assert_true(np.all(r["T_discharge_K"] > m.T_inlet),
                f"min T_disch={r['T_discharge_K'].min():.1f} > T_in={m.T_inlet:.1f} K")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"mass_flow_kg_s": 100.0, "dt": 0.5, "duration_s": 20.0})
    for key in ["t", "P_discharge_bar", "T_discharge_K", "pressure_ratio",
                "shaft_power_MW", "in_surge", "in_choke"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["P_discharge_bar"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC191", "get_info component_id correct")


def test_benchmark():
    print("\n[Test 13] Benchmark: 120 s sim at dt=0.1")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(100.0, 1.0, None, None, 0.1, 120.0)
    elapsed = time.perf_counter() - t0
    print(f"  120 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_z_factor_physical,
        test_head_curve_peak_at_design,
        test_pressure_ratio_above_one,
        test_discharge_temp_above_inlet,
        test_power_positive_and_increases_with_flow,
        test_fuel_power_exceeds_shaft,
        test_surge_limit,
        test_choke_limit,
        test_energy_conservation,
        test_transient_pressure_settles,
        test_transient_temp_above_inlet,
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
    print(f"EC191 Gas Compressor Station F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
