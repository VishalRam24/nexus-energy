"""
EC182 -- Distribution Line Model -- F2a Feeder Voltage-Profile ODE
Test suite: physics sanity (energy conservation, voltage drop, I^2*R losses,
high R/X), edge cases, predict() interface, benchmark timing. NO pytest.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import FeederVoltageProfileModel
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
def test_predict_interface():
    print("\n[Test 1] ComponentModel predict() interface keys present")
    _, cm = make_model()
    r = cm.predict({"V_s_kV": 11.0, "P_total_kW": 1000.0, "Q_total_kVAR": 400.0})
    for key in ["x_km", "V_profile_kV", "I_profile_A", "P_flow_kW", "V_r_kV",
                "P_loss_kW", "efficiency", "voltage_drop_pct", "min_voltage_pu",
                "ansi_compliant", "r_over_x", "energy_balance_residual_kW"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["x_km"]) == len(r["V_profile_kV"]), "Profile arrays same length")
    assert_true(r["converged"], "Shooting BVP converged")


def test_high_r_over_x():
    print("\n[Test 2] High R/X ratio (resistive distribution feeder)")
    m, _ = make_model()
    assert_true(m.r_over_x > 0.5,
                f"R/X={m.r_over_x:.3f} > 0.5 (resistive, unlike transmission R/X<<0.1)")


def test_energy_conservation():
    print("\n[Test 3] Energy conservation: P_send = P_delivered + P_loss")
    _, cm = make_model()
    r = cm.predict({"V_s_kV": 11.0, "P_total_kW": 1500.0, "Q_total_kVAR": 600.0,
                    "length_km": 8.0})
    residual = abs(r["P_send_kW"] - r["P_delivered_kW"] - r["P_loss_kW"])
    assert_true(residual < 1e-6, f"P balance residual = {residual:.3e} kW ~ 0")
    assert_true(r["energy_balance_residual_kW"] < 1e-3,
                f"Internal balance residual = {r['energy_balance_residual_kW']:.3e} kW")


def test_voltage_drops_along_feeder():
    print("\n[Test 4] Voltage falls monotonically along feeder (no Ferranti)")
    _, cm = make_model()
    r = cm.predict({"V_s_kV": 11.0, "P_total_kW": 2000.0, "Q_total_kVAR": 800.0,
                    "length_km": 10.0})
    V = r["V_profile_kV"]
    diffs = np.diff(V)
    assert_true(np.all(diffs <= 1e-9), "V(x) non-increasing (drop, no Ferranti rise)")
    assert_true(V[-1] < V[0], f"V_end={V[-1]:.3f} < V_send={V[0]:.3f} kV")


def test_losses_positive_and_I2R():
    print("\n[Test 5] Losses positive and scale with load^2 (I^2*R)")
    _, cm = make_model()
    r1 = cm.predict({"V_s_kV": 11.0, "P_total_kW": 1000.0, "Q_total_kVAR": 400.0})
    r2 = cm.predict({"V_s_kV": 11.0, "P_total_kW": 2000.0, "Q_total_kVAR": 800.0})
    assert_true(r1["P_loss_kW"] > 0, f"P_loss={r1['P_loss_kW']:.2f} kW > 0")
    # Doubling load (~doubling current) should ~quadruple loss (I^2 R)
    ratio = r2["P_loss_kW"] / r1["P_loss_kW"]
    assert_true(3.0 < ratio < 5.0,
                f"Loss ratio {ratio:.2f} ~ 4x for 2x load (I^2*R scaling)")


def test_longer_line_more_loss_and_drop():
    print("\n[Test 6] Longer feeder => more loss and more voltage drop")
    _, cm = make_model()
    r_short = cm.predict({"V_s_kV": 11.0, "P_total_kW": 1500.0,
                          "Q_total_kVAR": 600.0, "length_km": 2.0})
    r_long = cm.predict({"V_s_kV": 11.0, "P_total_kW": 1500.0,
                         "Q_total_kVAR": 600.0, "length_km": 12.0})
    assert_true(r_long["P_loss_kW"] > r_short["P_loss_kW"],
                f"loss {r_long['P_loss_kW']:.1f} > {r_short['P_loss_kW']:.1f} kW")
    assert_true(r_long["voltage_drop_pct"] > r_short["voltage_drop_pct"],
                f"dV {r_long['voltage_drop_pct']:.2f}% > {r_short['voltage_drop_pct']:.2f}%")


def test_efficiency_range():
    print("\n[Test 7] Efficiency in (0, 1)")
    _, cm = make_model()
    r = cm.predict({"V_s_kV": 11.0, "P_total_kW": 1500.0, "Q_total_kVAR": 600.0,
                    "length_km": 8.0})
    assert_true(0.0 < r["efficiency"] < 1.0, f"eta={r['efficiency']:.4f} in (0,1)")


def test_zero_load_no_loss_no_drop():
    print("\n[Test 8] Zero load => no current, no loss, flat voltage")
    _, cm = make_model()
    r = cm.predict({"V_s_kV": 11.0, "P_total_kW": 0.0, "Q_total_kVAR": 0.0})
    assert_true(r["P_loss_kW"] < 1e-6, f"P_loss={r['P_loss_kW']:.3e} kW ~ 0")
    assert_true(abs(r["voltage_drop_pct"]) < 1e-3,
                f"dV={r['voltage_drop_pct']:.3e}% ~ 0 (flat profile, no shunt)")
    assert_true(r["I_send_A"] < 1e-3, f"I_send={r['I_send_A']:.3e} A ~ 0")


def test_ansi_voltage_band():
    print("\n[Test 9] ANSI C84.1 band: light load compliant, heavy load violates")
    _, cm = make_model()
    r_light = cm.predict({"V_s_kV": 11.0, "P_total_kW": 300.0,
                          "Q_total_kVAR": 120.0, "length_km": 5.0})
    r_heavy = cm.predict({"V_s_kV": 11.0, "P_total_kW": 4500.0,
                          "Q_total_kVAR": 1800.0, "length_km": 20.0})
    assert_true(r_light["ansi_compliant"],
                f"Light load ANSI compliant (min_V={r_light['min_voltage_pu']*100:.1f}%)")
    assert_true(not r_heavy["ansi_compliant"],
                f"Heavy/long ANSI violation (min_V={r_heavy['min_voltage_pu']*100:.1f}%)")


def test_current_decreases_along_feeder():
    print("\n[Test 10] Distributed load: current decreases toward open end -> 0")
    _, cm = make_model()
    r = cm.predict({"V_s_kV": 11.0, "P_total_kW": 2000.0, "Q_total_kVAR": 800.0,
                    "length_km": 10.0, "n_sections": 60})
    I = r["I_profile_A"]
    assert_true(I[0] > I[-1], f"I_send={I[0]:.1f} > I_end={I[-1]:.1f} A")
    assert_true(I[-1] < 1e-3, f"I at open radial end ~ 0 ({I[-1]:.3e} A)")


def test_profile_loss_vs_end_lumped():
    print("\n[Test 11] Distributed load gives less loss than same load at far end")
    _, cm = make_model()
    # distributed total 2000 kW over 10 km
    r_dist = cm.predict({"V_s_kV": 11.0, "P_total_kW": 2000.0,
                         "Q_total_kVAR": 800.0, "length_km": 10.0})
    # concentrate same load at the very end via a very short feeder of equal Z?
    # Physics check: distributed load draws current only over part of the line,
    # so distributed P_loss < lumped-end P_loss for the same total load & length.
    I_send = r_dist["I_send_A"]
    R_total = cm._model.r * 10.0
    lumped_end_loss_kW = 3.0 * I_send ** 2 * R_total / 1000.0
    assert_true(r_dist["P_loss_kW"] < lumped_end_loss_kW,
                f"distributed loss {r_dist['P_loss_kW']:.1f} < lumped-end "
                f"{lumped_end_loss_kW:.1f} kW")


def test_benchmark():
    print("\n[Test 12] Benchmark: 50-section feeder solve")
    _, cm = make_model()
    t0 = time.perf_counter()
    for _ in range(10):
        cm.predict({"V_s_kV": 11.0, "P_total_kW": 1500.0,
                    "Q_total_kVAR": 600.0, "length_km": 8.0, "n_sections": 50})
    elapsed = time.perf_counter() - t0
    print(f"  10 feeder solves in {elapsed*1000:.1f} ms ({elapsed*100:.1f} ms each)")
    assert_true(elapsed < 5.0, "10 solves complete in < 5 s")


if __name__ == "__main__":
    tests = [
        test_predict_interface,
        test_high_r_over_x,
        test_energy_conservation,
        test_voltage_drops_along_feeder,
        test_losses_positive_and_I2R,
        test_longer_line_more_loss_and_drop,
        test_efficiency_range,
        test_zero_load_no_loss_no_drop,
        test_ansi_voltage_band,
        test_current_decreases_along_feeder,
        test_profile_loss_vs_end_lumped,
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
    print(f"EC182 Distribution Line F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
