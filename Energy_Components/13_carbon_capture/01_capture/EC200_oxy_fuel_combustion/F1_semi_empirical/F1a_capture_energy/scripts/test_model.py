"""
EC200 -- Oxy-Fuel Combustion Capture -- F1a Capture & Energy Model
Test suite: physics sanity (conservation, monotonicity), edge cases, interface.
Custom assert_true harness -- NO pytest.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import OxyFuelF1a
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
def test_predict_keys():
    print("\n[Test 1] predict() returns all expected keys")
    _, cm = make_model()
    r = cm.predict({"fuel_rate": 50.0})
    for k in ["o2_demand_kgs", "co2_produced_kgs", "co2_captured_kgs",
              "capture_rate", "asu_power_mw", "compression_power_mw",
              "parasitic_power_mw", "efficiency_drop_pts", "net_efficiency"]:
        assert_true(k in r, f"Key '{k}' present")


def test_get_info():
    print("\n[Test 2] get_info() metadata")
    _, cm = make_model()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC200", "component_id == EC200")
    assert_true("F1a" in info["fidelity"], "fidelity contains F1a")


def test_carbon_conservation():
    print("\n[Test 3] Carbon conservation: CO2 produced matches fuel carbon")
    m, _ = make_model()
    fuel = 50.0
    co2 = float(m.co2_produced(fuel))
    # carbon in = fuel * w_C ; carbon in CO2 = co2 * (MW_C/MW_CO2)
    c_in = fuel * m.w_C
    c_out = co2 * (m.MW_C / m.MW_CO2)
    assert_true(abs(c_in - c_out) / c_in < 1e-6,
                f"C_in={c_in:.3f} == C_out={c_out:.3f} kg/s")


def test_captured_le_produced():
    print("\n[Test 4] CO2 captured <= CO2 produced (mass balance)")
    _, cm = make_model()
    for p in [0.90, 0.95, 0.99]:
        r = cm.predict({"fuel_rate": 50.0, "co2_purity_dry": p})
        assert_true(float(r["co2_captured_kgs"]) <= float(r["co2_produced_kgs"]) + 1e-9,
                    f"purity {p}: captured <= produced")


def test_high_capture_rate():
    print("\n[Test 5] Capture rate high (>0.90) -- near-complete capture")
    _, cm = make_model()
    for p in [0.90, 0.95, 0.99]:
        r = cm.predict({"fuel_rate": 50.0, "co2_purity_dry": p})
        cr = float(r["capture_rate"])
        assert_true(cr > 0.90, f"purity {p}: capture rate {cr:.3f} > 0.90")


def test_o2_demand_realistic():
    print("\n[Test 6] O2/fuel mass ratio realistic for coal (~2.0-2.8)")
    m, _ = make_model()
    ratio = m.stoichiometric_o2()
    assert_true(2.0 < ratio < 2.8, f"stoich O2/fuel = {ratio:.3f} (coal ~2.3)")
    # with excess at full load slightly higher
    o2 = float(m.o2_demand(50.0, 1.0))
    assert_true(o2 / 50.0 > ratio, "actual O2 demand includes excess O2")


def test_energy_penalty_range():
    print("\n[Test 7] Efficiency drop 8-12 pts (Buhre 2005, Toftegaard 2010)")
    _, cm = make_model()
    r = cm.predict({"fuel_rate": 50.0, "load": 1.0, "base_efficiency": 0.40})
    d = float(r["efficiency_drop_pts"])
    assert_true(0.07 < d < 0.13, f"efficiency drop = {d*100:.1f} pts in [7,13]")
    assert_true(float(r["net_efficiency"]) < 0.40,
                "net efficiency below base efficiency")


def test_asu_dominates_penalty():
    print("\n[Test 8] ASU is the dominant energy penalty (> compression)")
    _, cm = make_model()
    r = cm.predict({"fuel_rate": 50.0})
    assert_true(float(r["asu_power_mw"]) > float(r["compression_power_mw"]),
                f"ASU {float(r['asu_power_mw']):.1f} MW > comp {float(r['compression_power_mw']):.1f} MW")


def test_purity_increases_asu():
    print("\n[Test 9] Higher O2 purity raises ASU specific energy")
    m, _ = make_model()
    e_low = float(m.asu_specific_energy(0.90))
    e_high = float(m.asu_specific_energy(0.995))
    assert_true(e_high > e_low, f"ASU energy: {e_high:.1f} > {e_low:.1f} kWh/tO2")
    assert_true(180 < e_low < 260, f"ASU energy {e_low:.1f} in Buhre range 200-240")


def test_partload_penalty():
    print("\n[Test 10] Part load raises specific ASU energy")
    m, _ = make_model()
    e_full = float(m.asu_specific_energy(0.95, load=1.0))
    e_part = float(m.asu_specific_energy(0.95, load=0.5))
    assert_true(e_part > e_full, f"part-load {e_part:.1f} > full {e_full:.1f} kWh/tO2")


def test_co2_proportional_to_fuel():
    print("\n[Test 11] CO2 produced scales linearly with fuel rate")
    _, cm = make_model()
    fuels = np.array([10.0, 30.0, 60.0, 90.0])
    co2 = np.array([float(cm.predict({"fuel_rate": float(f)})["co2_produced_kgs"])
                    for f in fuels])
    ratios = co2 / fuels
    assert_true(np.std(ratios) / np.mean(ratios) < 1e-6,
                "CO2 strictly proportional to fuel")


def test_array_input():
    print("\n[Test 12] Array input handled")
    _, cm = make_model()
    fuels = np.linspace(5.0, 80.0, 10)
    r = cm.predict({"fuel_rate": fuels})
    assert_true(len(np.atleast_1d(r["co2_captured_kgs"])) == 10, "array output length 10")


def test_benchmark():
    print("\n[Test 13] Benchmark: 1000 predictions")
    _, cm = make_model()
    fuels = np.random.uniform(1, 100, 1000)
    loads = np.random.uniform(0.4, 1.0, 1000)
    t0 = time.perf_counter()
    cm.predict({"fuel_rate": fuels, "load": loads})
    elapsed = time.perf_counter() - t0
    print(f"  1000 predictions in {elapsed*1000:.2f} ms")
    assert_true(elapsed < 1.0, "completes in < 1 s")


if __name__ == "__main__":
    tests = [
        test_predict_keys,
        test_get_info,
        test_carbon_conservation,
        test_captured_le_produced,
        test_high_capture_rate,
        test_o2_demand_realistic,
        test_energy_penalty_range,
        test_asu_dominates_penalty,
        test_purity_increases_asu,
        test_partload_penalty,
        test_co2_proportional_to_fuel,
        test_array_input,
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
    print(f"EC200 Oxy-Fuel F1a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
