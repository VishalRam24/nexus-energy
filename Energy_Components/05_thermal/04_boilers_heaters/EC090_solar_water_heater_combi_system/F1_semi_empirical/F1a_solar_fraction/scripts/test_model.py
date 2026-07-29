"""EC090 -- Solar Water Heater Combi System -- F1a Solar Fraction -- Test Suite"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"


def assert_true(condition, message):
    if condition:
        print(f"  {PASS}  {message}")
    else:
        print(f"  {FAIL}  FAILED: {message}")
        raise AssertionError(message)


def test_instantiation():
    print("Test: instantiation")
    m = ComponentModel()
    assert_true(m is not None, "ComponentModel instantiates")
    info = m.get_info()
    assert_true(info["ec_id"] == "EC090", "ec_id == EC090")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")


def test_output_keys():
    print("Test: output keys")
    m = ComponentModel()
    r = m.predict({"G_W_m2": 500.0})
    for k in ["Q_solar_W", "Q_aux_input_W", "Q_aux_delivered_W", "f_solar", "Q_demand_W", "eta_system"]:
        assert_true(k in r, f"key '{k}' in output")


def test_solar_formula():
    print("Test: Q_solar = eta_coll * G * A")
    m = ComponentModel()
    G = 800.0
    eta = 0.50
    A = 6.0
    r = m.predict({"G_W_m2": G})
    expected = min(eta * G * A, 10000.0)  # clamped to demand
    assert_true(abs(r["Q_solar_W"] - expected) < 0.01,
                f"Q_solar = {expected:.2f}W (got {r['Q_solar_W']:.2f})")


def test_zero_irradiance_no_solar():
    print("Test: zero irradiance -> Q_solar = 0, f_solar = 0")
    m = ComponentModel()
    r = m.predict({"G_W_m2": 0.0})
    assert_true(r["Q_solar_W"] == 0.0, "Q_solar=0 at G=0")
    assert_true(r["f_solar"] == 0.0, "f_solar=0 at G=0")


def test_high_irradiance_full_solar():
    print("Test: very high irradiance -> f_solar = 1.0")
    m = ComponentModel()
    r = m.predict({"G_W_m2": 10000.0})  # 50kW >> 10kW demand
    assert_true(abs(r["f_solar"] - 1.0) < 1e-6,
                f"f_solar = 1.0 at high G (got {r['f_solar']:.6f})")
    assert_true(r["Q_aux_input_W"] == 0.0, "Q_aux = 0 when solar covers demand")


def test_solar_fraction_bounded():
    print("Test: f_solar in [0, 1] for various G")
    m = ComponentModel()
    for G in [0, 100, 500, 800, 1200]:
        r = m.predict({"G_W_m2": float(G)})
        assert_true(0.0 <= r["f_solar"] <= 1.0,
                    f"f_solar in [0,1] at G={G} (got {r['f_solar']:.4f})")


def test_aux_covers_shortfall():
    print("Test: Q_aux_delivered = Q_demand - Q_solar")
    m = ComponentModel()
    Q_demand = 10000.0
    r = m.predict({"G_W_m2": 400.0, "Q_demand_W": Q_demand})
    shortfall = Q_demand - r["Q_solar_W"]
    assert_true(abs(r["Q_aux_delivered_W"] - shortfall) < 0.01,
                f"Q_aux_delivered = Q_demand - Q_solar (diff={abs(r['Q_aux_delivered_W'] - shortfall):.4f})")


def test_solar_increases_with_irradiance():
    print("Test: Q_solar increases with irradiance")
    m = ComponentModel()
    r200 = m.predict({"G_W_m2": 200.0, "Q_demand_W": 50000.0})
    r800 = m.predict({"G_W_m2": 800.0, "Q_demand_W": 50000.0})
    assert_true(r800["Q_solar_W"] > r200["Q_solar_W"],
                "Q_solar increases with G")


def test_benchmark():
    print("Test: benchmark 1000 predictions < 1s")
    m = ComponentModel()
    start = time.perf_counter()
    for i in range(1000):
        G = float(i % 1200)
        m.predict({"G_W_m2": G})
    elapsed = time.perf_counter() - start
    print(f"    1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions complete in < 1s")


if __name__ == "__main__":
    tests = [
        test_instantiation,
        test_output_keys,
        test_solar_formula,
        test_zero_irradiance_no_solar,
        test_high_irradiance_full_solar,
        test_solar_fraction_bounded,
        test_aux_covers_shortfall,
        test_solar_increases_with_irradiance,
        test_benchmark,
    ]
    p = f = 0
    for t in tests:
        try:
            t()
            p += 1
        except Exception:
            f += 1
    print(f"\nEC090 F1a -- {p} passed, {f} failed")
    sys.exit(0 if f == 0 else 1)
