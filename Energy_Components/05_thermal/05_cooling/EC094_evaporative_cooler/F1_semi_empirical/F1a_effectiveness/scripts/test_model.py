"""EC094 -- Evaporative Cooler -- F1a Effectiveness -- Test Suite"""

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
    assert_true(info["ec_id"] == "EC094", "ec_id == EC094")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")


def test_output_keys():
    print("Test: output keys")
    m = ComponentModel()
    r = m.predict({"T_db": 35.0, "T_wb": 20.0})
    for k in ["T_out", "Q_cool_W", "COP", "delta_T", "P_fan_W"]:
        assert_true(k in r, f"key '{k}' in output")


def test_temperature_formula():
    print("Test: T_out = T_db - eps*(T_db - T_wb)")
    m = ComponentModel()
    T_db, T_wb = 35.0, 20.0
    eps = 0.85
    expected = T_db - eps * (T_db - T_wb)
    r = m.predict({"T_db": T_db, "T_wb": T_wb})
    assert_true(abs(r["T_out"] - expected) < 1e-6,
                f"T_out = {expected:.4f}degC (got {r['T_out']:.4f})")


def test_t_out_between_db_and_wb():
    print("Test: T_wb <= T_out <= T_db")
    m = ComponentModel()
    r = m.predict({"T_db": 35.0, "T_wb": 20.0})
    assert_true(20.0 <= r["T_out"] <= 35.0,
                f"T_out in [T_wb, T_db] (got {r['T_out']:.4f})")


def test_zero_wet_bulb_depression_no_cooling():
    print("Test: T_db = T_wb -> no cooling effect (100% RH)")
    m = ComponentModel()
    r = m.predict({"T_db": 30.0, "T_wb": 30.0})
    assert_true(abs(r["T_out"] - 30.0) < 1e-6,
                f"T_out = T_db = 30 when T_wb = T_db (got {r['T_out']:.4f})")
    assert_true(abs(r["Q_cool_W"]) < 1e-6, "Q_cool = 0 at 100% RH")


def test_q_cool_formula():
    print("Test: Q_cool = m_dot * Cp * delta_T")
    m = ComponentModel()
    m_dot = 2.0
    Cp = 1005.0
    r = m.predict({"T_db": 35.0, "T_wb": 20.0, "m_dot_air": m_dot})
    expected = m_dot * Cp * r["delta_T"]
    assert_true(abs(r["Q_cool_W"] - expected) < 0.01,
                f"Q_cool = m*Cp*dT = {expected:.2f}W (got {r['Q_cool_W']:.2f})")


def test_q_cool_increases_with_wb_depression():
    print("Test: Q_cool increases with wet-bulb depression")
    m = ComponentModel()
    r_small = m.predict({"T_db": 35.0, "T_wb": 30.0})
    r_large = m.predict({"T_db": 35.0, "T_wb": 15.0})
    assert_true(r_large["Q_cool_W"] > r_small["Q_cool_W"],
                "Q_cool increases with T_db - T_wb")


def test_cop_positive():
    print("Test: COP > 0")
    m = ComponentModel()
    r = m.predict({"T_db": 35.0, "T_wb": 20.0, "m_dot_air": 2.0})
    assert_true(r["COP"] > 0, f"COP > 0 (got {r['COP']:.4f})")


def test_fan_power_constant():
    print("Test: P_fan = 200 W (constant)")
    m = ComponentModel()
    r = m.predict({"T_db": 35.0, "T_wb": 20.0})
    assert_true(abs(r["P_fan_W"] - 200.0) < 1e-6,
                f"P_fan = 200W (got {r['P_fan_W']:.4f})")


def test_benchmark():
    print("Test: benchmark 1000 predictions < 1s")
    m = ComponentModel()
    start = time.perf_counter()
    for i in range(1000):
        T_db = 25.0 + (i % 20)
        T_wb = 15.0 + (i % 10)
        m.predict({"T_db": float(T_db), "T_wb": float(T_wb), "m_dot_air": 1.0 + i % 5})
    elapsed = time.perf_counter() - start
    print(f"    1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions complete in < 1s")


if __name__ == "__main__":
    tests = [
        test_instantiation,
        test_output_keys,
        test_temperature_formula,
        test_t_out_between_db_and_wb,
        test_zero_wet_bulb_depression_no_cooling,
        test_q_cool_formula,
        test_q_cool_increases_with_wb_depression,
        test_cop_positive,
        test_fan_power_constant,
        test_benchmark,
    ]
    p = f = 0
    for t in tests:
        try:
            t()
            p += 1
        except Exception:
            f += 1
    print(f"\nEC094 F1a -- {p} passed, {f} failed")
    sys.exit(0 if f == 0 else 1)
