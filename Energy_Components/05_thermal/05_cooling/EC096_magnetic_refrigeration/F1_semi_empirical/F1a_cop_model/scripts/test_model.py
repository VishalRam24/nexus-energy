"""EC096 -- Magnetic Refrigeration -- F1a COP Model -- Test Suite"""

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
    assert_true(info["ec_id"] == "EC096", "ec_id == EC096")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")


def test_output_keys():
    print("Test: output keys")
    m = ComponentModel()
    r = m.predict({"W_input_W": 1000.0})
    for k in ["COP", "COP_Carnot", "Q_cool_W", "Q_hot_W", "T_span_K", "W_input_W"]:
        assert_true(k in r, f"key '{k}' in output")


def test_cop_formula():
    print("Test: COP = eta_2nd * T_cold / (T_hot - T_cold)")
    m = ComponentModel()
    T_cold = 273.15
    T_hot = 298.15
    eta = 0.4
    expected_COP = eta * T_cold / (T_hot - T_cold)
    r = m.predict({"W_input_W": 1000.0, "T_cold_K": T_cold, "T_hot_K": T_hot})
    assert_true(abs(r["COP"] - expected_COP) < 1e-6,
                f"COP = eta*T_cold/dT = {expected_COP:.6f} (got {r['COP']:.6f})")


def test_cop_less_than_carnot():
    print("Test: COP < COP_Carnot (eta_2nd < 1)")
    m = ComponentModel()
    r = m.predict({"W_input_W": 1000.0})
    assert_true(r["COP"] < r["COP_Carnot"],
                f"COP ({r['COP']:.4f}) < COP_Carnot ({r['COP_Carnot']:.4f})")


def test_q_cool_formula():
    print("Test: Q_cool = COP * W")
    m = ComponentModel()
    W = 1500.0
    r = m.predict({"W_input_W": W})
    assert_true(abs(r["Q_cool_W"] - r["COP"] * W) < 0.01,
                f"Q_cool = COP*W = {r['COP']*W:.2f}W (got {r['Q_cool_W']:.2f})")


def test_energy_balance():
    print("Test: Q_hot = Q_cool + W (energy balance)")
    m = ComponentModel()
    W = 1000.0
    r = m.predict({"W_input_W": W})
    assert_true(abs(r["Q_hot_W"] - r["Q_cool_W"] - W) < 0.01,
                f"Q_hot = Q_cool + W (diff={abs(r['Q_hot_W']-r['Q_cool_W']-W):.4f})")


def test_cop_increases_with_smaller_delta_T():
    print("Test: COP increases as T_hot-T_cold decreases")
    m = ComponentModel()
    r_small = m.predict({"W_input_W": 1000.0, "T_cold_K": 273.15, "T_hot_K": 283.15})
    r_large = m.predict({"W_input_W": 1000.0, "T_cold_K": 273.15, "T_hot_K": 313.15})
    assert_true(r_small["COP"] > r_large["COP"],
                "COP higher for smaller temperature lift")


def test_t_span():
    print("Test: T_span = N_stages * delta_T_MCE = 6 * 3 = 18 K")
    m = ComponentModel()
    r = m.predict({"W_input_W": 1000.0})
    assert_true(abs(r["T_span_K"] - 18.0) < 1e-6,
                f"T_span = 18K (got {r['T_span_K']:.2f})")


def test_zero_work_zero_cooling():
    print("Test: W=0 -> Q_cool=0")
    m = ComponentModel()
    r = m.predict({"W_input_W": 0.0})
    assert_true(r["Q_cool_W"] == 0.0, "Q_cool = 0 when W = 0")


def test_benchmark():
    print("Test: benchmark 1000 predictions < 1s")
    m = ComponentModel()
    start = time.perf_counter()
    for i in range(1000):
        W = 100.0 + i * 10.0
        T_cold = 260.0 + (i % 30)
        T_hot = T_cold + 10 + (i % 40)
        m.predict({"W_input_W": W, "T_cold_K": T_cold, "T_hot_K": T_hot})
    elapsed = time.perf_counter() - start
    print(f"    1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions complete in < 1s")


if __name__ == "__main__":
    tests = [
        test_instantiation,
        test_output_keys,
        test_cop_formula,
        test_cop_less_than_carnot,
        test_q_cool_formula,
        test_energy_balance,
        test_cop_increases_with_smaller_delta_T,
        test_t_span,
        test_zero_work_zero_cooling,
        test_benchmark,
    ]
    p = f = 0
    for t in tests:
        try:
            t()
            p += 1
        except Exception:
            f += 1
    print(f"\nEC096 F1a -- {p} passed, {f} failed")
    sys.exit(0 if f == 0 else 1)
