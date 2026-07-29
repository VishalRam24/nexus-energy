"""EC102 -- Kalina Cycle -- F1a Eta Curve -- Test Suite"""

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
    assert_true(info["ec_id"] == "EC102", "ec_id == EC102")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")


def test_output_keys():
    print("Test: output keys")
    m = ComponentModel()
    r = m.predict({"T_source_K": 423.15})
    for k in ["eta", "eta_Carnot", "P_out_W", "P_out_kW", "Q_source_W", "Q_rejected_W", "T_source_K"]:
        assert_true(k in r, f"key '{k}' in output")


def test_carnot_formula():
    print("Test: eta_Carnot = 1 - T_sink/T_source")
    m = ComponentModel()
    T_source = 423.15
    T_sink = m._model.T_sink
    expected = 1.0 - T_sink / T_source
    r = m.predict({"T_source_K": T_source})
    assert_true(abs(r["eta_Carnot"] - expected) < 1e-6,
                f"eta_Carnot = {expected:.6f} (got {r['eta_Carnot']:.6f})")


def test_eta_formula():
    print("Test: eta = eta_2nd * eta_Carnot")
    m = ComponentModel()
    r = m.predict({"T_source_K": 423.15})
    expected = 0.55 * r["eta_Carnot"]
    assert_true(abs(r["eta"] - expected) < 1e-6,
                f"eta = 0.55*eta_Carnot = {expected:.6f} (got {r['eta']:.6f})")


def test_eta_less_than_carnot():
    print("Test: eta < eta_Carnot")
    m = ComponentModel()
    r = m.predict({"T_source_K": 423.15})
    assert_true(r["eta"] < r["eta_Carnot"],
                f"eta ({r['eta']:.4f}) < eta_Carnot ({r['eta_Carnot']:.4f})")


def test_energy_balance():
    print("Test: Q_source = P_out + Q_rejected")
    m = ComponentModel()
    Q = 2e6
    r = m.predict({"T_source_K": 423.15, "Q_source_W": Q})
    assert_true(abs(r["Q_source_W"] + 0 - r["P_out_W"] - r["Q_rejected_W"]) < 1.0,
                "Q_source = P_out + Q_rejected")


def test_eta_increases_with_source_temp():
    print("Test: eta increases with T_source")
    m = ComponentModel()
    r_lo = m.predict({"T_source_K": 373.15})   # 100 degC
    r_hi = m.predict({"T_source_K": 473.15})   # 200 degC
    assert_true(r_hi["eta"] > r_lo["eta"],
                f"eta increases with T_source ({r_lo['eta']:.4f} < {r_hi['eta']:.4f})")


def test_p_out_kw_consistent():
    print("Test: P_out_kW = P_out_W / 1000")
    m = ComponentModel()
    r = m.predict({"T_source_K": 423.15})
    assert_true(abs(r["P_out_kW"] - r["P_out_W"] / 1000.0) < 1e-6,
                "P_out_kW = P_out_W / 1000")


def test_q_source_input():
    print("Test: P_out = Q_source * eta when Q_source provided")
    m = ComponentModel()
    Q = 1e6
    r = m.predict({"T_source_K": 423.15, "Q_source_W": Q})
    expected_P = Q * r["eta"]
    assert_true(abs(r["P_out_W"] - expected_P) < 1.0,
                f"P_out = Q*eta = {expected_P:.0f}W (got {r['P_out_W']:.0f})")


def test_benchmark():
    print("Test: benchmark 1000 predictions < 1s")
    m = ComponentModel()
    start = time.perf_counter()
    for i in range(1000):
        T = 373.15 + (i % 100)
        m.predict({"T_source_K": float(T)})
    elapsed = time.perf_counter() - start
    print(f"    1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions complete in < 1s")


if __name__ == "__main__":
    tests = [
        test_instantiation,
        test_output_keys,
        test_carnot_formula,
        test_eta_formula,
        test_eta_less_than_carnot,
        test_energy_balance,
        test_eta_increases_with_source_temp,
        test_p_out_kw_consistent,
        test_q_source_input,
        test_benchmark,
    ]
    p = f = 0
    for t in tests:
        try:
            t()
            p += 1
        except Exception:
            f += 1
    print(f"\nEC102 F1a -- {p} passed, {f} failed")
    sys.exit(0 if f == 0 else 1)
