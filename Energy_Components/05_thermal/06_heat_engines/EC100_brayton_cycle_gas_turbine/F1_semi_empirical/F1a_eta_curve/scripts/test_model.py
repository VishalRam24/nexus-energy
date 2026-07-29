"""EC100 -- Brayton Cycle Gas Turbine -- F1a Eta Curve -- Test Suite"""

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
    assert_true(info["ec_id"] == "EC100", "ec_id == EC100")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")


def test_output_keys():
    print("Test: output keys")
    m = ComponentModel()
    r = m.predict({"load_fraction": 0.8})
    for k in ["eta", "P_out_W", "P_out_MW", "Q_in_W", "Q_exhaust_W", "load_fraction", "f_load"]:
        assert_true(k in r, f"key '{k}' in output")


def test_full_load_efficiency():
    print("Test: eta at full load = eta_rated * (0.2 + 0.8*1.0) = 0.38")
    m = ComponentModel()
    r = m.predict({"load_fraction": 1.0})
    expected_eta = 0.38 * (0.2 + 0.8 * 1.0)
    assert_true(abs(r["eta"] - expected_eta) < 1e-6,
                f"eta at load=1.0 = {expected_eta:.6f} (got {r['eta']:.6f})")


def test_full_load_power():
    print("Test: P_out = 50 MW at full load")
    m = ComponentModel()
    r = m.predict({"load_fraction": 1.0})
    assert_true(abs(r["P_out_MW"] - 50.0) < 1e-6,
                f"P_out = 50MW (got {r['P_out_MW']:.6f})")


def test_zero_load_zero_power():
    print("Test: P_out = 0 at zero load")
    m = ComponentModel()
    r = m.predict({"load_fraction": 0.0})
    assert_true(r["P_out_W"] == 0.0, f"P_out=0 at load=0 (got {r['P_out_W']:.2f})")


def test_eta_increases_with_load():
    print("Test: eta increases with load fraction")
    m = ComponentModel()
    r_lo = m.predict({"load_fraction": 0.3})
    r_hi = m.predict({"load_fraction": 0.9})
    assert_true(r_hi["eta"] > r_lo["eta"],
                f"eta increases with load ({r_lo['eta']:.4f} < {r_hi['eta']:.4f})")


def test_energy_balance():
    print("Test: Q_in = P_out + Q_exhaust")
    m = ComponentModel()
    r = m.predict({"load_fraction": 0.8})
    assert_true(abs(r["Q_in_W"] - r["P_out_W"] - r["Q_exhaust_W"]) < 1.0,
                "Q_in = P_out + Q_exhaust")


def test_p_out_mw_consistent():
    print("Test: P_out_MW = P_out_W / 1e6")
    m = ComponentModel()
    r = m.predict({"load_fraction": 0.7})
    assert_true(abs(r["P_out_MW"] - r["P_out_W"] / 1e6) < 1e-9,
                "P_out_MW = P_out_W / 1e6")


def test_half_load():
    print("Test: P_out = 25 MW at 50% load")
    m = ComponentModel()
    r = m.predict({"load_fraction": 0.5})
    assert_true(abs(r["P_out_MW"] - 25.0) < 1e-6,
                f"P_out = 25MW at 50% load (got {r['P_out_MW']:.6f})")


def test_benchmark():
    print("Test: benchmark 1000 predictions < 1s")
    m = ComponentModel()
    start = time.perf_counter()
    for i in range(1000):
        load = (i % 100) / 100.0
        m.predict({"load_fraction": load})
    elapsed = time.perf_counter() - start
    print(f"    1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions complete in < 1s")


if __name__ == "__main__":
    tests = [
        test_instantiation,
        test_output_keys,
        test_full_load_efficiency,
        test_full_load_power,
        test_zero_load_zero_power,
        test_eta_increases_with_load,
        test_energy_balance,
        test_p_out_mw_consistent,
        test_half_load,
        test_benchmark,
    ]
    p = f = 0
    for t in tests:
        try:
            t()
            p += 1
        except Exception:
            f += 1
    print(f"\nEC100 F1a -- {p} passed, {f} failed")
    sys.exit(0 if f == 0 else 1)
