"""EC106 -- SOFC CHP -- F1a Eta Curve -- Test Suite"""

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
    assert_true(info["ec_id"] == "EC106", "ec_id == EC106")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")


def test_output_keys():
    print("Test: output keys")
    m = ComponentModel()
    r = m.predict({"load_fraction": 1.0})
    for k in ["eta_e", "eta_th", "P_e_W", "P_e_kW", "Q_th_W", "Q_in_W", "PER", "load_fraction"]:
        assert_true(k in r, f"key '{k}' in output")


def test_full_load_eta_e():
    print("Test: eta_e at full load = 0.55 * (0.3+0.7*1.0) = 0.55")
    m = ComponentModel()
    r = m.predict({"load_fraction": 1.0})
    expected = 0.55 * (0.3 + 0.7 * 1.0)
    assert_true(abs(r["eta_e"] - expected) < 1e-6,
                f"eta_e at load=1.0 = {expected:.6f} (got {r['eta_e']:.6f})")


def test_full_load_eta_th():
    print("Test: eta_th at full load = 0.30 * (0.2+0.8*1.0) = 0.30")
    m = ComponentModel()
    r = m.predict({"load_fraction": 1.0})
    expected = 0.30 * (0.2 + 0.8 * 1.0)
    assert_true(abs(r["eta_th"] - expected) < 1e-6,
                f"eta_th at load=1.0 = {expected:.6f} (got {r['eta_th']:.6f})")


def test_full_load_power():
    print("Test: P_e = 5 kW at full load")
    m = ComponentModel()
    r = m.predict({"load_fraction": 1.0})
    assert_true(abs(r["P_e_kW"] - 5.0) < 1e-6,
                f"P_e = 5kW (got {r['P_e_kW']:.6f})")


def test_zero_load():
    print("Test: P_e = 0, Q_th = 0 at zero load")
    m = ComponentModel()
    r = m.predict({"load_fraction": 0.0})
    assert_true(r["P_e_W"] == 0.0, "P_e = 0 at zero load")
    assert_true(r["Q_th_W"] == 0.0, "Q_th = 0 at zero load")


def test_eta_e_increases_with_load():
    print("Test: eta_e increases with load")
    m = ComponentModel()
    r_lo = m.predict({"load_fraction": 0.3})
    r_hi = m.predict({"load_fraction": 0.9})
    assert_true(r_hi["eta_e"] > r_lo["eta_e"],
                "eta_e increases with load")


def test_per_above_eta_e():
    print("Test: PER > eta_e (thermal output adds to total useful energy)")
    m = ComponentModel()
    r = m.predict({"load_fraction": 1.0})
    assert_true(r["PER"] > r["eta_e"],
                f"PER ({r['PER']:.4f}) > eta_e ({r['eta_e']:.4f})")


def test_p_e_kw_consistent():
    print("Test: P_e_kW = P_e_W / 1000")
    m = ComponentModel()
    r = m.predict({"load_fraction": 0.7})
    assert_true(abs(r["P_e_kW"] - r["P_e_W"] / 1000.0) < 1e-9,
                "P_e_kW = P_e_W / 1000")


def test_half_load_power():
    print("Test: P_e = 2.5 kW at 50% load")
    m = ComponentModel()
    r = m.predict({"load_fraction": 0.5})
    assert_true(abs(r["P_e_kW"] - 2.5) < 1e-6,
                f"P_e = 2.5kW at 50% load (got {r['P_e_kW']:.6f})")


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
        test_full_load_eta_e,
        test_full_load_eta_th,
        test_full_load_power,
        test_zero_load,
        test_eta_e_increases_with_load,
        test_per_above_eta_e,
        test_p_e_kw_consistent,
        test_half_load_power,
        test_benchmark,
    ]
    p = f = 0
    for t in tests:
        try:
            t()
            p += 1
        except Exception:
            f += 1
    print(f"\nEC106 F1a -- {p} passed, {f} failed")
    sys.exit(0 if f == 0 else 1)
