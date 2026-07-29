"""EC107 -- Stirling Micro-CHP -- F1a Eta Curve -- Test Suite"""

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
    assert_true(info["ec_id"] == "EC107", "ec_id == EC107")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")


def test_output_keys():
    print("Test: output keys")
    m = ComponentModel()
    r = m.predict({"load_fraction": 1.0})
    for k in ["eta_e", "eta_th", "P_e_W", "P_e_kW", "Q_th_W", "Q_in_W", "PER", "load_fraction"]:
        assert_true(k in r, f"key '{k}' in output")


def test_full_load_eta_e():
    print("Test: eta_e at full load = 0.15 * (0.2+0.8*1.0) = 0.15")
    m = ComponentModel()
    r = m.predict({"load_fraction": 1.0})
    expected = 0.15 * (0.2 + 0.8 * 1.0)
    assert_true(abs(r["eta_e"] - expected) < 1e-6,
                f"eta_e at load=1.0 = {expected:.6f} (got {r['eta_e']:.6f})")


def test_full_load_eta_th():
    print("Test: eta_th at full load = 0.70 * (0.2+0.8*1.0) = 0.70")
    m = ComponentModel()
    r = m.predict({"load_fraction": 1.0})
    expected = 0.70 * (0.2 + 0.8 * 1.0)
    assert_true(abs(r["eta_th"] - expected) < 1e-6,
                f"eta_th at load=1.0 = {expected:.6f} (got {r['eta_th']:.6f})")


def test_full_load_power():
    print("Test: P_e = 1 kW at full load")
    m = ComponentModel()
    r = m.predict({"load_fraction": 1.0})
    assert_true(abs(r["P_e_kW"] - 1.0) < 1e-6,
                f"P_e = 1.0 kW (got {r['P_e_kW']:.6f})")


def test_heat_led():
    print("Test: eta_th >> eta_e (heat-led Stirling)")
    m = ComponentModel()
    r = m.predict({"load_fraction": 1.0})
    assert_true(r["eta_th"] > r["eta_e"] * 3,
                f"eta_th ({r['eta_th']:.4f}) >> eta_e ({r['eta_e']:.4f})")


def test_zero_load():
    print("Test: P_e = Q_th = 0 at zero load")
    m = ComponentModel()
    r = m.predict({"load_fraction": 0.0})
    assert_true(r["P_e_W"] == 0.0, "P_e = 0 at zero load")
    assert_true(r["Q_th_W"] == 0.0, "Q_th = 0 at zero load")


def test_per_above_eta_e():
    print("Test: PER > eta_e (thermal output increases combined efficiency)")
    m = ComponentModel()
    r = m.predict({"load_fraction": 1.0})
    assert_true(r["PER"] > r["eta_e"],
                f"PER ({r['PER']:.4f}) > eta_e ({r['eta_e']:.4f})")


def test_same_f_factor_for_both():
    print("Test: eta_th/eta_e ratio constant with load (same f factor)")
    m = ComponentModel()
    r1 = m.predict({"load_fraction": 0.5})
    r2 = m.predict({"load_fraction": 0.9})
    ratio1 = r1["eta_th"] / r1["eta_e"] if r1["eta_e"] > 0 else 0
    ratio2 = r2["eta_th"] / r2["eta_e"] if r2["eta_e"] > 0 else 0
    assert_true(abs(ratio1 - ratio2) < 1e-9,
                f"eta_th/eta_e constant with load ({ratio1:.6f} vs {ratio2:.6f})")


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
        test_heat_led,
        test_zero_load,
        test_per_above_eta_e,
        test_same_f_factor_for_both,
        test_benchmark,
    ]
    p = f = 0
    for t in tests:
        try:
            t()
            p += 1
        except Exception:
            f += 1
    print(f"\nEC107 F1a -- {p} passed, {f} failed")
    sys.exit(0 if f == 0 else 1)
