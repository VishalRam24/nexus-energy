"""EC088 -- Oil-Fired Boiler -- F1a Constant Efficiency -- Test Suite"""

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
    assert_true(info["ec_id"] == "EC088", "ec_id == EC088")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")


def test_output_keys():
    print("Test: output keys")
    m = ComponentModel()
    r = m.predict({"Q_demand_W": 200000.0})
    for k in ["Q_out_W", "Q_out_kW", "m_fuel_kg_s", "Q_in_W", "losses_W", "load_fraction", "eta"]:
        assert_true(k in r, f"key '{k}' in output")


def test_efficiency_formula_fuel_input():
    print("Test: Q_out = eta * m_fuel * LHV")
    m = ComponentModel()
    eta = 0.87
    LHV = 42.6e6
    m_fuel = 0.01
    r = m.predict({"m_fuel_kg_s": m_fuel})
    expected_Q = eta * m_fuel * LHV
    assert_true(abs(r["Q_out_W"] - expected_Q) < 1.0,
                f"Q_out = eta*m_fuel*LHV = {expected_Q:.0f}W (got {r['Q_out_W']:.0f})")


def test_inverse_fuel_demand():
    print("Test: m_fuel back-calculated from Q_demand")
    m = ComponentModel()
    Q_demand = 200000.0
    r = m.predict({"Q_demand_W": Q_demand})
    assert_true(abs(r["Q_out_W"] - Q_demand) < 1.0,
                f"Q_out matches Q_demand={Q_demand:.0f}W (got {r['Q_out_W']:.0f})")


def test_eta_constant():
    print("Test: eta constant at 0.87")
    m = ComponentModel()
    r = m.predict({"Q_demand_W": 300000.0})
    assert_true(abs(r["eta"] - 0.87) < 1e-6, f"eta = 0.87 (got {r['eta']:.6f})")


def test_losses_positive():
    print("Test: losses > 0 (Q_in > Q_out)")
    m = ComponentModel()
    r = m.predict({"Q_demand_W": 200000.0})
    assert_true(r["losses_W"] > 0, f"losses > 0 (got {r['losses_W']:.2f}W)")
    assert_true(r["Q_in_W"] > r["Q_out_W"], "Q_in > Q_out")


def test_energy_balance():
    print("Test: Q_out + losses = Q_in")
    m = ComponentModel()
    r = m.predict({"Q_demand_W": 200000.0})
    assert_true(abs(r["Q_out_W"] + r["losses_W"] - r["Q_in_W"]) < 1.0,
                "Q_out + losses = Q_in")


def test_load_fraction_half():
    print("Test: load fraction = 0.5 at 250 kW (half of 500 kW rated)")
    m = ComponentModel()
    r = m.predict({"Q_demand_W": 250000.0})
    assert_true(abs(r["load_fraction"] - 0.5) < 1e-5,
                f"load = 0.5 at 250 kW (got {r['load_fraction']:.6f})")


def test_q_out_kw_consistent():
    print("Test: Q_out_kW = Q_out_W / 1000")
    m = ComponentModel()
    r = m.predict({"Q_demand_W": 300000.0})
    assert_true(abs(r["Q_out_kW"] - r["Q_out_W"] / 1000.0) < 1e-3,
                "Q_out_kW = Q_out_W / 1000")


def test_benchmark():
    print("Test: benchmark 1000 predictions < 1s")
    m = ComponentModel()
    start = time.perf_counter()
    for i in range(1000):
        Q = 10000.0 + i * 490.0
        m.predict({"Q_demand_W": Q})
    elapsed = time.perf_counter() - start
    print(f"    1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions complete in < 1s")


if __name__ == "__main__":
    tests = [
        test_instantiation,
        test_output_keys,
        test_efficiency_formula_fuel_input,
        test_inverse_fuel_demand,
        test_eta_constant,
        test_losses_positive,
        test_energy_balance,
        test_load_fraction_half,
        test_q_out_kw_consistent,
        test_benchmark,
    ]
    p = f = 0
    for t in tests:
        try:
            t()
            p += 1
        except Exception:
            f += 1
    print(f"\nEC088 F1a -- {p} passed, {f} failed")
    sys.exit(0 if f == 0 else 1)
