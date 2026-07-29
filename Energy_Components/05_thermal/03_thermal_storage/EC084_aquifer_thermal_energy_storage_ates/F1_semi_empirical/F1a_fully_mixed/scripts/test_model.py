"""EC084 -- ATES -- F1a Fully Mixed -- Test Suite"""

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
    assert_true(info["ec_id"] == "EC084", "ec_id == EC084")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")


def test_output_keys():
    print("Test: output keys")
    m = ComponentModel()
    r = m.predict({"T_storage": 20.0})
    for k in ["E_stored_J", "E_stored_kWh", "Q_thermal_W", "delta_T", "SOC", "T_storage"]:
        assert_true(k in r, f"key '{k}' in output")


def test_energy_formula():
    print("Test: E_stored = V*rho*Cp*dT*eta")
    m = ComponentModel()
    V = 50000.0
    rho = 1000.0
    Cp = 4186.0
    eta = 0.70
    T_ground = m._model.T_ground
    T_storage = 20.0
    dT = T_storage - T_ground
    expected_J = V * rho * Cp * dT * eta
    r = m.predict({"T_storage": T_storage})
    assert_true(abs(r["E_stored_J"] - expected_J) < 1.0,
                f"E_stored_J = {expected_J:.0f}J (got {r['E_stored_J']:.0f})")


def test_energy_zero_at_ground_temp():
    print("Test: zero energy stored at ground temperature")
    m = ComponentModel()
    T_ground = m._model.T_ground
    r = m.predict({"T_storage": T_ground})
    assert_true(abs(r["E_stored_J"]) < 1.0,
                f"E_stored_J = 0 at T_ground (got {r['E_stored_J']:.0f})")


def test_energy_increases_with_temperature():
    print("Test: stored energy increases with temperature")
    m = ComponentModel()
    r_lo = m.predict({"T_storage": 15.0})
    r_hi = m.predict({"T_storage": 25.0})
    assert_true(r_hi["E_stored_J"] > r_lo["E_stored_J"],
                "E_stored increases with T_storage")


def test_soc_bounded():
    print("Test: SOC in [0, 1]")
    m = ComponentModel()
    for T in [5.0, 12.0, 20.0, 25.0, 30.0]:
        r = m.predict({"T_storage": T})
        assert_true(0.0 <= r["SOC"] <= 1.0, f"SOC in [0,1] at T={T}degC (got {r['SOC']:.4f})")


def test_energy_in_kwh():
    print("Test: E_stored_kWh = E_stored_J / 3600000")
    m = ComponentModel()
    r = m.predict({"T_storage": 20.0})
    assert_true(abs(r["E_stored_kWh"] - r["E_stored_J"] / 3.6e6) < 0.01,
                "E_kWh = E_J / 3.6e6")


def test_thermal_power_with_flow():
    print("Test: Q_thermal_W increases with flow rate")
    m = ComponentModel()
    r1 = m.predict({"T_storage": 20.0, "m_dot": 1.0, "T_in": 20.0})
    r5 = m.predict({"T_storage": 20.0, "m_dot": 5.0, "T_in": 20.0})
    assert_true(r5["Q_thermal_W"] > r1["Q_thermal_W"],
                "Q_thermal increases with m_dot")


def test_benchmark():
    print("Test: benchmark 1000 predictions < 1s")
    m = ComponentModel()
    start = time.perf_counter()
    for i in range(1000):
        T = 12.0 + (i % 18)
        m.predict({"T_storage": float(T), "m_dot": float(i % 10)})
    elapsed = time.perf_counter() - start
    print(f"    1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions complete in < 1s")


if __name__ == "__main__":
    tests = [
        test_instantiation,
        test_output_keys,
        test_energy_formula,
        test_energy_zero_at_ground_temp,
        test_energy_increases_with_temperature,
        test_soc_bounded,
        test_energy_in_kwh,
        test_thermal_power_with_flow,
        test_benchmark,
    ]
    p = f = 0
    for t in tests:
        try:
            t()
            p += 1
        except Exception:
            f += 1
    print(f"\nEC084 F1a -- {p} passed, {f} failed")
    sys.exit(0 if f == 0 else 1)
