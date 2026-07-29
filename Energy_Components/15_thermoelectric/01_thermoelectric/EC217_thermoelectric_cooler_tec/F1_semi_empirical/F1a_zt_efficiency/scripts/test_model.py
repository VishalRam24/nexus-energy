"""EC217 — Thermoelectric Cooler (TEC) — F1a — Test Suite"""
import sys, os, numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

PASS = "\u2713"
FAIL = "\u2717"


def assert_true(condition, msg):
    if condition:
        print(f"  {PASS} {msg}")
    else:
        print(f"  {FAIL} {msg}")
        raise AssertionError(msg)


def run_tests():
    print("EC217 Thermoelectric Cooler (TEC) F1a — Tests")
    print("=" * 50)
    model = ComponentModel()

    r = model.predict({"Tc_K": 280.0, "Th_K": 310.0})
    for key in ["COP_carnot", "COP_ZT", "COP_physical", "Q_cool_W",
                "W_input_W", "eta_zt", "ZT_eff", "I_optimal_A"]:
        assert_true(key in r, f"Output key '{key}' present")

    info = model.get_info()
    assert_true(info["ec_id"] == "EC217", "ec_id == EC217")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Carnot COP = Tc / (Th - Tc) = 280 / 30 = 9.333...
    r = model.predict({"Tc_K": 280.0, "Th_K": 310.0})
    expected_cop_carnot = 280.0 / 30.0
    assert_true(abs(float(r["COP_carnot"]) - expected_cop_carnot) < 1e-9,
                f"COP_Carnot = Tc/(Th-Tc) = {expected_cop_carnot:.4f}")

    # ZT-limited COP <= Carnot COP
    assert_true(float(r["COP_ZT"]) <= float(r["COP_carnot"]) + 1e-9,
                "COP_ZT <= COP_Carnot")

    # COP decreases as dT increases (harder to pump heat)
    r_small = model.predict({"Tc_K": 280.0, "Th_K": 285.0})
    r_large = model.predict({"Tc_K": 280.0, "Th_K": 320.0})
    assert_true(float(r_small["COP_ZT"]) > float(r_large["COP_ZT"]),
                "COP_ZT decreases as dT increases")

    # Carnot COP = infinity at dT=0 — avoid; check monotonicity
    assert_true(float(r_small["COP_carnot"]) > float(r_large["COP_carnot"]),
                "COP_Carnot decreases as dT increases")

    # eta_zt in (0, 1) for reasonable temperatures
    assert_true(0.0 <= float(r["eta_zt"]) <= 1.0, "eta_zt in [0, 1]")

    # ZT_eff > 0
    assert_true(float(r["ZT_eff"]) > 0.0, "ZT_eff > 0")

    # W_input > 0
    assert_true(float(r["W_input_W"]) > 0.0, "W_input > 0")

    # I_optimal > 0
    assert_true(float(r["I_optimal_A"]) > 0.0, "I_optimal > 0")

    # Q_cool at optimal current should be > 0 for small dT
    r_small2 = model.predict({"Tc_K": 280.0, "Th_K": 285.0})
    assert_true(float(r_small2["Q_cool_W"]) > 0.0, "Q_cool > 0 at small dT (Tc=280, Th=285)")

    # COP_physical = Q_cool / W_input
    r_test = model.predict({"Tc_K": 280.0, "Th_K": 300.0, "I_A": 2.0})
    Q = float(r_test["Q_cool_W"])
    W = float(r_test["W_input_W"])
    cop_check = float(r_test["COP_physical"])
    if W > 0 and Q > 0:
        assert_true(abs(cop_check - Q / W) < 1e-9, "COP_physical = Q_cool / W_input")

    # Vectorized
    Tc_arr = np.linspace(260.0, 300.0, 10)
    Th_arr = Tc_arr + 25.0
    r_vec = model.predict({"Tc_K": Tc_arr, "Th_K": Th_arr})
    assert_true(len(r_vec["COP_ZT"]) == 10, "Vectorized: 10 outputs")

    # COP_ZT > 0 for all vectorized cases
    assert_true(np.all(r_vec["COP_ZT"] >= 0.0), "All vectorized COP_ZT >= 0")

    print()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
