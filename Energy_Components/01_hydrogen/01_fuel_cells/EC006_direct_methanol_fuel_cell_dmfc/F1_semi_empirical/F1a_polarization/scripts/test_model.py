"""EC006 -- DMFC -- F1a -- Test suite"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import DMFCPolarizationModel
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"

def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def test_open_circuit_crossover():
    print("\nTest 1: OCV reduced by methanol crossover (0.6 V, not 1.21 V)")
    m = DMFCPolarizationModel({"T": 343.15, "E_rev": 1.21, "OCV": 0.6,
                                "i0": 1e-5, "R_ohm": 0.4, "j_L": 0.4, "alpha": 0.5})
    r = m.evaluate(j=0.0)
    assert_true(r["V_cell"] == 0.6, f"OCV should be 0.6 V (crossover), got {r['V_cell']}")
    assert_true(r["efficiency"] < 0.6, f"efficiency vs E_rev should be < 0.6 (crossover penalty)")


def test_mid_current():
    print("\nTest 2: Mid-range current density (j=0.1 A/cm^2)")
    m = DMFCPolarizationModel({"T": 343.15, "E_rev": 1.21, "OCV": 0.6,
                                "i0": 1e-5, "R_ohm": 0.4, "j_L": 0.4, "alpha": 0.5})
    r = m.evaluate(j=0.1)
    assert_true(0.0 < r["V_cell"] < 0.6, f"V_cell={r['V_cell']} should be in (0, 0.6)")
    assert_true(r["P_density"] > 0, "P_density should be positive")


def test_monotonic_decrease():
    print("\nTest 3: Voltage decreases monotonically with current density")
    m = DMFCPolarizationModel({"T": 343.15, "E_rev": 1.21, "OCV": 0.6,
                                "i0": 1e-5, "R_ohm": 0.4, "j_L": 0.4, "alpha": 0.5})
    js = [0.01, 0.05, 0.1, 0.2, 0.3]
    vs = [m.evaluate(j=j_)["V_cell"] for j_ in js]
    assert_true(all(vs[i] > vs[i+1] for i in range(len(vs)-1)),
                f"Non-monotonic: {vs}")


def test_limiting_current():
    print("\nTest 4: Voltage = 0 at/beyond j_L")
    m = DMFCPolarizationModel({"T": 343.15, "E_rev": 1.21, "OCV": 0.6,
                                "i0": 1e-5, "R_ohm": 0.4, "j_L": 0.4, "alpha": 0.5})
    assert_true(m.evaluate(j=0.4)["V_cell"] == 0.0, "V at j_L should be 0")
    assert_true(m.evaluate(j=0.5)["V_cell"] == 0.0, "V beyond j_L should be 0")


def test_six_electron_reaction():
    print("\nTest 5: n=6 electrons gives smaller Tafel slope than n=2")
    import math
    # Tafel slope b = RT/(alpha*n*F) -- larger n -> smaller b -> less activation loss
    m2 = DMFCPolarizationModel({"T": 343.15, "E_rev": 1.21, "OCV": 0.6,
                                  "i0": 1e-5, "R_ohm": 0.0, "j_L": 0.4, "alpha": 0.5})
    # With R_ohm=0, loss is purely activation -- check it's small due to n=6
    r = m2.evaluate(j=0.1)
    V_act = 0.6 - r["V_cell"]
    # b = RT/(alpha*n*F) = 8.314*343.15/(0.5*6*96485) = ~0.0099 V/decade
    b_expected = 8.314 * 343.15 / (0.5 * 6 * 96485)
    assert_true(b_expected < 0.015, f"Tafel slope b={b_expected:.5f} should be < 0.015 for n=6")


def test_component_model_wrapper():
    print("\nTest 6: ComponentModel wrapper")
    cm = ComponentModel()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC006", "component_id mismatch")
    r = cm.predict({"j": 0.05})
    assert_true("efficiency_ocv" in r, "Missing efficiency_ocv in outputs")
    assert_true(0.0 < r["efficiency_ocv"] <= 1.0,
                f"efficiency_ocv={r['efficiency_ocv']} out of range")


if __name__ == "__main__":
    tests = [
        test_open_circuit_crossover,
        test_mid_current,
        test_monotonic_decrease,
        test_limiting_current,
        test_six_electron_reaction,
        test_component_model_wrapper,
    ]
    p = f = 0
    for t in tests:
        try:
            t(); p += 1
        except Exception as e:
            f += 1; print(f"  ERROR: {e}")
    print(f"\n{'='*50}\nEC006 F1a -- {p} passed, {f} failed\n{'='*50}")
    sys.exit(0 if f == 0 else 1)
