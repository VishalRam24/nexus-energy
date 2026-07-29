"""EC007 -- RFC -- F1a -- Test suite"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import RFCPolarizationModel
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"

def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def test_ocv_mode():
    print("\nTest 1: OCV at j=0")
    m = RFCPolarizationModel({"T": 353.15, "E_rev": 1.23, "i0": 1e-3,
                               "R_ohm": 0.3, "j_L_fc": 1.0, "j_L_el": 2.0, "alpha": 0.5})
    r = m.evaluate(j=0.0)
    assert_true(r["V_cell"] == 1.23, f"OCV should be E_rev=1.23, got {r['V_cell']}")
    assert_true(r["mode"] == "OCV", f"mode should be OCV, got {r['mode']}")


def test_fc_mode():
    print("\nTest 2: FC mode (j > 0), voltage decreases")
    m = RFCPolarizationModel({"T": 353.15, "E_rev": 1.23, "i0": 1e-3,
                               "R_ohm": 0.3, "j_L_fc": 1.0, "j_L_el": 2.0, "alpha": 0.5})
    r = m.evaluate(j=0.3)
    assert_true(r["mode"] == "FC", f"mode should be FC, got {r['mode']}")
    assert_true(r["V_cell"] < 1.23, f"V_cell={r['V_cell']} should be < E_rev in FC mode")
    assert_true(r["P_density"] > 0, "Power density should be positive in FC mode")


def test_el_mode():
    print("\nTest 3: EL mode (j < 0), voltage increases above E_rev")
    m = RFCPolarizationModel({"T": 353.15, "E_rev": 1.23, "i0": 1e-3,
                               "R_ohm": 0.3, "j_L_fc": 1.0, "j_L_el": 2.0, "alpha": 0.5})
    r = m.evaluate(j=-0.3)
    assert_true(r["mode"] == "EL", f"mode should be EL, got {r['mode']}")
    assert_true(r["V_cell"] > 1.23, f"V_cell={r['V_cell']} should be > E_rev in EL mode")
    assert_true(r["P_density"] < 0, "Power density should be negative (consumed) in EL mode")


def test_fc_voltage_decreases():
    print("\nTest 4: FC-mode voltages decrease monotonically with j")
    m = RFCPolarizationModel({"T": 353.15, "E_rev": 1.23, "i0": 1e-3,
                               "R_ohm": 0.3, "j_L_fc": 1.0, "j_L_el": 2.0, "alpha": 0.5})
    js = [0.05, 0.1, 0.2, 0.4, 0.7]
    vs = [m.evaluate(j=j_)["V_cell"] for j_ in js]
    assert_true(all(vs[i] > vs[i+1] for i in range(len(vs)-1)),
                f"FC voltages not decreasing: {vs}")


def test_el_voltage_increases():
    print("\nTest 5: EL-mode voltages increase (more negative j -> higher V)")
    m = RFCPolarizationModel({"T": 353.15, "E_rev": 1.23, "i0": 1e-3,
                               "R_ohm": 0.3, "j_L_fc": 1.0, "j_L_el": 2.0, "alpha": 0.5})
    js = [-0.05, -0.1, -0.3, -0.6, -1.0]
    vs = [m.evaluate(j=j_)["V_cell"] for j_ in js]
    assert_true(all(vs[i] < vs[i+1] for i in range(len(vs)-1)),
                f"EL voltages not increasing with |j|: {vs}")


def test_component_model_wrapper():
    print("\nTest 6: ComponentModel wrapper")
    cm = ComponentModel()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC007", "component_id mismatch")
    r_fc = cm.predict({"j": 0.2})
    r_el = cm.predict({"j": -0.2})
    assert_true(r_fc["mode"] == "FC", "FC mode not detected")
    assert_true(r_el["mode"] == "EL", "EL mode not detected")
    assert_true(r_fc["V_cell"] < r_el["V_cell"],
                f"FC V={r_fc['V_cell']} should be < EL V={r_el['V_cell']}")


if __name__ == "__main__":
    tests = [
        test_ocv_mode,
        test_fc_mode,
        test_el_mode,
        test_fc_voltage_decreases,
        test_el_voltage_increases,
        test_component_model_wrapper,
    ]
    p = f = 0
    for t in tests:
        try:
            t(); p += 1
        except Exception as e:
            f += 1; print(f"  ERROR: {e}")
    print(f"\n{'='*50}\nEC007 F1a -- {p} passed, {f} failed\n{'='*50}")
    sys.exit(0 if f == 0 else 1)
