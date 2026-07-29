"""EC033 -- Iron-Air Battery -- F1a -- Test suite"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import IronAirBatteryModel
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"

def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def test_ocv_at_full():
    print("\nTest 1: OCV at full charge = ocv_flat = 1.28 V")
    m = IronAirBatteryModel({"Q_nom": 4.0, "R_int": 0.100, "V_min": 0.8, "V_max": 1.5,
                              "ocv_flat": 1.28, "ocv_droop": 0.15})
    r = m.evaluate(soc=1.0)
    assert_true(abs(r["OCV"] - 1.28) < 1e-6, f"OCV at SOC=1 should be 1.28, got {r['OCV']}")


def test_ocv_drops_to_low():
    print("\nTest 2: OCV drops at low SOC")
    m = IronAirBatteryModel({"Q_nom": 4.0, "R_int": 0.100, "V_min": 0.8, "V_max": 1.5,
                              "ocv_flat": 1.28, "ocv_droop": 0.15})
    v_empty = m.evaluate(soc=0.0)["OCV"]
    assert_true(abs(v_empty - (1.28 - 0.15)) < 1e-6,
                f"OCV at SOC=0 should be {1.28-0.15:.2f}, got {v_empty}")


def test_r_int_drop():
    print("\nTest 3: Voltage drop = I * R_int = 0.100 V at 1 A")
    m = IronAirBatteryModel({"Q_nom": 4.0, "R_int": 0.100, "V_min": 0.8, "V_max": 1.5,
                              "ocv_flat": 1.28, "ocv_droop": 0.15})
    r0 = m.evaluate(soc=0.5, I=0.0)
    r1 = m.evaluate(soc=0.5, I=1.0)
    assert_true(abs(r0["V_terminal"] - r1["V_terminal"] - 0.100) < 1e-6,
                f"Voltage drop should be 0.100 V")


def test_soc_update():
    print("\nTest 4: SOC update by Coulomb counting")
    m = IronAirBatteryModel({"Q_nom": 4.0, "R_int": 0.100, "V_min": 0.8, "V_max": 1.5,
                              "ocv_flat": 1.28, "ocv_droop": 0.15})
    # 2 A for 3600 s = 2 Ah; delta_SOC = 2/4 = 0.5
    r = m.evaluate(soc=0.9, I=2.0, dt=3600.0)
    assert_true(abs(r["SOC_new"] - 0.4) < 1e-6, f"SOC_new={r['SOC_new']} should be 0.4")


def test_energy_positive():
    print("\nTest 5: Energy is positive")
    m = IronAirBatteryModel({"Q_nom": 4.0, "R_int": 0.100, "V_min": 0.8, "V_max": 1.5,
                              "ocv_flat": 1.28, "ocv_droop": 0.15})
    r = m.evaluate(soc=0.7)
    assert_true(r["energy_Wh"] > 0, f"energy_Wh={r['energy_Wh']} should be > 0")


def test_component_model():
    print("\nTest 6: ComponentModel wrapper")
    cm = ComponentModel()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC033", "component_id mismatch")
    r = cm.predict({"soc": 0.5, "I": 1.0, "dt": 60.0})
    assert_true(r["SOC_new"] < 0.5, "SOC should decrease during discharge")


if __name__ == "__main__":
    tests = [
        test_ocv_at_full,
        test_ocv_drops_to_low,
        test_r_int_drop,
        test_soc_update,
        test_energy_positive,
        test_component_model,
    ]
    p = f = 0
    for t in tests:
        try:
            t(); p += 1
        except Exception as e:
            f += 1; print(f"  ERROR: {e}")
    print(f"\n{'='*50}\nEC033 F1a -- {p} passed, {f} failed\n{'='*50}")
    sys.exit(0 if f == 0 else 1)
