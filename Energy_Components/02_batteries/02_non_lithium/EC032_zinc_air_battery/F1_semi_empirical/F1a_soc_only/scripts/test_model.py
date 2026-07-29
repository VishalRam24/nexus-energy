"""EC032 -- Zinc-Air Battery -- F1a -- Test suite"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import ZincAirBatteryModel
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"

def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def test_flat_ocv():
    print("\nTest 1: OCV nearly flat ~1.65 V")
    m = ZincAirBatteryModel({"Q_nom": 3.0, "R_int": 0.050, "V_min": 1.0, "V_max": 1.9,
                              "ocv_flat": 1.65, "ocv_droop": 0.1})
    v_full = m.evaluate(soc=1.0)["OCV"]
    v_mid  = m.evaluate(soc=0.5)["OCV"]
    assert_true(abs(v_full - 1.65) < 1e-6, f"OCV at SOC=1 should be 1.65, got {v_full}")
    assert_true(abs(v_mid - v_full) < 0.03, f"OCV fairly flat at mid SOC: {v_mid:.4f}")


def test_ocv_droop():
    print("\nTest 2: OCV droops at low SOC")
    m = ZincAirBatteryModel({"Q_nom": 3.0, "R_int": 0.050, "V_min": 1.0, "V_max": 1.9,
                              "ocv_flat": 1.65, "ocv_droop": 0.1})
    v_full  = m.evaluate(soc=1.0)["OCV"]
    v_empty = m.evaluate(soc=0.0)["OCV"]
    assert_true(abs(v_full - v_empty - 0.1) < 1e-6,
                f"OCV drop should be 0.1 V, got {v_full - v_empty:.6f}")


def test_r_int_drop():
    print("\nTest 3: Voltage drop = I * R_int")
    m = ZincAirBatteryModel({"Q_nom": 3.0, "R_int": 0.050, "V_min": 1.0, "V_max": 1.9,
                              "ocv_flat": 1.65, "ocv_droop": 0.1})
    r0 = m.evaluate(soc=0.5, I=0.0)
    r1 = m.evaluate(soc=0.5, I=1.0)
    assert_true(abs(r0["V_terminal"] - r1["V_terminal"] - 0.050) < 1e-6,
                f"Drop should be 0.050 V at 1 A")


def test_soc_update():
    print("\nTest 4: SOC Coulomb counting")
    m = ZincAirBatteryModel({"Q_nom": 3.0, "R_int": 0.050, "V_min": 1.0, "V_max": 1.9,
                              "ocv_flat": 1.65, "ocv_droop": 0.1})
    # 3 A for 3600 s = 3 Ah = full capacity; SOC 1.0 -> 0.0
    r = m.evaluate(soc=1.0, I=3.0, dt=3600.0)
    assert_true(r["SOC_new"] == 0.0, f"Full discharge SOC_new should be 0, got {r['SOC_new']}")


def test_power_positive_discharge():
    print("\nTest 5: Power is positive during discharge")
    m = ZincAirBatteryModel({"Q_nom": 3.0, "R_int": 0.050, "V_min": 1.0, "V_max": 1.9,
                              "ocv_flat": 1.65, "ocv_droop": 0.1})
    r = m.evaluate(soc=0.5, I=1.0)
    assert_true(r["P"] > 0, f"P={r['P']} should be positive during discharge")


def test_component_model():
    print("\nTest 6: ComponentModel wrapper")
    cm = ComponentModel()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC032", "component_id mismatch")
    r = cm.predict({"soc": 0.8})
    assert_true(abs(r["OCV"] - 1.65) < 0.01, f"OCV at high SOC should be ~1.65 V")


if __name__ == "__main__":
    tests = [
        test_flat_ocv,
        test_ocv_droop,
        test_r_int_drop,
        test_soc_update,
        test_power_positive_discharge,
        test_component_model,
    ]
    p = f = 0
    for t in tests:
        try:
            t(); p += 1
        except Exception as e:
            f += 1; print(f"  ERROR: {e}")
    print(f"\n{'='*50}\nEC032 F1a -- {p} passed, {f} failed\n{'='*50}")
    sys.exit(0 if f == 0 else 1)
