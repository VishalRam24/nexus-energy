"""EC030 -- NiCd Battery -- F1a -- Test suite"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import NiCdBatteryModel
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"

def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def test_ocv_range():
    print("\nTest 1: OCV in 1.0-1.35 V range")
    m = NiCdBatteryModel({"Q_nom": 2.0, "R_int": 0.020, "V_min": 1.0, "V_max": 1.55,
                           "ocv_flat": 1.2, "ocv_rise": 0.08, "ocv_droop": 0.12})
    v_full  = m.evaluate(soc=1.0)["OCV"]
    v_empty = m.evaluate(soc=0.0)["OCV"]
    assert_true(1.0 <= v_full <= 1.4, f"Full OCV={v_full} should be in [1.0, 1.4]")
    assert_true(1.0 <= v_empty <= 1.3, f"Empty OCV={v_empty} should be in [1.0, 1.3]")


def test_flat_middle():
    print("\nTest 2: Flat OCV in middle SOC range")
    m = NiCdBatteryModel({"Q_nom": 2.0, "R_int": 0.020, "V_min": 1.0, "V_max": 1.55,
                           "ocv_flat": 1.2, "ocv_rise": 0.08, "ocv_droop": 0.12})
    v30 = m.evaluate(soc=0.3)["OCV"]
    v50 = m.evaluate(soc=0.5)["OCV"]
    v70 = m.evaluate(soc=0.7)["OCV"]
    spread = max(v30, v50, v70) - min(v30, v50, v70)
    assert_true(spread < 0.05, f"OCV spread in middle (30-70%): {spread:.4f} V should be < 0.05")


def test_low_r_int():
    print("\nTest 3: Low R_int=20 mOhm gives small voltage drop")
    m = NiCdBatteryModel({"Q_nom": 2.0, "R_int": 0.020, "V_min": 1.0, "V_max": 1.55,
                           "ocv_flat": 1.2, "ocv_rise": 0.08, "ocv_droop": 0.12})
    r0 = m.evaluate(soc=0.5, I=0.0)
    r1 = m.evaluate(soc=0.5, I=1.0)
    assert_true(abs(r0["V_terminal"] - r1["V_terminal"] - 0.020) < 1e-6,
                f"Voltage drop should be 0.020 V at 1 A")


def test_soc_update():
    print("\nTest 4: SOC decreases during discharge")
    m = NiCdBatteryModel({"Q_nom": 2.0, "R_int": 0.020, "V_min": 1.0, "V_max": 1.55,
                           "ocv_flat": 1.2, "ocv_rise": 0.08, "ocv_droop": 0.12})
    # 1 A for 3600 s = 1 Ah; delta_SOC = 1/2 = 0.5
    r = m.evaluate(soc=0.8, I=1.0, dt=3600.0)
    assert_true(abs(r["SOC_new"] - 0.3) < 1e-6,
                f"SOC_new={r['SOC_new']} should be 0.3")


def test_charge_raises_voltage():
    print("\nTest 5: Charge current raises terminal voltage")
    m = NiCdBatteryModel({"Q_nom": 2.0, "R_int": 0.020, "V_min": 1.0, "V_max": 1.55,
                           "ocv_flat": 1.2, "ocv_rise": 0.08, "ocv_droop": 0.12})
    r0   = m.evaluate(soc=0.5, I=0.0)
    r_ch = m.evaluate(soc=0.5, I=-1.0)
    assert_true(r_ch["V_terminal"] > r0["V_terminal"],
                f"Charge V={r_ch['V_terminal']} should > idle V={r0['V_terminal']}")


def test_component_model():
    print("\nTest 6: ComponentModel wrapper")
    cm = ComponentModel()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC030", "component_id mismatch")
    r = cm.predict({"soc": 0.6})
    assert_true(1.0 <= r["V_terminal"] <= 1.55, f"V_terminal={r['V_terminal']} out of range")


if __name__ == "__main__":
    tests = [
        test_ocv_range,
        test_flat_middle,
        test_low_r_int,
        test_soc_update,
        test_charge_raises_voltage,
        test_component_model,
    ]
    p = f = 0
    for t in tests:
        try:
            t(); p += 1
        except Exception as e:
            f += 1; print(f"  ERROR: {e}")
    print(f"\n{'='*50}\nEC030 F1a -- {p} passed, {f} failed\n{'='*50}")
    sys.exit(0 if f == 0 else 1)
