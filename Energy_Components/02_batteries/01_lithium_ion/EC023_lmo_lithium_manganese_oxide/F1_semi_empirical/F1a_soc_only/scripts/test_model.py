"""EC023 -- LMO Battery -- F1a -- Test suite"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import LMOBatteryModel
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"

def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def test_ocv_range():
    print("\nTest 1: OCV spans 3.0-4.2 V across SOC range")
    m = LMOBatteryModel({"Q_nom": 2.5, "R_int": 0.030, "V_min": 3.0, "V_max": 4.2,
                          "ocv_coeffs": [4.2, -0.8, 0.4, -0.2, 0.1]})
    v_full  = m.evaluate(soc=1.0)["OCV"]
    v_empty = m.evaluate(soc=0.0)["OCV"]
    assert_true(3.8 < v_full <= 4.2, f"Full OCV={v_full} should be ~4.2 V")
    assert_true(3.0 <= v_empty < 3.5, f"Empty OCV={v_empty} should be ~3.0 V")


def test_discharge_voltage_drop():
    print("\nTest 2: Discharge current reduces terminal voltage")
    m = LMOBatteryModel({"Q_nom": 2.5, "R_int": 0.030, "V_min": 3.0, "V_max": 4.2,
                          "ocv_coeffs": [4.2, -0.8, 0.4, -0.2, 0.1]})
    r_idle = m.evaluate(soc=0.5, I=0.0)
    r_dsch = m.evaluate(soc=0.5, I=1.0)
    assert_true(r_dsch["V_terminal"] < r_idle["V_terminal"],
                f"Discharge V={r_dsch['V_terminal']} should be < idle V={r_idle['V_terminal']}")
    assert_true(abs(r_idle["V_terminal"] - r_dsch["V_terminal"] - 0.030) < 1e-6,
                f"Voltage drop should be I*R_int = 0.030 V")


def test_charge_voltage_rise():
    print("\nTest 3: Charge current raises terminal voltage")
    m = LMOBatteryModel({"Q_nom": 2.5, "R_int": 0.030, "V_min": 3.0, "V_max": 4.2,
                          "ocv_coeffs": [4.2, -0.8, 0.4, -0.2, 0.1]})
    r_idle = m.evaluate(soc=0.5, I=0.0)
    r_chg  = m.evaluate(soc=0.5, I=-1.0)
    assert_true(r_chg["V_terminal"] > r_idle["V_terminal"],
                f"Charge V={r_chg['V_terminal']} should be > idle V={r_idle['V_terminal']}")


def test_soc_update():
    print("\nTest 4: SOC decreases during discharge")
    m = LMOBatteryModel({"Q_nom": 2.5, "R_int": 0.030, "V_min": 3.0, "V_max": 4.2,
                          "ocv_coeffs": [4.2, -0.8, 0.4, -0.2, 0.1]})
    # Discharge at 1 A for 1 hour = 1 Ah = 1/2.5 = 0.4 SOC decrease
    r = m.evaluate(soc=1.0, I=1.0, dt=3600.0)
    assert_true(abs(r["SOC_new"] - 0.6) < 1e-6,
                f"SOC after 1h@1A should be 0.6, got {r['SOC_new']}")


def test_soc_clamping():
    print("\nTest 5: SOC clamped to [0, 1]")
    m = LMOBatteryModel({"Q_nom": 2.5, "R_int": 0.030, "V_min": 3.0, "V_max": 4.2,
                          "ocv_coeffs": [4.2, -0.8, 0.4, -0.2, 0.1]})
    r = m.evaluate(soc=0.1, I=10.0, dt=3600.0)  # Would over-discharge
    assert_true(r["SOC_new"] >= 0.0, f"SOC_new={r['SOC_new']} should be >= 0")
    r2 = m.evaluate(soc=0.9, I=-10.0, dt=3600.0)  # Would over-charge
    assert_true(r2["SOC_new"] <= 1.0, f"SOC_new={r2['SOC_new']} should be <= 1")


def test_component_model_wrapper():
    print("\nTest 6: ComponentModel wrapper")
    cm = ComponentModel()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC023", "component_id mismatch")
    r = cm.predict({"soc": 0.7, "I": 0.5})
    assert_true("V_terminal" in r and "OCV" in r, "Missing outputs")
    assert_true(r["V_terminal"] < r["OCV"], "Discharge should reduce terminal voltage below OCV")


if __name__ == "__main__":
    tests = [
        test_ocv_range,
        test_discharge_voltage_drop,
        test_charge_voltage_rise,
        test_soc_update,
        test_soc_clamping,
        test_component_model_wrapper,
    ]
    p = f = 0
    for t in tests:
        try:
            t(); p += 1
        except Exception as e:
            f += 1; print(f"  ERROR: {e}")
    print(f"\n{'='*50}\nEC023 F1a -- {p} passed, {f} failed\n{'='*50}")
    sys.exit(0 if f == 0 else 1)
