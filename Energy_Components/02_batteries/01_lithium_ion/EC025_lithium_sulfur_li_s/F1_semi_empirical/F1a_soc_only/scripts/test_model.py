"""EC025 -- Li-S Battery -- F1a -- Test suite"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import LiSBatteryModel
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"

def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def test_two_plateau_ocv():
    print("\nTest 1: Two distinct plateaus in OCV")
    m = LiSBatteryModel({"Q_nom": 5.0, "R_int": 0.100, "V_min": 1.7, "V_max": 2.45,
                          "V_high_plateau": 2.3, "V_low_plateau": 2.1, "soc_transition": 0.25})
    v_high = m.evaluate(soc=0.7)["OCV"]
    v_mid  = m.evaluate(soc=0.25)["OCV"]
    v_low  = m.evaluate(soc=0.1)["OCV"]
    assert_true(v_high > v_mid, f"High-SOC OCV={v_high} should > transition OCV={v_mid}")
    assert_true(v_mid > v_low,  f"Transition OCV={v_mid} should > low-SOC OCV={v_low}")
    assert_true(abs(v_high - 2.3) < 0.2, f"High plateau OCV~2.3 V, got {v_high}")
    assert_true(abs(v_low - 2.1) < 0.2,  f"Low plateau OCV~2.1 V, got {v_low}")


def test_plateau_flag():
    print("\nTest 2: Plateau flag switches at soc_transition")
    m = LiSBatteryModel({"Q_nom": 5.0, "R_int": 0.100, "V_min": 1.7, "V_max": 2.45,
                          "V_high_plateau": 2.3, "V_low_plateau": 2.1, "soc_transition": 0.25})
    assert_true(m.evaluate(soc=0.5)["plateau"] == "high",  "SOC=0.5 should be high plateau")
    assert_true(m.evaluate(soc=0.1)["plateau"] == "low",   "SOC=0.1 should be low plateau")
    assert_true(m.evaluate(soc=0.25)["plateau"] == "low",  "SOC=0.25 edge is low plateau")


def test_discharge_voltage_drop():
    print("\nTest 3: Discharge causes voltage drop = I*R_int")
    m = LiSBatteryModel({"Q_nom": 5.0, "R_int": 0.100, "V_min": 1.7, "V_max": 2.45,
                          "V_high_plateau": 2.3, "V_low_plateau": 2.1, "soc_transition": 0.25})
    r0 = m.evaluate(soc=0.5, I=0.0)
    r1 = m.evaluate(soc=0.5, I=1.0)
    assert_true(abs(r0["V_terminal"] - r1["V_terminal"] - 0.100) < 1e-6,
                f"Voltage drop should be 0.100 V at 1 A")


def test_soc_update():
    print("\nTest 4: SOC decreases by I*dt/Q")
    m = LiSBatteryModel({"Q_nom": 5.0, "R_int": 0.100, "V_min": 1.7, "V_max": 2.45,
                          "V_high_plateau": 2.3, "V_low_plateau": 2.1, "soc_transition": 0.25})
    # 1 A for 3600 s = 1 Ah; delta_SOC = 1/5 = 0.2
    r = m.evaluate(soc=0.8, I=1.0, dt=3600.0)
    assert_true(abs(r["SOC_new"] - 0.6) < 1e-6,
                f"SOC_new={r['SOC_new']} should be 0.6")


def test_high_capacity():
    print("\nTest 5: High capacity Q_nom=5.0 Ah")
    m = LiSBatteryModel({"Q_nom": 5.0, "R_int": 0.100, "V_min": 1.7, "V_max": 2.45,
                          "V_high_plateau": 2.3, "V_low_plateau": 2.1, "soc_transition": 0.25})
    r = m.evaluate(soc=1.0)
    energy = r["energy_Wh"]
    assert_true(energy > 8.0, f"Energy={energy:.2f} Wh should be > 8 Wh for 5 Ah @ ~2.2V")


def test_component_model():
    print("\nTest 6: ComponentModel wrapper")
    cm = ComponentModel()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC025", "component_id mismatch")
    r = cm.predict({"soc": 0.3, "I": 1.0})
    assert_true("plateau" in r, "Missing plateau in output")
    assert_true(r["V_terminal"] > 1.7, f"V_terminal={r['V_terminal']} below V_min")


if __name__ == "__main__":
    tests = [
        test_two_plateau_ocv,
        test_plateau_flag,
        test_discharge_voltage_drop,
        test_soc_update,
        test_high_capacity,
        test_component_model,
    ]
    p = f = 0
    for t in tests:
        try:
            t(); p += 1
        except Exception as e:
            f += 1; print(f"  ERROR: {e}")
    print(f"\n{'='*50}\nEC025 F1a -- {p} passed, {f} failed\n{'='*50}")
    sys.exit(0 if f == 0 else 1)
