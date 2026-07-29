"""EC034 -- Aluminum-Ion Battery -- F1a -- Test suite"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import AluminumIonBatteryModel
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"

def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def test_ocv_range():
    print("\nTest 1: OCV ~2.0 V at full charge")
    m = AluminumIonBatteryModel({"Q_nom": 1.0, "R_int": 0.080, "V_min": 1.5, "V_max": 2.3,
                                   "ocv_coeffs": [2.0, -0.2, 0.1, -0.05]})
    v_full  = m.evaluate(soc=1.0)["OCV"]
    v_empty = m.evaluate(soc=0.0)["OCV"]
    assert_true(1.7 <= v_full <= 2.3, f"Full OCV={v_full} should be ~2.0 V")
    assert_true(v_full > v_empty, f"Full OCV should > empty OCV")


def test_r_int_drop():
    print("\nTest 2: Voltage drop = I * R_int = 0.080 V at 1 A")
    m = AluminumIonBatteryModel({"Q_nom": 1.0, "R_int": 0.080, "V_min": 1.5, "V_max": 2.3,
                                   "ocv_coeffs": [2.0, -0.2, 0.1, -0.05]})
    r0 = m.evaluate(soc=0.5, I=0.0)
    r1 = m.evaluate(soc=0.5, I=1.0)
    assert_true(abs(r0["V_terminal"] - r1["V_terminal"] - 0.080) < 1e-6,
                f"Voltage drop should be 0.080 V at 1 A")


def test_soc_update():
    print("\nTest 3: SOC update by Coulomb counting")
    m = AluminumIonBatteryModel({"Q_nom": 1.0, "R_int": 0.080, "V_min": 1.5, "V_max": 2.3,
                                   "ocv_coeffs": [2.0, -0.2, 0.1, -0.05]})
    # 1 A for 1800 s = 0.5 Ah; delta_SOC = 0.5/1.0 = 0.5
    r = m.evaluate(soc=0.8, I=1.0, dt=1800.0)
    assert_true(abs(r["SOC_new"] - 0.3) < 1e-6, f"SOC_new={r['SOC_new']} should be 0.3")


def test_soc_clamping():
    print("\nTest 4: SOC clamped to [0, 1]")
    m = AluminumIonBatteryModel({"Q_nom": 1.0, "R_int": 0.080, "V_min": 1.5, "V_max": 2.3,
                                   "ocv_coeffs": [2.0, -0.2, 0.1, -0.05]})
    r = m.evaluate(soc=0.0, I=5.0, dt=3600.0)
    assert_true(r["SOC_new"] == 0.0, "SOC_new should not go below 0")


def test_charge_mode():
    print("\nTest 5: Charge current (I<0) raises terminal voltage")
    m = AluminumIonBatteryModel({"Q_nom": 1.0, "R_int": 0.080, "V_min": 1.5, "V_max": 2.3,
                                   "ocv_coeffs": [2.0, -0.2, 0.1, -0.05]})
    r0  = m.evaluate(soc=0.5, I=0.0)
    rch = m.evaluate(soc=0.5, I=-1.0)
    assert_true(rch["V_terminal"] > r0["V_terminal"],
                f"Charge V={rch['V_terminal']} should > idle V={r0['V_terminal']}")


def test_component_model():
    print("\nTest 6: ComponentModel wrapper")
    cm = ComponentModel()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC034", "component_id mismatch")
    r = cm.predict({"soc": 0.5})
    assert_true(1.5 <= r["V_terminal"] <= 2.3, f"V_terminal={r['V_terminal']} out of range")


if __name__ == "__main__":
    tests = [
        test_ocv_range,
        test_r_int_drop,
        test_soc_update,
        test_soc_clamping,
        test_charge_mode,
        test_component_model,
    ]
    p = f = 0
    for t in tests:
        try:
            t(); p += 1
        except Exception as e:
            f += 1; print(f"  ERROR: {e}")
    print(f"\n{'='*50}\nEC034 F1a -- {p} passed, {f} failed\n{'='*50}")
    sys.exit(0 if f == 0 else 1)
