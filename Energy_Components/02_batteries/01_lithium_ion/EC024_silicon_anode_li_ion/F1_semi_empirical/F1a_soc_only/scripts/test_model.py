"""EC024 -- Silicon-Anode Li-ion Battery -- F1a -- Test suite"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import SiAnodeBatteryModel
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"

def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def test_ocv_range():
    print("\nTest 1: OCV spans 2.8-4.2 V")
    m = SiAnodeBatteryModel({"Q_nom": 4.0, "R_int": 0.050, "V_min": 2.8, "V_max": 4.2,
                               "ocv_coeffs": [4.2, -1.0, 0.6, -0.3, 0.15]})
    v_full  = m.evaluate(soc=1.0)["OCV"]
    v_empty = m.evaluate(soc=0.0)["OCV"]
    assert_true(3.8 <= v_full <= 4.2, f"Full OCV={v_full} should be ~4.2 V")
    assert_true(2.8 <= v_empty < 3.5, f"Empty OCV={v_empty} should be ~2.8 V")


def test_high_capacity():
    print("\nTest 2: High capacity Q_nom = 4.0 Ah")
    m = SiAnodeBatteryModel({"Q_nom": 4.0, "R_int": 0.050, "V_min": 2.8, "V_max": 4.2,
                               "ocv_coeffs": [4.2, -1.0, 0.6, -0.3, 0.15]})
    # Discharge 4 A for 1 h should fully deplete (SOC 1.0 -> 0.0)
    r = m.evaluate(soc=1.0, I=4.0, dt=3600.0)
    assert_true(r["SOC_new"] == 0.0, f"Full discharge SOC_new={r['SOC_new']} should be 0")


def test_discharge_voltage_drop():
    print("\nTest 3: Discharge causes I*R_int voltage drop")
    m = SiAnodeBatteryModel({"Q_nom": 4.0, "R_int": 0.050, "V_min": 2.8, "V_max": 4.2,
                               "ocv_coeffs": [4.2, -1.0, 0.6, -0.3, 0.15]})
    r0 = m.evaluate(soc=0.5, I=0.0)
    r1 = m.evaluate(soc=0.5, I=1.0)
    assert_true(abs(r0["V_terminal"] - r1["V_terminal"] - 0.050) < 1e-6,
                f"Voltage drop should be 0.050 V at 1 A")


def test_soc_update():
    print("\nTest 4: SOC update by Coulomb counting")
    m = SiAnodeBatteryModel({"Q_nom": 4.0, "R_int": 0.050, "V_min": 2.8, "V_max": 4.2,
                               "ocv_coeffs": [4.2, -1.0, 0.6, -0.3, 0.15]})
    # 1 A for 3600 s = 1 Ah; delta_SOC = 1/4.0 = 0.25
    r = m.evaluate(soc=0.8, I=1.0, dt=3600.0)
    assert_true(abs(r["SOC_new"] - 0.55) < 1e-6,
                f"SOC_new={r['SOC_new']} should be 0.55 (0.8 - 0.25)")


def test_ocv_decreases_with_soc():
    print("\nTest 5: OCV decreases as SOC decreases")
    m = SiAnodeBatteryModel({"Q_nom": 4.0, "R_int": 0.050, "V_min": 2.8, "V_max": 4.2,
                               "ocv_coeffs": [4.2, -1.0, 0.6, -0.3, 0.15]})
    ocvs = [m.evaluate(soc=s)["OCV"] for s in [1.0, 0.75, 0.5, 0.25, 0.0]]
    assert_true(all(ocvs[i] >= ocvs[i+1] for i in range(len(ocvs)-1)),
                f"OCV should decrease with SOC: {ocvs}")


def test_component_model():
    print("\nTest 6: ComponentModel wrapper")
    cm = ComponentModel()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC024", "component_id mismatch")
    r = cm.predict({"soc": 0.6, "I": 2.0, "dt": 60.0})
    assert_true("SOC_new" in r, "Missing SOC_new in output")
    assert_true(r["SOC_new"] < 0.6, f"SOC should decrease during discharge")


if __name__ == "__main__":
    tests = [
        test_ocv_range,
        test_high_capacity,
        test_discharge_voltage_drop,
        test_soc_update,
        test_ocv_decreases_with_soc,
        test_component_model,
    ]
    p = f = 0
    for t in tests:
        try:
            t(); p += 1
        except Exception as e:
            f += 1; print(f"  ERROR: {e}")
    print(f"\n{'='*50}\nEC024 F1a -- {p} passed, {f} failed\n{'='*50}")
    sys.exit(0 if f == 0 else 1)
