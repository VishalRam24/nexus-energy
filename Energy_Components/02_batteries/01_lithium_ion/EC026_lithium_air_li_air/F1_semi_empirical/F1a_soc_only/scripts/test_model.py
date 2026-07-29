"""EC026 -- Li-Air Battery -- F1a -- Test suite"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import LiAirBatteryModel
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"

def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def test_flat_ocv():
    print("\nTest 1: OCV nearly flat ~2.96 V")
    m = LiAirBatteryModel({"Q_nom": 10.0, "R_int": 0.200, "V_min": 2.0, "V_max": 3.2,
                            "ocv_flat": 2.96, "ocv_droop": 0.3})
    v_full = m.evaluate(soc=1.0)["OCV"]
    v_mid  = m.evaluate(soc=0.5)["OCV"]
    assert_true(abs(v_full - 2.96) < 1e-6, f"OCV at SOC=1 should be exactly 2.96, got {v_full}")
    assert_true(abs(v_mid - v_full) < 0.05, f"OCV at SOC=0.5 should be close to 2.96, got {v_mid}")


def test_droop_at_low_soc():
    print("\nTest 2: OCV droops at low SOC (pore clogging)")
    m = LiAirBatteryModel({"Q_nom": 10.0, "R_int": 0.200, "V_min": 2.0, "V_max": 3.2,
                            "ocv_flat": 2.96, "ocv_droop": 0.3})
    v_full = m.evaluate(soc=1.0)["OCV"]
    v_empty = m.evaluate(soc=0.0)["OCV"]
    expected_drop = 0.3
    assert_true(abs(v_full - v_empty - expected_drop) < 1e-6,
                f"OCV drop should be {expected_drop} V, got {v_full - v_empty:.6f}")


def test_high_r_int():
    print("\nTest 3: High R_int=200 mOhm causes significant voltage drop")
    m = LiAirBatteryModel({"Q_nom": 10.0, "R_int": 0.200, "V_min": 2.0, "V_max": 3.2,
                            "ocv_flat": 2.96, "ocv_droop": 0.3})
    r0 = m.evaluate(soc=0.5, I=0.0)
    r5 = m.evaluate(soc=0.5, I=5.0)
    drop = r0["V_terminal"] - r5["V_terminal"]
    assert_true(abs(drop - 1.0) < 1e-6, f"Voltage drop at 5 A = 5*0.2=1.0 V, got {drop:.6f}")


def test_soc_update():
    print("\nTest 4: SOC update by Coulomb counting")
    m = LiAirBatteryModel({"Q_nom": 10.0, "R_int": 0.200, "V_min": 2.0, "V_max": 3.2,
                            "ocv_flat": 2.96, "ocv_droop": 0.3})
    # 1 A for 3600 s = 1 Ah; delta_SOC = 1/10 = 0.1
    r = m.evaluate(soc=0.9, I=1.0, dt=3600.0)
    assert_true(abs(r["SOC_new"] - 0.8) < 1e-6, f"SOC_new={r['SOC_new']} should be 0.8")


def test_high_energy():
    print("\nTest 5: High energy density (10 Ah @ ~2.96 V)")
    m = LiAirBatteryModel({"Q_nom": 10.0, "R_int": 0.200, "V_min": 2.0, "V_max": 3.2,
                            "ocv_flat": 2.96, "ocv_droop": 0.3})
    r = m.evaluate(soc=1.0)
    assert_true(r["energy_Wh"] > 20.0, f"Energy={r['energy_Wh']:.2f} Wh should be > 20 Wh")


def test_component_model():
    print("\nTest 6: ComponentModel wrapper")
    cm = ComponentModel()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC026", "component_id mismatch")
    r = cm.predict({"soc": 0.5, "I": 1.0})
    assert_true("V_terminal" in r and "OCV" in r, "Missing outputs")
    assert_true(r["V_terminal"] < r["OCV"], "Discharge reduces terminal voltage below OCV")


if __name__ == "__main__":
    tests = [
        test_flat_ocv,
        test_droop_at_low_soc,
        test_high_r_int,
        test_soc_update,
        test_high_energy,
        test_component_model,
    ]
    p = f = 0
    for t in tests:
        try:
            t(); p += 1
        except Exception as e:
            f += 1; print(f"  ERROR: {e}")
    print(f"\n{'='*50}\nEC026 F1a -- {p} passed, {f} failed\n{'='*50}")
    sys.exit(0 if f == 0 else 1)
