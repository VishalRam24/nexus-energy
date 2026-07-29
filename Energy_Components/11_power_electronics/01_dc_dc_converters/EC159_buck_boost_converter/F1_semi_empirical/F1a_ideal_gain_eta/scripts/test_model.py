"""EC159 -- Buck-Boost Converter -- F1a Ideal Gain + Efficiency -- Test Suite"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def assert_true(condition, msg=""):
    if condition:
        print(f"  \u2713 {msg}")
    else:
        print(f"  \u2717 FAIL: {msg}")
        raise AssertionError(msg)


def run_tests():
    model = ComponentModel()
    info = model.get_info()

    print("EC159 Buck-Boost Converter -- F1a Ideal Gain + Efficiency")
    print("=" * 58)

    # Test 1: identity
    assert_true(info["ec_id"] == "EC159", "ec_id == EC159")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 2: output keys
    r = model.predict({"v_in": 24.0, "D": 0.5})
    for k in ["D_clamped", "voltage_gain", "V_out_mag", "V_in", "eta",
              "P_in_W", "P_out_W", "I_out_A", "I_in_A"]:
        assert_true(k in r, f"output key: {k}")

    # Test 3: ideal gain at D=0.5: V_out = 0.5/0.5 * 24 = 24 V (unity gain)
    assert_true(abs(r["V_out_mag"] - 24.0) < 1e-6,
                f"V_out at D=0.5, V_in=24V = 24V (unity gain, got {r['V_out_mag']:.4f})")

    # Test 4: gain = D/(1-D) formula
    for D_test, V_in in [(0.3, 12.0), (0.7, 48.0), (0.5, 36.0)]:
        r2 = model.predict({"v_in": V_in, "D": D_test})
        expected_gain = D_test / (1.0 - D_test)
        assert_true(abs(r2["voltage_gain"] - expected_gain) < 1e-9,
                    f"gain = D/(1-D) at D={D_test} (expected {expected_gain:.4f}, got {r2['voltage_gain']:.4f})")

    # Test 5: V_out = gain * V_in
    r3 = model.predict({"v_in": 12.0, "D": 0.3})
    expected_V = (0.3 / 0.7) * 12.0
    assert_true(abs(r3["V_out_mag"] - expected_V) < 1e-6,
                f"V_out = gain * V_in (expected {expected_V:.4f}, got {r3['V_out_mag']:.4f})")

    # Test 6: duty cycle clamping
    r_low = model.predict({"v_in": 24.0, "D": 0.05})  # below 0.1
    assert_true(abs(r_low["D_clamped"] - 0.1) < 1e-6, "D clamped to min 0.1")
    r_high = model.predict({"v_in": 24.0, "D": 0.95})  # above 0.9
    assert_true(abs(r_high["D_clamped"] - 0.9) < 1e-6, "D clamped to max 0.9")

    # Test 7: eta = 0.88
    assert_true(abs(r["eta"] - 0.88) < 1e-6, "eta = 0.88")

    # Test 8: energy balance with load
    r4 = model.predict({"v_in": 24.0, "D": 0.5, "P_out": 100.0})
    expected_P_in = 100.0 / 0.88
    assert_true(abs(r4["P_in_W"] - expected_P_in) < 1e-4,
                f"P_in = P_out/eta = {expected_P_in:.4f} W (got {r4['P_in_W']:.4f})")

    # Test 9: V_out increases monotonically with D
    V_prev = 0.0
    for D in [0.2, 0.4, 0.6, 0.8]:
        r_d = model.predict({"v_in": 24.0, "D": D})
        assert_true(r_d["V_out_mag"] > V_prev, f"V_out increases with D at D={D}")
        V_prev = r_d["V_out_mag"]

    # Test 10: duty_cycle_for_voltage roundtrip
    m = model._model
    v_in, v_out_target = 12.0, 36.0
    D_calc = m.duty_cycle_for_voltage(v_in, v_out_target)
    r5 = model.predict({"v_in": v_in, "D": D_calc})
    assert_true(abs(r5["V_out_mag"] - v_out_target) < 0.01,
                f"duty_cycle_for_voltage roundtrip: V_out={r5['V_out_mag']:.3f} V (target {v_out_target})")

    # Test 11: D=0.9 gives maximum step-up: gain = 9
    r_max = model.predict({"v_in": 10.0, "D": 0.9})
    assert_true(abs(r_max["voltage_gain"] - 9.0) < 1e-6,
                f"D=0.9: gain = 9 (got {r_max['voltage_gain']:.4f})")

    # Test 12: benchmark
    t0 = time.perf_counter()
    for _ in range(1000):
        model.predict({"v_in": 24.0, "D": 0.5, "P_out": 100.0})
    elapsed = time.perf_counter() - t0
    print(f"  \u2713 Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions < 1 s")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()
