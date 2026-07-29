"""EC142 -- Biogas Upgrading to Biomethane -- F1a Yield Model -- Test Suite"""
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

    print("EC142 Biogas Upgrading to Biomethane -- F1a Yield Model")
    print("=" * 55)

    # Test 1: identity
    assert_true(info["ec_id"] == "EC142", "ec_id == EC142")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 2: output keys
    r = model.predict({"biogas_flow_Nm3_per_h": 100.0, "CH4_fraction_in": 0.60})
    for k in ["biomethane_Nm3_per_h", "CH4_recovered_Nm3_per_h", "electricity_kW",
              "energy_output_kW", "CH4_recovery", "biomethane_purity"]:
        assert_true(k in r, f"output key: {k}")

    # Test 3: CH4_recovered = 100 * 0.60 * 0.97 = 58.2 Nm3/h
    expected_CH4 = 100.0 * 0.60 * 0.97
    assert_true(abs(r["CH4_recovered_Nm3_per_h"] - expected_CH4) < 0.01,
                f"CH4_recovered = {expected_CH4:.2f} Nm3/h (got {r['CH4_recovered_Nm3_per_h']:.2f})")

    # Test 4: biomethane volume = CH4_recovered / purity
    expected_bm = expected_CH4 / 0.97
    assert_true(abs(r["biomethane_Nm3_per_h"] - expected_bm) < 0.01,
                f"biomethane = {expected_bm:.2f} Nm3/h (got {r['biomethane_Nm3_per_h']:.2f})")

    # Test 5: electricity = 100 * 0.25 = 25 kW
    assert_true(abs(r["electricity_kW"] - 25.0) < 0.01,
                f"electricity = 25 kW (got {r['electricity_kW']:.2f})")

    # Test 6: CH4_recovery = 0.97
    assert_true(abs(r["CH4_recovery"] - 0.97) < 1e-6, "CH4_recovery = 0.97")

    # Test 7: purity = 0.97
    assert_true(abs(r["biomethane_purity"] - 0.97) < 1e-6, "biomethane_purity = 0.97")

    # Test 8: zero input -> zero output
    r0 = model.predict({"biogas_flow_Nm3_per_h": 0.0})
    assert_true(r0["biomethane_Nm3_per_h"] == 0.0, "zero input -> zero biomethane")

    # Test 9: linear scaling
    r2 = model.predict({"biogas_flow_Nm3_per_h": 200.0, "CH4_fraction_in": 0.60})
    assert_true(abs(r2["CH4_recovered_Nm3_per_h"] - 2 * r["CH4_recovered_Nm3_per_h"]) < 0.01,
                "output scales linearly with flow")

    # Test 10: energy_output > electricity (net positive)
    assert_true(r["energy_output_kW"] > r["electricity_kW"],
                "energy_output > electricity consumption")

    # Test 11: benchmark
    t0 = time.perf_counter()
    for _ in range(1000):
        model.predict({"biogas_flow_Nm3_per_h": 100.0, "CH4_fraction_in": 0.60})
    elapsed = time.perf_counter() - t0
    print(f"  \u2713 Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions < 1 s")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()
