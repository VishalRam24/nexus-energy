"""EC149 -- Biodiesel Transesterification -- F1a Yield Model -- Test Suite"""
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

    print("EC149 Biodiesel Transesterification -- F1a Yield Model")
    print("=" * 55)

    # Test 1: identity
    assert_true(info["ec_id"] == "EC149", "ec_id == EC149")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 2: output keys
    r = model.predict({"oil_input_kg_per_h": 1000.0})
    for k in ["FAME_kg_per_h", "FAME_L_per_h", "glycerol_kg_per_h",
              "methanol_consumed_kg_per_h", "energy_output_MW", "FAME_yield"]:
        assert_true(k in r, f"output key: {k}")

    # Test 3: FAME = 1000 * 0.95 = 950 kg/h
    assert_true(abs(r["FAME_kg_per_h"] - 950.0) < 0.01,
                f"FAME = 950 kg/h (got {r['FAME_kg_per_h']:.2f})")

    # Test 4: glycerol = 1000 * 0.10 = 100 kg/h
    assert_true(abs(r["glycerol_kg_per_h"] - 100.0) < 0.01,
                f"glycerol = 100 kg/h (got {r['glycerol_kg_per_h']:.2f})")

    # Test 5: FAME_L = FAME_kg / 0.875
    expected_L = 950.0 / 0.875
    assert_true(abs(r["FAME_L_per_h"] - expected_L) < 0.1,
                f"FAME_L = {expected_L:.1f} L/h (got {r['FAME_L_per_h']:.1f})")

    # Test 6: FAME yield = 0.95
    assert_true(abs(r["FAME_yield"] - 0.95) < 1e-6, "FAME_yield = 0.95")

    # Test 7: energy output > 0
    assert_true(r["energy_output_MW"] > 0, "energy_output > 0")

    # Test 8: methanol consumed > 0
    assert_true(r["methanol_consumed_kg_per_h"] > 0, "methanol consumed > 0")

    # Test 9: zero input -> zero output
    r0 = model.predict({"oil_input_kg_per_h": 0.0})
    assert_true(r0["FAME_kg_per_h"] == 0.0, "zero input -> zero FAME")

    # Test 10: linear scaling
    r2 = model.predict({"oil_input_kg_per_h": 2000.0})
    assert_true(abs(r2["FAME_kg_per_h"] - 2 * r["FAME_kg_per_h"]) < 0.01,
                "FAME scales linearly")

    # Test 11: benchmark
    t0 = time.perf_counter()
    for _ in range(1000):
        model.predict({"oil_input_kg_per_h": 500.0})
    elapsed = time.perf_counter() - t0
    print(f"  \u2713 Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions < 1 s")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()
