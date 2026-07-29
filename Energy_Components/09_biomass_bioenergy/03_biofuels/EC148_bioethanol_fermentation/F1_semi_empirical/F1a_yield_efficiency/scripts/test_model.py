"""EC148 -- Bioethanol Fermentation -- F1a Yield Model -- Test Suite"""
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

    print("EC148 Bioethanol Fermentation -- F1a Yield Model")
    print("=" * 50)

    # Test 1: identity
    assert_true(info["ec_id"] == "EC148", "ec_id == EC148")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 2: output keys
    r = model.predict({"sugar_input_kg_per_h": 1000.0})
    for k in ["ethanol_L_per_h", "ethanol_kg_per_h", "CO2_kg_per_h",
              "energy_output_MW", "eta_conversion"]:
        assert_true(k in r, f"output key: {k}")

    # Test 3: ethanol_L = 1000 * 0.45 * 0.90 = 405 L/h
    expected_L = 1000.0 * 0.45 * 0.90
    assert_true(abs(r["ethanol_L_per_h"] - expected_L) < 0.01,
                f"ethanol = {expected_L:.1f} L/h (got {r['ethanol_L_per_h']:.2f})")

    # Test 4: ethanol_kg = ethanol_L * 0.789
    expected_kg = expected_L * 0.789
    assert_true(abs(r["ethanol_kg_per_h"] - expected_kg) < 0.01,
                f"ethanol = {expected_kg:.2f} kg/h (got {r['ethanol_kg_per_h']:.2f})")

    # Test 5: CO2 > 0
    assert_true(r["CO2_kg_per_h"] > 0, "CO2 co-product > 0")

    # Test 6: energy output > 0
    assert_true(r["energy_output_MW"] > 0, "energy_output > 0")

    # Test 7: eta_conversion = 0.90
    assert_true(abs(r["eta_conversion"] - 0.90) < 1e-6, "eta_conversion = 0.90")

    # Test 8: below theoretical max (0.511 L/kg at 100% conversion)
    assert_true(r["ethanol_L_per_h"] / 1000.0 < 0.511,
                "yield < theoretical maximum (0.511 L/kg)")

    # Test 9: zero input -> zero output
    r0 = model.predict({"sugar_input_kg_per_h": 0.0})
    assert_true(r0["ethanol_L_per_h"] == 0.0, "zero input -> zero ethanol")

    # Test 10: linear scaling
    r2 = model.predict({"sugar_input_kg_per_h": 2000.0})
    assert_true(abs(r2["ethanol_L_per_h"] - 2 * r["ethanol_L_per_h"]) < 0.01,
                "ethanol scales linearly")

    # Test 11: benchmark
    t0 = time.perf_counter()
    for _ in range(1000):
        model.predict({"sugar_input_kg_per_h": 500.0})
    elapsed = time.perf_counter() - t0
    print(f"  \u2713 Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions < 1 s")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()
