"""EC150 -- Fischer-Tropsch Synthesis (BTL) -- F1a Yield Model -- Test Suite"""
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

    print("EC150 Fischer-Tropsch Synthesis (BTL) -- F1a Yield Model")
    print("=" * 57)

    # Test 1: identity
    assert_true(info["ec_id"] == "EC150", "ec_id == EC150")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 2: output keys
    r = model.predict({"syngas_flow_Nm3_per_h": 1000.0, "CO_fraction_in": 0.40})
    for k in ["CO_reacted_Nm3_per_h", "FT_liquid_kg_per_h", "diesel_kg_per_h",
              "naphtha_kg_per_h", "wax_kg_per_h", "light_gas_kg_per_h",
              "energy_output_MW", "CO_conversion"]:
        assert_true(k in r, f"output key: {k}")

    # Test 3: CO_reacted = 1000 * 0.40 * 0.80 = 320 Nm3/h
    expected_CO = 1000.0 * 0.40 * 0.80
    assert_true(abs(r["CO_reacted_Nm3_per_h"] - expected_CO) < 0.01,
                f"CO_reacted = {expected_CO:.0f} Nm3/h (got {r['CO_reacted_Nm3_per_h']:.2f})")

    # Test 4: CO_conversion = 0.80
    assert_true(abs(r["CO_conversion"] - 0.80) < 1e-6, "CO_conversion = 0.80")

    # Test 5: FT_liquid > 0
    assert_true(r["FT_liquid_kg_per_h"] > 0, "FT_liquid > 0")

    # Test 6: diesel + naphtha + wax = FT_liquid (all liquid products)
    liquid_sum = r["diesel_kg_per_h"] + r["naphtha_kg_per_h"] + r["wax_kg_per_h"]
    assert_true(abs(liquid_sum - r["FT_liquid_kg_per_h"]) < 0.01,
                "diesel + naphtha + wax = FT_liquid")

    # Test 7: energy output > 0
    assert_true(r["energy_output_MW"] > 0, "energy_output > 0")

    # Test 8: light_gas_kg > 0 (always some C1-C4 by-product)
    assert_true(r["light_gas_kg_per_h"] > 0, "light gas > 0 (C1-C4 by-product)")

    # Test 9: zero syngas -> zero products
    r0 = model.predict({"syngas_flow_Nm3_per_h": 0.0, "CO_fraction_in": 0.40})
    assert_true(r0["FT_liquid_kg_per_h"] == 0.0, "zero syngas -> zero FT liquid")

    # Test 10: linear scaling
    r2 = model.predict({"syngas_flow_Nm3_per_h": 2000.0, "CO_fraction_in": 0.40})
    assert_true(abs(r2["FT_liquid_kg_per_h"] - 2 * r["FT_liquid_kg_per_h"]) < 0.01,
                "FT_liquid scales linearly with syngas flow")

    # Test 11: ASF check - at alpha=0.85, liquid (C5+) fraction > 0.5
    m = model._model
    lf = m._asf_liquid_fraction(0.85)
    assert_true(lf > 0.5, f"ASF liquid fraction at alpha=0.85 > 0.5 (got {lf:.3f})")

    # Test 12: benchmark
    t0 = time.perf_counter()
    for _ in range(1000):
        model.predict({"syngas_flow_Nm3_per_h": 1000.0, "CO_fraction_in": 0.40})
    elapsed = time.perf_counter() - t0
    print(f"  \u2713 Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions < 1 s")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()
