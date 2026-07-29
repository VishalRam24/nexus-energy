"""EC147 -- Hydrothermal Liquefaction (HTL) -- F1a Yield Model -- Test Suite"""
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

    print("EC147 Hydrothermal Liquefaction -- F1a Yield Model")
    print("=" * 52)

    # Test 1: identity
    assert_true(info["ec_id"] == "EC147", "ec_id == EC147")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 2: output keys
    r = model.predict({"feedstock_dry_kg_per_h": 1000.0})
    for k in ["bio_crude_kg_per_h", "aqueous_kg_per_h", "gas_kg_per_h",
              "solid_kg_per_h", "energy_output_MW", "electricity_kW"]:
        assert_true(k in r, f"output key: {k}")

    # Test 3: bio_crude = 1000 * 0.35 = 350 kg/h
    assert_true(abs(r["bio_crude_kg_per_h"] - 350.0) < 0.01,
                f"bio_crude = 350 kg/h (got {r['bio_crude_kg_per_h']:.2f})")

    # Test 4: energy output > 0
    assert_true(r["energy_output_MW"] > 0, "energy_output > 0")

    # Test 5: electricity > 0 for non-zero feed
    assert_true(r["electricity_kW"] > 0, "electricity > 0 (parasitic load)")

    # Test 6: LHV check: energy = bio_crude * 33 MJ/kg / 3600 s
    expected_MW = 350.0 * 33.0 / 3600.0
    assert_true(abs(r["energy_output_MW"] - expected_MW) < 0.01,
                f"energy = {expected_MW:.3f} MW (got {r['energy_output_MW']:.3f})")

    # Test 7: aqueous + gas + solid + bio_crude = feedstock (mass balance)
    total_out = r["bio_crude_kg_per_h"] + r["aqueous_kg_per_h"] + r["gas_kg_per_h"] + r["solid_kg_per_h"]
    assert_true(abs(total_out - 1000.0) < 0.1,
                f"mass balance: all phases sum to ~1000 kg/h (got {total_out:.1f})")

    # Test 8: zero input -> zero output
    r0 = model.predict({"feedstock_dry_kg_per_h": 0.0})
    assert_true(r0["bio_crude_kg_per_h"] == 0.0, "zero input -> zero bio_crude")

    # Test 9: linear scaling
    r2 = model.predict({"feedstock_dry_kg_per_h": 2000.0})
    assert_true(abs(r2["bio_crude_kg_per_h"] - 2 * r["bio_crude_kg_per_h"]) < 0.01,
                "bio_crude scales linearly")

    # Test 10: benchmark
    t0 = time.perf_counter()
    for _ in range(1000):
        model.predict({"feedstock_dry_kg_per_h": 500.0})
    elapsed = time.perf_counter() - t0
    print(f"  \u2713 Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions < 1 s")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()
