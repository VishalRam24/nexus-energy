"""EC145 -- Pyrolysis Reactor -- F1a Yield Model -- Test Suite"""
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

    print("EC145 Pyrolysis Reactor -- F1a Yield Model")
    print("=" * 50)

    # Test 1: identity
    assert_true(info["ec_id"] == "EC145", "ec_id == EC145")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 2: output keys
    r = model.predict({"feedstock_dry_kg_per_h": 1000.0})
    for k in ["bio_oil_kg_per_h", "char_kg_per_h", "gas_kg_per_h",
              "energy_bio_oil_MW", "energy_char_MW", "energy_gas_MW", "mass_balance_check"]:
        assert_true(k in r, f"output key: {k}")

    # Test 3: bio_oil = 1000 * 0.60 = 600 kg/h
    assert_true(abs(r["bio_oil_kg_per_h"] - 600.0) < 0.01,
                f"bio_oil = 600 kg/h (got {r['bio_oil_kg_per_h']:.2f})")

    # Test 4: char = 1000 * 0.20 = 200 kg/h
    assert_true(abs(r["char_kg_per_h"] - 200.0) < 0.01,
                f"char = 200 kg/h (got {r['char_kg_per_h']:.2f})")

    # Test 5: gas = 1000 * 0.20 = 200 kg/h
    assert_true(abs(r["gas_kg_per_h"] - 200.0) < 0.01,
                f"gas = 200 kg/h (got {r['gas_kg_per_h']:.2f})")

    # Test 6: mass balance = 1.0 (bio_oil + char + gas yields sum to 1)
    assert_true(abs(r["mass_balance_check"] - 1.0) < 1e-6,
                f"mass balance = 1.0 (got {r['mass_balance_check']:.6f})")

    # Test 7: energy outputs > 0
    assert_true(r["energy_bio_oil_MW"] > 0 and r["energy_char_MW"] > 0 and r["energy_gas_MW"] > 0,
                "all energy outputs > 0")

    # Test 8: bio-oil has highest energy (60% yield * 17 MJ/kg > others)
    assert_true(r["energy_bio_oil_MW"] > r["energy_gas_MW"],
                "bio-oil energy > gas energy (dominant product)")

    # Test 9: zero input -> zero output
    r0 = model.predict({"feedstock_dry_kg_per_h": 0.0})
    assert_true(r0["bio_oil_kg_per_h"] == 0.0, "zero input -> zero bio-oil")

    # Test 10: linear scaling
    r2 = model.predict({"feedstock_dry_kg_per_h": 2000.0})
    assert_true(abs(r2["bio_oil_kg_per_h"] - 2 * r["bio_oil_kg_per_h"]) < 0.01,
                "bio-oil scales linearly")

    # Test 11: benchmark
    t0 = time.perf_counter()
    for _ in range(1000):
        model.predict({"feedstock_dry_kg_per_h": 500.0})
    elapsed = time.perf_counter() - t0
    print(f"  \u2713 Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions < 1 s")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()
