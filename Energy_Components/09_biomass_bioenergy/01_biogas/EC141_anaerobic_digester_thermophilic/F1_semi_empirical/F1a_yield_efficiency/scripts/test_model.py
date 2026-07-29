"""EC141 -- Anaerobic Digester (Thermophilic) -- F1a Yield Model -- Test Suite"""
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

    print("EC141 Thermophilic Anaerobic Digester -- F1a Yield Model")
    print("=" * 55)

    # Test 1: identity
    assert_true(info["ec_id"] == "EC141", "ec_id == EC141")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 2: output keys
    r = model.predict({"feedstock_VS_kg_per_day": 1000.0})
    for k in ["VS_destroyed_kg_per_day", "biogas_m3_per_day", "CH4_m3_per_day",
              "CO2_m3_per_day", "energy_kWh_per_day", "CH4_fraction"]:
        assert_true(k in r, f"output key: {k}")

    # Test 3: VS destroyed = 1000 * 0.65 = 650 kgVS/day
    assert_true(abs(r["VS_destroyed_kg_per_day"] - 650.0) < 0.1,
                f"VS_destroyed = 650 kgVS/day (got {r['VS_destroyed_kg_per_day']:.1f})")

    # Test 4: biogas = 650 * 0.50 = 325 m3/day
    assert_true(abs(r["biogas_m3_per_day"] - 325.0) < 0.1,
                f"biogas = 325 m3/day (got {r['biogas_m3_per_day']:.1f})")

    # Test 5: CH4 = 325 * 0.60 = 195 m3/day
    assert_true(abs(r["CH4_m3_per_day"] - 195.0) < 0.1,
                f"CH4 = 195 m3/day (got {r['CH4_m3_per_day']:.1f})")

    # Test 6: CH4 + CO2 = biogas
    assert_true(abs(r["CH4_m3_per_day"] + r["CO2_m3_per_day"] - r["biogas_m3_per_day"]) < 0.01,
                "CH4 + CO2 = biogas (mass balance)")

    # Test 7: CH4 fraction = 0.60
    assert_true(abs(r["CH4_fraction"] - 0.60) < 1e-6, "CH4_fraction = 0.60")

    # Test 8: energy > 0
    assert_true(r["energy_kWh_per_day"] > 0, "energy > 0")

    # Test 9: zero input -> zero output
    r0 = model.predict({"feedstock_VS_kg_per_day": 0.0})
    assert_true(r0["biogas_m3_per_day"] == 0.0, "zero input -> zero biogas")

    # Test 10: linear scaling
    r2 = model.predict({"feedstock_VS_kg_per_day": 2000.0})
    assert_true(abs(r2["biogas_m3_per_day"] - 2 * r["biogas_m3_per_day"]) < 0.1,
                "biogas scales linearly with VS input")

    # Test 11: benchmark
    t0 = time.perf_counter()
    for _ in range(1000):
        model.predict({"feedstock_VS_kg_per_day": 500.0})
    elapsed = time.perf_counter() - t0
    print(f"  \u2713 Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions < 1 s")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()
