"""EC146 -- Torrefaction Reactor -- F1a Yield Model -- Test Suite"""
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

    print("EC146 Torrefaction Reactor -- F1a Yield Model")
    print("=" * 50)

    # Test 1: identity
    assert_true(info["ec_id"] == "EC146", "ec_id == EC146")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 2: output keys
    r = model.predict({"feedstock_dry_kg_per_h": 1000.0})
    for k in ["torrefied_solid_kg_per_h", "volatile_loss_kg_per_h", "energy_in_MW",
              "energy_out_MW", "energy_yield", "LHV_torrefied_MJ_kg", "energy_density_factor"]:
        assert_true(k in r, f"output key: {k}")

    # Test 3: torrefied solid = 1000 * 0.70 = 700 kg/h
    assert_true(abs(r["torrefied_solid_kg_per_h"] - 700.0) < 0.01,
                f"torrefied = 700 kg/h (got {r['torrefied_solid_kg_per_h']:.2f})")

    # Test 4: volatile loss = 1000 * 0.30 = 300 kg/h
    assert_true(abs(r["volatile_loss_kg_per_h"] - 300.0) < 0.01,
                f"volatile loss = 300 kg/h (got {r['volatile_loss_kg_per_h']:.2f})")

    # Test 5: mass balance: solid + volatiles = feed
    assert_true(abs(r["torrefied_solid_kg_per_h"] + r["volatile_loss_kg_per_h"] - 1000.0) < 0.01,
                "solid + volatiles = 1000 kg/h (mass balance)")

    # Test 6: energy_density_factor = 1.3
    assert_true(abs(r["energy_density_factor"] - 1.30) < 1e-6, "energy_density_factor = 1.30")

    # Test 7: LHV_torrefied = 23.4 MJ/kg (18.0 * 1.3)
    assert_true(abs(r["LHV_torrefied_MJ_kg"] - 23.4) < 0.01,
                f"LHV_torrefied = 23.4 MJ/kg (got {r['LHV_torrefied_MJ_kg']:.2f})")

    # Test 8: energy_out <= energy_in (no energy creation)
    assert_true(r["energy_out_MW"] <= r["energy_in_MW"] + 1e-6,
                "energy_out <= energy_in (2nd law)")

    # Test 9: zero input -> zero output
    r0 = model.predict({"feedstock_dry_kg_per_h": 0.0})
    assert_true(r0["torrefied_solid_kg_per_h"] == 0.0, "zero input -> zero solid")

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
