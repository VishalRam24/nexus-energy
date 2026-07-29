"""EC121 -- High Temperature Gas Reactor (HTGR) -- F1a Power-Map -- Test Suite"""
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

    print("EC121 High Temperature Gas Reactor -- F1a Power Map")
    print("=" * 50)

    # Test 1: identity
    assert_true(info["ec_id"] == "EC121", "ec_id == EC121")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 2: output keys
    r = model.predict({"load_factor": 1.0})
    for k in ["load_factor_clamped", "P_thermal_mw", "P_electric_mw", "eta_thermal"]:
        assert_true(k in r, f"output key: {k}")

    # Test 3: rated output = 0.47 * 600 = 282 MW_e
    expected_el = 0.47 * 600.0
    assert_true(abs(r["P_electric_mw"] - expected_el) < 0.1,
                f"P_electric at full load = {expected_el:.1f} MW_e (got {r['P_electric_mw']:.1f})")

    # Test 4: rated thermal = 600 MW
    assert_true(abs(r["P_thermal_mw"] - 600.0) < 0.1,
                f"P_thermal at full load = 600 MW (got {r['P_thermal_mw']:.1f})")

    # Test 5: linear scaling
    r_half = model.predict({"load_factor": 0.5})
    assert_true(abs(r_half["P_electric_mw"] - expected_el / 2) < 0.1,
                f"P_electric at LF=0.5 = {expected_el/2:.1f} MW_e")

    # Test 6: load factor clamping (min=0.4)
    r_low = model.predict({"load_factor": 0.1})
    assert_true(abs(r_low["load_factor_clamped"] - 0.4) < 1e-6, "LF clamped to min 0.4")
    r_high = model.predict({"load_factor": 1.5})
    assert_true(abs(r_high["load_factor_clamped"] - 1.0) < 1e-6, "LF clamped to max 1.0")

    # Test 7: eta = 0.47 constant
    for lf in [0.4, 0.7, 1.0]:
        r2 = model.predict({"load_factor": lf})
        assert_true(abs(r2["eta_thermal"] - 0.47) < 1e-6, f"eta=0.47 at LF={lf}")

    # Test 8: HTGR efficiency > FBR (0.47 > 0.40) — documented advantage
    assert_true(r["eta_thermal"] > 0.40, "HTGR eta > FBR eta (high temperature advantage)")

    # Test 9: benchmark
    t0 = time.perf_counter()
    for _ in range(1000):
        model.predict({"load_factor": 0.8})
    elapsed = time.perf_counter() - t0
    print(f"  \u2713 Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions < 1 s")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()
