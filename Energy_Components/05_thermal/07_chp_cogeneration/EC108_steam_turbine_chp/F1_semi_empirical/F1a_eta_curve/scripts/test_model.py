"""EC108 -- Steam Turbine CHP -- F1a Efficiency Curve -- Test Suite"""
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

    print("EC108 Steam Turbine CHP -- F1a Efficiency Curve")
    print("=" * 50)

    # Test 1: component ID
    assert_true(info["ec_id"] == "EC108", "ec_id == EC108")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 2: predict keys
    r = model.predict({"PLR": 1.0})
    for k in ["PLR_clamped", "P_el_kw", "Q_th_kw", "P_fuel_kw", "eta_el", "eta_th", "eta_total", "HPR"]:
        assert_true(k in r, f"output key present: {k}")

    # Test 3: rated power at PLR=1.0 => P_el = 10000 kW
    assert_true(abs(r["P_el_kw"] - 10000.0) < 1.0,
                f"P_el at full load = 10000 kW (got {r['P_el_kw']:.1f})")

    # Test 4: eta_el = 0.30, eta_th = 0.45
    assert_true(abs(r["eta_el"] - 0.30) < 1e-6, "eta_el = 0.30")
    assert_true(abs(r["eta_th"] - 0.45) < 1e-6, "eta_th = 0.45")

    # Test 5: energy balance: P_el + Q_th <= P_fuel (equality at rated eta)
    fuel = r["P_fuel_kw"]
    assert_true(abs(r["P_el_kw"] + r["Q_th_kw"] - fuel * (0.30 + 0.45)) < 1.0,
                "energy balance: P_el + Q_th = (eta_el + eta_th) * P_fuel")

    # Test 6: HPR = Q_th / P_el
    assert_true(abs(r["HPR"] - r["Q_th_kw"] / r["P_el_kw"]) < 1e-6,
                f"HPR = Q_th / P_el (got {r['HPR']:.3f})")

    # Test 7: linear scaling with PLR
    r_half = model.predict({"PLR": 0.5})
    assert_true(abs(r_half["P_el_kw"] - 5000.0) < 1.0,
                f"P_el at PLR=0.5 = 5000 kW (got {r_half['P_el_kw']:.1f})")

    # Test 8: PLR clamping
    r_low = model.predict({"PLR": 0.1})  # below min 0.3
    assert_true(abs(r_low["PLR_clamped"] - 0.3) < 1e-6, "PLR clamped to 0.3 (min)")
    r_high = model.predict({"PLR": 1.5})  # above max
    assert_true(abs(r_high["PLR_clamped"] - 1.0) < 1e-6, "PLR clamped to 1.0 (max)")

    # Test 9: power always positive
    for plr in [0.3, 0.5, 0.8, 1.0]:
        r2 = model.predict({"PLR": plr})
        assert_true(r2["P_el_kw"] > 0 and r2["Q_th_kw"] > 0,
                    f"P_el and Q_th > 0 at PLR={plr}")

    # Test 10: benchmark
    t0 = time.perf_counter()
    for _ in range(1000):
        model.predict({"PLR": 0.75})
    elapsed = time.perf_counter() - t0
    print(f"  \u2713 Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions < 1 s")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()
