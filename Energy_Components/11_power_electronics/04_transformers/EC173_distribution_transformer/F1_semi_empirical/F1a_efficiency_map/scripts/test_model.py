"""EC173 -- Distribution Transformer -- F1a -- Test Suite"""
import sys, os, time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

PASS = "\u2713"
FAIL = "\u2717"


def assert_true(condition, msg):
    if condition:
        print(f"  {PASS} {msg}")
    else:
        print(f"  {FAIL} {msg}")
        raise AssertionError(msg)


def main():
    print("EC173 Distribution Transformer F1a -- Test Suite")
    model = ComponentModel()

    # Test 1: keys
    r = model.predict({"load_fraction": 1.0})
    assert_true(all(k in r for k in ["efficiency", "p_out_w", "p_in_w", "p_losses_w",
                                      "p_core_w", "p_copper_w"]),
                "predict returns required keys")

    # Test 2: get_info
    info = model.get_info()
    assert_true(info["ec_id"] == "EC173", "ec_id == EC173")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 3: efficiency < 1
    plr = np.linspace(0.05, 1.5, 100)
    r2 = model.predict({"load_fraction": plr})
    assert_true(np.all(r2["efficiency"] < 1.0), "Efficiency < 1 everywhere")

    # Test 4: efficiency > 0
    assert_true(np.all(r2["efficiency"] > 0.0), "Efficiency > 0 everywhere")

    # Test 5: peak efficiency near PLR = sqrt(P_core/P_cu) = 0.447
    peak_plr = model._model.peak_efficiency_load()
    plr_range = np.linspace(0.01, 1.5, 1000)
    r3 = model.predict({"load_fraction": plr_range})
    peak_idx = np.argmax(r3["efficiency"])
    assert_true(abs(plr_range[peak_idx] - peak_plr) < 0.01,
                f"Peak efficiency near PLR={peak_plr:.3f} (got {plr_range[peak_idx]:.3f})")

    # Test 6: core losses are constant
    r4 = model.predict({"load_fraction": np.array([0.1, 0.5, 1.0])})
    p_cores = r4["p_core_w"]
    assert_true(np.all(np.abs(np.diff(p_cores)) < 1e-6), "Core losses are constant")

    # Test 7: copper losses scale as load^2
    r5 = model.predict({"load_fraction": np.array([0.5, 1.0])})
    ratio = float(r5["p_copper_w"][1]) / float(r5["p_copper_w"][0])
    assert_true(abs(ratio - 4.0) < 1e-6, "Copper losses scale as PLR^2 (ratio 4 at 2x load)")

    # Test 8: power balance P_in = P_out + P_loss
    plr_arr = np.linspace(0.1, 1.2, 50)
    r6 = model.predict({"load_fraction": plr_arr})
    diff = np.abs(r6["p_in_w"] - r6["p_out_w"] - r6["p_losses_w"])
    assert_true(np.all(diff < 1e-6), "Power balance: P_in = P_out + P_loss")

    # Test 9: eta at full load (~0.99 for P_core=0.2%, P_cu=1.0%, pf=1)
    r7 = model.predict({"load_fraction": 1.0, "power_factor": 1.0})
    eta_full = float(r7["efficiency"])
    # eta = 1/(1+0.002+0.010) = 1/1.012 = 0.98814
    assert_true(abs(eta_full - 1.0/1.012) < 0.0005,
                f"Full-load eta = {eta_full:.5f}, expected ~{1/1.012:.5f}")

    # Test 10: V_out with v_in
    r8 = model.predict({"load_fraction": 1.0, "v_in": 11000.0})
    assert_true("v_out" in r8, "v_out returned when v_in provided")
    N = model._model.N
    assert_true(abs(float(r8["v_out"]) - N * 11000.0) < 1e-6,
                f"V_out = N*V_in = {N*11000:.1f} V")

    # Test 11: losses at zero load = core losses only
    r9 = model.predict({"load_fraction": 0.0})
    assert_true(abs(float(r9["p_losses_w"]) - model._model.P_core) < 1e-6,
                "Losses at PLR=0 equal core losses only")

    # Test 12: benchmark
    plr_bench = np.random.uniform(0.05, 1.5, 1000)
    t0 = time.perf_counter()
    model.predict({"load_fraction": plr_bench})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")

    print(f"\nAll tests passed.")


if __name__ == "__main__":
    main()
