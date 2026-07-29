"""EC174 -- Instrument Transformer (CT/PT) -- F1a -- Test Suite"""
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
    print("EC174 Instrument Transformer (CT/PT) F1a -- Test Suite")
    model = ComponentModel()

    # Test 1: keys
    r = model.predict({"i_primary": 1000.0, "v_primary": 11000.0, "current_fraction": 1.0})
    assert_true(all(k in r for k in ["ct_i_secondary", "pt_v_secondary", "ct_burden_va",
                                      "pt_burden_va", "ratio_error_limit_pct",
                                      "phase_error_limit_min", "within_accuracy_class"]),
                "predict returns required keys")

    # Test 2: get_info
    info = model.get_info()
    assert_true(info["ec_id"] == "EC174", "ec_id == EC174")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 3: CT ratio I_sec = I_pri / N
    m = model._model
    I_pri = 1000.0
    r = model.predict({"i_primary": I_pri, "v_primary": 11000.0, "current_fraction": 1.0})
    expected = I_pri / m.ct_N
    assert_true(abs(float(r["ct_i_secondary"]) - expected) < 1e-9,
                f"CT: I_sec = I_pri/N = {expected:.2f} A")

    # Test 4: PT ratio V_sec = V_pri * N_pt
    V_pri = 11000.0
    expected_v = V_pri * m.pt_N
    assert_true(abs(float(r["pt_v_secondary"]) - expected_v) < 1e-9,
                f"PT: V_sec = V_pri*N_pt = {expected_v:.1f} V")

    # Test 5: CT linearity -- I_sec proportional to I_pri
    i_arr = np.array([200.0, 500.0, 1000.0, 1500.0])
    r2 = model.predict({"i_primary": i_arr, "v_primary": 11000.0})
    expected_i = i_arr / m.ct_N
    np.testing.assert_allclose(r2["ct_i_secondary"], expected_i, rtol=1e-9)
    assert_true(True, "CT secondary current is linear with primary")

    # Test 6: CT burden scales as I^2
    r3 = model.predict({"i_primary": np.array([500.0, 1000.0]), "v_primary": 11000.0})
    ratio = float(r3["ct_burden_va"][1]) / float(r3["ct_burden_va"][0])
    assert_true(abs(ratio - 4.0) < 1e-6, "CT burden scales as I^2 (ratio 4 at 2x current)")

    # Test 7: within accuracy class at rated
    r4 = model.predict({"i_primary": 1000.0, "v_primary": 11000.0, "current_fraction": 1.0})
    assert_true(bool(r4["within_accuracy_class"]), "Within accuracy class at rated current")

    # Test 8: outside accuracy class at overload (current_fraction > 1.2)
    r5 = model.predict({"i_primary": 2000.0, "v_primary": 11000.0, "current_fraction": 2.0})
    assert_true(not bool(r5["within_accuracy_class"]),
                "Outside accuracy class at 200% rated current")

    # Test 9: ratio error limit is 0.1%
    assert_true(abs(float(r4["ratio_error_limit_pct"]) - 0.1) < 1e-9,
                "Ratio error limit = 0.1%")

    # Test 10: phase error limit is 5 arcmin
    assert_true(abs(float(r4["phase_error_limit_min"]) - 5.0) < 1e-9,
                "Phase error limit = 5 arcmin")

    # Test 11: PT burden is fixed (voltage source)
    v_arr = np.array([9000.0, 11000.0, 13000.0])
    r6 = model.predict({"i_primary": 1000.0, "v_primary": v_arr})
    assert_true(np.all(np.abs(np.diff(r6["pt_burden_va"])) < 1e-6), "PT burden is fixed (voltage source)")

    # Test 12: benchmark
    i_bench = np.random.uniform(10.0, 2000.0, 1000)
    t0 = time.perf_counter()
    model.predict({"i_primary": i_bench, "v_primary": 11000.0})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")

    print(f"\nAll tests passed.")


if __name__ == "__main__":
    main()
