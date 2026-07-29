"""
EC198 -- Post-Combustion Capture (Amine Scrubbing) -- F2a Equilibrium Stage -- Test suite.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import AmineCapture_F2a
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"


def assert_true(c, m):
    if c:
        print(f"  {PASS}  {m}")
    else:
        print(f"  {FAIL}  FAILED: {m}")
        raise AssertionError(m)


def make():
    cm = ComponentModel()
    return cm._model, cm


# ---------------------------------------------------------------- Tests


def test_nominal_capture_rate():
    """Capture rate should be 85-95% for a well-designed system."""
    print("\n[Test 1] Nominal capture rate 85-95%")
    m, _ = make()
    r = m.compute(y_CO2_in=0.12, L_G=3.8)
    cr = r["capture_rate"]
    assert_true(0.85 <= cr <= 0.95,
                f"capture_rate={cr*100:.1f}% in [85%, 95%] at L/G=3.8")


def test_srd_range():
    """Specific reboiler duty should be 3.0-4.5 GJ/tCO2 for 30 wt% MEA."""
    print("\n[Test 2] SRD in 3.0-4.5 GJ/tCO2")
    m, _ = make()
    r = m.compute(y_CO2_in=0.12, L_G=3.8)
    srd = r["SRD_GJ_per_tCO2"]
    assert_true(3.0 <= srd <= 4.5,
                f"SRD={srd:.2f} GJ/tCO2 in [3.0, 4.5]")


def test_higher_LG_higher_capture():
    """Higher L/G ratio should give higher capture rate (monotonic)."""
    print("\n[Test 3] Higher L/G -> higher capture rate (monotonic)")
    m, _ = make()
    crs = []
    L_Gs = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    for lg in L_Gs:
        r = m.compute(y_CO2_in=0.12, L_G=lg)
        crs.append(r["capture_rate"])
    monotonic = all(crs[i] <= crs[i + 1] + 1e-6 for i in range(len(crs) - 1))
    for lg, cr in zip(L_Gs, crs):
        print(f"    L/G={lg:.1f}  ->  CR={cr*100:.1f}%")
    assert_true(monotonic, "Capture rate is monotonically non-decreasing with L/G")


def test_mass_balance():
    """CO2 absorbed in absorber = CO2 stripped (mass balance check)."""
    print("\n[Test 4] Mass balance: CO2 absorbed = CO2 stripped")
    m, _ = make()
    r = m.compute(y_CO2_in=0.12, L_G=2.5, flue_gas_kg_s=600.0)
    # CO2 in from flue gas
    y = 0.12
    MW_flue = y * 0.04401 + (1 - y) * 0.02897
    mass_frac = y * 0.04401 / MW_flue
    CO2_in = 600.0 * mass_frac
    CO2_absorbed = CO2_in * r["capture_rate"]
    CO2_stripped = r["CO2_captured_kg_s"]
    rel_err = abs(CO2_absorbed - CO2_stripped) / (CO2_absorbed + 1e-30)
    assert_true(rel_err < 0.01,
                f"CO2_absorbed={CO2_absorbed:.2f} kg/s vs stripped={CO2_stripped:.2f} kg/s "
                f"(err={rel_err*100:.2f}%)")


def test_rich_gt_lean():
    """Rich loading must be greater than lean loading."""
    print("\n[Test 5] Rich loading > lean loading")
    m, _ = make()
    r = m.compute(y_CO2_in=0.12, L_G=2.5)
    assert_true(r["rich_loading"] > r["lean_loading"],
                f"rich={r['rich_loading']:.3f} > lean={r['lean_loading']:.3f}")


def test_predict_interface():
    """ComponentModel predict() returns all expected keys."""
    print("\n[Test 6] ComponentModel predict() interface")
    _, cm = make()
    r = cm.predict({"y_CO2_in": 0.12, "L_G": 2.5})
    expected_keys = [
        "capture_rate", "y_CO2_out", "rich_loading", "lean_loading",
        "Q_reboiler_MW", "SRD_GJ_per_tCO2", "CO2_captured_kg_s",
        "electricity_MW", "total_energy_MW",
    ]
    for k in expected_keys:
        assert_true(k in r, f"Key '{k}' present")


def test_get_info():
    """ComponentModel get_info() returns metadata."""
    print("\n[Test 7] ComponentModel get_info()")
    _, cm = make()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC198", f"component_id={info['component_id']}")
    assert_true("inputs" in info, "has inputs")
    assert_true("outputs" in info, "has outputs")


def test_more_stages_better_capture():
    """More absorber stages should improve capture rate."""
    print("\n[Test 8] More stages -> better capture")
    m, _ = make()
    r5 = m.compute(y_CO2_in=0.12, L_G=2.5, N_stages=5)
    r15 = m.compute(y_CO2_in=0.12, L_G=2.5, N_stages=15)
    assert_true(r15["capture_rate"] >= r5["capture_rate"] - 1e-6,
                f"CR(15 stages)={r15['capture_rate']*100:.1f}% >= "
                f"CR(5 stages)={r5['capture_rate']*100:.1f}%")


def test_vle_consistency():
    """VLE round-trip: loading -> P_CO2 -> loading should be consistent."""
    print("\n[Test 9] VLE round-trip consistency")
    m, _ = make()
    alpha_test = 0.35
    T = 313.15
    P = m.co2_equilibrium_pressure(alpha_test, T)
    alpha_back = m.equilibrium_loading(P, T)
    err = abs(alpha_back - alpha_test)
    assert_true(err < 0.01,
                f"alpha={alpha_test} -> P={P:.1f} Pa -> alpha_back={alpha_back:.4f} "
                f"(err={err:.4f})")


def test_benchmark():
    """Benchmark: full compute should be fast."""
    print("\n[Test 10] Benchmark: full compute")
    m, _ = make()
    t0 = time.perf_counter()
    N_runs = 100
    for _ in range(N_runs):
        m.compute(y_CO2_in=0.12, L_G=2.5)
    elapsed = (time.perf_counter() - t0) / N_runs
    print(f"  Single compute in {elapsed*1000:.2f} ms")
    assert_true(elapsed < 5.0, "< 5 s per compute")


# ---------------------------------------------------------------- Runner

if __name__ == "__main__":
    tests = [
        test_nominal_capture_rate,
        test_srd_range,
        test_higher_LG_higher_capture,
        test_mass_balance,
        test_rich_gt_lean,
        test_predict_interface,
        test_get_info,
        test_more_stages_better_capture,
        test_vle_consistency,
        test_benchmark,
    ]
    p = f = 0
    for t in tests:
        try:
            t()
            p += 1
        except Exception as e:
            f += 1
            print(f"  ERROR: {e}")
    print(f"\n{'='*60}")
    print(f"EC198 Amine Scrubbing F2a -- {p} passed, {f} failed")
    print(f"{'='*60}")
    sys.exit(0 if f == 0 else 1)
