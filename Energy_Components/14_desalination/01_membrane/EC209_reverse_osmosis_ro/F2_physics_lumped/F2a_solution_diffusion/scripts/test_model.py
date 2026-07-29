"""
EC209 -- Reverse Osmosis (RO) -- F2a Solution-Diffusion -- Test suite.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import RO_SolutionDiffusion_F2a
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


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_typical_swro_rejection():
    """Salt rejection > 99% for typical seawater RO at 60 bar."""
    print("\n[Test 1] Typical SWRO: rejection > 99%")
    m, _ = make()
    r = m.solve_vessel(Cf_gL=35.0, P_bar=60.0, Qf_m3h=8.0)
    rej_pct = r["rejection"] * 100
    assert_true(rej_pct > 99.0, f"Rejection = {rej_pct:.2f}% > 99%")
    assert_true(r["Cp_gL"] < 0.5, f"Permeate TDS = {r['Cp_gL']:.3f} g/L < 0.5 g/L")


def test_sec_range():
    """SEC should be 2-6 kWh/m3 for seawater RO."""
    print("\n[Test 2] SEC in range 2-6 kWh/m3")
    m, _ = make()
    r = m.solve_vessel(Cf_gL=35.0, P_bar=60.0, Qf_m3h=8.0)
    sec = r["SEC_kwhm3"]
    assert_true(2.0 <= sec <= 6.0, f"SEC = {sec:.2f} kWh/m3 in [2, 6]")


def test_higher_pressure_higher_recovery():
    """Higher feed pressure should give higher recovery."""
    print("\n[Test 3] Higher pressure -> higher recovery")
    m, _ = make()
    r_low = m.solve_vessel(Cf_gL=35.0, P_bar=50.0, Qf_m3h=8.0)
    r_high = m.solve_vessel(Cf_gL=35.0, P_bar=70.0, Qf_m3h=8.0)
    assert_true(
        r_high["recovery"] > r_low["recovery"],
        f"R(70bar)={r_high['recovery']:.3f} > R(50bar)={r_low['recovery']:.3f}"
    )


def test_higher_concentration_lower_flux():
    """Higher feed concentration should reduce permeate flux (higher osmotic pressure)."""
    print("\n[Test 4] Higher feed concentration -> lower flux")
    m, _ = make()
    r_low = m.solve_vessel(Cf_gL=15.0, P_bar=60.0, Qf_m3h=8.0)
    r_high = m.solve_vessel(Cf_gL=40.0, P_bar=60.0, Qf_m3h=8.0)
    assert_true(
        r_low["Qp_m3h"] > r_high["Qp_m3h"],
        f"Qp(15g/L)={r_low['Qp_m3h']:.2f} > Qp(40g/L)={r_high['Qp_m3h']:.2f} m3/h"
    )


def test_mass_balance():
    """Salt mass balance: salt_in = salt_permeate + salt_concentrate."""
    print("\n[Test 5] Salt mass balance")
    m, _ = make()
    Cf = 35.0
    Qf = 8.0
    r = m.solve_vessel(Cf_gL=Cf, P_bar=60.0, Qf_m3h=Qf)

    salt_in = Cf * Qf  # g/h
    salt_perm = r["Cp_gL"] * r["Qp_m3h"]
    salt_conc = r["Cc_gL"] * r["Qc_m3h"]
    salt_out = salt_perm + salt_conc

    rel_err = abs(salt_in - salt_out) / salt_in
    assert_true(rel_err < 0.01, f"Salt balance error = {rel_err:.4%} < 1%")


def test_recovery_below_100():
    """Recovery must be < 100% (physical constraint)."""
    print("\n[Test 6] Recovery < 100%")
    m, _ = make()
    r = m.solve_vessel(Cf_gL=35.0, P_bar=80.0, Qf_m3h=8.0)
    assert_true(r["recovery"] < 1.0, f"Recovery = {r['recovery']:.1%} < 100%")
    assert_true(r["recovery"] > 0.0, f"Recovery = {r['recovery']:.1%} > 0%")


def test_single_element():
    """Single element should produce less permeate than full vessel."""
    print("\n[Test 7] Single element vs full vessel")
    m, _ = make()
    r1 = m.solve_vessel(Cf_gL=35.0, P_bar=60.0, Qf_m3h=8.0, N_elements=1)
    r7 = m.solve_vessel(Cf_gL=35.0, P_bar=60.0, Qf_m3h=8.0, N_elements=7)
    assert_true(
        r7["Qp_m3h"] > r1["Qp_m3h"],
        f"Qp(7elem)={r7['Qp_m3h']:.2f} > Qp(1elem)={r1['Qp_m3h']:.2f} m3/h"
    )


def test_element_profiles_monotonic():
    """In series elements: flux should decrease, feed concentration should increase."""
    print("\n[Test 8] Element profiles: flux decreasing, Cf increasing")
    m, _ = make()
    r = m.solve_vessel(Cf_gL=35.0, P_bar=60.0, Qf_m3h=8.0)
    Jw = r["profiles"]["Jw_LMH"]
    Cf_arr = r["profiles"]["Cf_gL"]
    # Flux should generally decrease along vessel
    assert_true(Jw[0] > Jw[-1], f"Jw[0]={Jw[0]:.2f} > Jw[-1]={Jw[-1]:.2f} LMH")
    # Feed concentration should increase
    assert_true(Cf_arr[-1] > Cf_arr[0], f"Cf[-1]={Cf_arr[-1]:.1f} > Cf[0]={Cf_arr[0]:.1f} g/L")


def test_predict_interface():
    """ComponentModel predict() returns expected keys."""
    print("\n[Test 9] ComponentModel predict() interface")
    _, cm = make()
    r = cm.predict({
        "feed_concentration_gL": 35.0,
        "feed_pressure_bar": 60.0,
        "feed_flow_m3h": 8.0,
    })
    expected_keys = ["Qp_m3h", "Cp_gL", "recovery", "rejection", "SEC_kwhm3",
                     "Cc_gL", "profiles"]
    for k in expected_keys:
        assert_true(k in r, f"Key '{k}' present")

    info = cm.get_info()
    assert_true(info["component_id"] == "EC209", f"component_id = {info['component_id']}")
    assert_true("F2a" in info["fidelity"], f"fidelity contains F2a")


def test_brackish_water():
    """Brackish water (5 g/L) at 15 bar should have very high rejection, low SEC."""
    print("\n[Test 10] Brackish water: low salinity, low pressure")
    m, _ = make()
    r = m.solve_vessel(Cf_gL=5.0, P_bar=15.0, Qf_m3h=8.0)
    assert_true(r["rejection"] > 0.98, f"Rejection = {r['rejection']:.4f} > 0.98")
    assert_true(r["SEC_kwhm3"] < 2.0, f"SEC = {r['SEC_kwhm3']:.2f} < 2.0 kWh/m3")


def test_benchmark():
    """Benchmark: single vessel solve should be fast."""
    print("\n[Test 11] Benchmark: vessel solve time")
    m, _ = make()
    t0 = time.perf_counter()
    N_runs = 100
    for _ in range(N_runs):
        m.solve_vessel(Cf_gL=35.0, P_bar=60.0, Qf_m3h=8.0)
    elapsed = (time.perf_counter() - t0) / N_runs
    print(f"  Single vessel solve in {elapsed*1000:.2f} ms")
    assert_true(elapsed < 1.0, "< 1 s per vessel solve")


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_typical_swro_rejection,
        test_sec_range,
        test_higher_pressure_higher_recovery,
        test_higher_concentration_lower_flux,
        test_mass_balance,
        test_recovery_below_100,
        test_single_element,
        test_element_profiles_monotonic,
        test_predict_interface,
        test_brackish_water,
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
    print(f"EC209 RO F2a Solution-Diffusion -- {p} passed, {f} failed")
    print(f"{'='*60}")
    sys.exit(0 if f == 0 else 1)
