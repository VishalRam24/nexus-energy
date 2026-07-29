"""
EC149 -- Biodiesel Transesterification -- F2a Physics-Lumped Kinetics
Test suite: mass conservation, kinetic/equilibrium sanity, methanol-excess
effect, temperature (Arrhenius) effect, energy balance, predict() interface.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import TransesterificationF2a
from predict import ComponentModel

PASS = "✓"
FAIL = "✗"


def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def make_model():
    cm = ComponentModel()
    return cm._model, cm


# ---------------------------------------------------------------------------
def test_arrhenius_increases_with_T():
    print("\n[Test 1] Arrhenius rate constants increase with temperature")
    m, _ = make_model()
    k_low = m.rate_constants(303.15)
    k_high = m.rate_constants(343.15)
    for i in range(6):
        assert_true(k_high[i] > k_low[i],
                    f"k{i+1}: {k_high[i]:.4g} (70C) > {k_low[i]:.4g} (30C)")


def test_kref_recovered():
    print("\n[Test 2] Rate constants match reported values at T_ref (50C)")
    m, _ = make_model()
    k = m.rate_constants(m.T_ref)
    for i in range(6):
        rel = abs(k[i] - m.k_ref[i]) / m.k_ref[i]
        assert_true(rel < 1e-6, f"k{i+1}({m.T_ref:.1f}K)={k[i]:.4f} ~= k_ref={m.k_ref[i]:.4f}")


def test_mass_conservation():
    print("\n[Test 3] Mass conservation (backbone, ester, methanol balances)")
    m, _ = make_model()
    r = m.simulate(TG0=1.0, methanol_ratio=6.0, T0=333.15, duration_min=120.0)
    res = m.mass_balance_residuals(r)
    assert_true(res["backbone"] < 1e-4, f"glyceride backbone drift={res['backbone']:.2e}")
    assert_true(res["ester_balance"] < 1e-4, f"ester balance drift={res['ester_balance']:.2e}")
    assert_true(res["methanol_balance"] < 1e-4, f"methanol balance drift={res['methanol_balance']:.2e}")


def test_nonnegative_concentrations():
    print("\n[Test 4] All concentrations remain non-negative")
    m, _ = make_model()
    r = m.simulate(TG0=1.0, methanol_ratio=6.0, T0=333.15, duration_min=90.0)
    for sp in ["TG", "DG", "MG", "FAME", "glycerol", "methanol"]:
        assert_true(np.all(r[sp] >= -1e-6), f"{sp} >= 0 throughout (min={r[sp].min():.2e})")


def test_fame_monotone_early():
    print("\n[Test 5] FAME and glycerol rise, TG falls over the run")
    m, _ = make_model()
    r = m.simulate(TG0=1.0, methanol_ratio=6.0, T0=333.15, duration_min=90.0)
    assert_true(r["FAME"][-1] > r["FAME"][0], "FAME final > initial")
    assert_true(r["glycerol"][-1] > r["glycerol"][0], "glycerol final > initial")
    assert_true(r["TG"][-1] < r["TG"][0], "TG final < initial")
    assert_true(r["conversion_final"] > 0.5, f"meaningful conversion={r['conversion_final']:.2f}")


def test_yield_approaches_equilibrium():
    print("\n[Test 6] FAME yield plateaus near equilibrium (slope flattens)")
    m, _ = make_model()
    r = m.simulate(TG0=1.0, methanol_ratio=6.0, T0=333.15, duration_min=180.0, n_points=200)
    y = r["FAME_yield"]
    # early slope vs late slope
    early = y[10] - y[2]
    late = y[-1] - y[-9]
    assert_true(late < early * 0.5, f"late slope {late:.4f} << early slope {early:.4f}")
    assert_true(0.5 < r["FAME_yield_final"] <= 1.0, f"final yield={r['FAME_yield_final']:.3f}")


def test_methanol_excess_drives_conversion():
    print("\n[Test 7] Higher MeOH:oil ratio -> higher conversion")
    m, _ = make_model()
    r3 = m.simulate(TG0=1.0, methanol_ratio=3.0, T0=333.15, duration_min=120.0)
    r6 = m.simulate(TG0=1.0, methanol_ratio=6.0, T0=333.15, duration_min=120.0)
    r9 = m.simulate(TG0=1.0, methanol_ratio=9.0, T0=333.15, duration_min=120.0)
    assert_true(r6["conversion_final"] > r3["conversion_final"],
                f"6:1 ({r6['conversion_final']:.3f}) > 3:1 ({r3['conversion_final']:.3f})")
    assert_true(r9["conversion_final"] >= r6["conversion_final"] - 1e-6,
                f"9:1 ({r9['conversion_final']:.3f}) >= 6:1 ({r6['conversion_final']:.3f})")


def test_temperature_speeds_reaction():
    print("\n[Test 8] Higher temperature -> faster conversion at fixed time")
    m, _ = make_model()
    r_cold = m.simulate(TG0=1.0, methanol_ratio=6.0, T0=313.15, duration_min=20.0, isothermal=True)
    r_hot = m.simulate(TG0=1.0, methanol_ratio=6.0, T0=338.15, duration_min=20.0, isothermal=True)
    assert_true(r_hot["conversion_final"] > r_cold["conversion_final"],
                f"hot ({r_hot['conversion_final']:.3f}) > cold ({r_cold['conversion_final']:.3f})")


def test_energy_balance_response():
    print("\n[Test 9] Energy balance: T tracks jacket, stays bounded")
    m, _ = make_model()
    # start below jacket -> reactor should warm toward jacket set point
    r = m.simulate(TG0=1.0, methanol_ratio=6.0, T0=313.15, duration_min=120.0)
    assert_true(r["T_final"] > 313.15, f"T warmed: {r['T_final']:.2f} K > 313.15 K")
    assert_true(r["T_final"] < m.T_jacket + 5.0,
                f"T bounded near jacket: {r['T_final']:.2f} K < {m.T_jacket+5:.2f} K")
    assert_true(np.all(r["temperature"] < 400.0), "no thermal runaway (<400 K)")


def test_catalyst_factor_effect():
    print("\n[Test 10] More catalyst -> faster conversion")
    m, _ = make_model()
    r_lo = m.simulate(TG0=1.0, methanol_ratio=6.0, T0=333.15, catalyst_factor=0.3,
                      duration_min=30.0)
    r_hi = m.simulate(TG0=1.0, methanol_ratio=6.0, T0=333.15, catalyst_factor=3.0,
                      duration_min=30.0)
    assert_true(r_hi["conversion_final"] > r_lo["conversion_final"],
                f"3x cat ({r_hi['conversion_final']:.3f}) > 0.3x cat ({r_lo['conversion_final']:.3f})")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"methanol_ratio": 6.0, "T0_K": 333.15, "duration_min": 30.0})
    for key in ["t", "TG", "DG", "MG", "FAME", "glycerol", "methanol",
                "temperature", "conversion", "FAME_yield"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["FAME"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC149", "get_info component_id == EC149")


def test_benchmark():
    print("\n[Test 12] Benchmark: 90 min batch sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(TG0=1.0, methanol_ratio=6.0, T0=333.15, duration_min=90.0, n_points=120)
    elapsed = time.perf_counter() - t0
    print(f"  90 min batch simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_arrhenius_increases_with_T,
        test_kref_recovered,
        test_mass_conservation,
        test_nonnegative_concentrations,
        test_fame_monotone_early,
        test_yield_approaches_equilibrium,
        test_methanol_excess_drives_conversion,
        test_temperature_speeds_reaction,
        test_energy_balance_response,
        test_catalyst_factor_effect,
        test_predict_interface,
        test_benchmark,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except (AssertionError, Exception) as e:
            failed += 1
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"EC149 Transesterification F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
