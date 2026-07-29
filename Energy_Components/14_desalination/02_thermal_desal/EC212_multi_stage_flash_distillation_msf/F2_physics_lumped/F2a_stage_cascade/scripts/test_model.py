"""
EC212 -- Multi-Stage Flash Distillation (MSF) -- F2a Stage-Cascade
Test suite: physics sanity (conservation, monotonicity, known limits),
edge cases, predict() interface, benchmark timing.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MSF_F2a
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
def test_gor_realistic():
    print("\n[Test 1] GOR in realistic MSF band (8-12) at design point")
    m, _ = make_model()
    ss = m.steady_state()
    assert_true(8.0 <= ss["GOR"] <= 12.0, f"GOR={ss['GOR']:.2f} in [8,12]")
    assert_true(abs(ss["PR"] - ss["GOR"]) < 1.5, f"PR ({ss['PR']:.2f}) ~ GOR ({ss['GOR']:.2f})")


def test_gor_rises_with_tbt():
    print("\n[Test 2] GOR increases with top brine temperature")
    m, _ = make_model()
    g_prev = m.steady_state(T_top=90)["GOR"]
    for Tt in [100, 110, 120]:
        g = m.steady_state(T_top=Tt)["GOR"]
        assert_true(g > g_prev, f"GOR(TBT={Tt})={g:.2f} > {g_prev:.2f}")
        g_prev = g


def test_gor_rises_with_stages():
    print("\n[Test 3] GOR increases with number of stages (heat recovery)")
    import json
    _, cm = make_model()
    base = cm._raw
    g_prev = 0.0
    for N in [12, 18, 24, 32]:
        p = json.loads(json.dumps(base))
        p["unit"]["N_stages"]["value"] = N
        ss = MSF_F2a(p).steady_state()
        assert_true(ss["GOR"] > g_prev, f"GOR(N={N})={ss['GOR']:.2f} > {g_prev:.2f}")
        g_prev = ss["GOR"]


def test_temperature_cascade_monotone():
    print("\n[Test 4] Stage temperature cascade strictly decreasing")
    m, _ = make_model()
    Ts = m.stage_temperatures()
    dT_stage = (m.T_top - m.T_last) / m.N
    assert_true(np.all(np.diff(Ts) < 0), "Cascade strictly decreasing")
    assert_true(Ts[0] < m.T_top and abs(Ts[0] - (m.T_top - dT_stage)) < 1e-9,
                f"Top stage = TBT - dT_stage ({Ts[0]:.2f} < TBT {m.T_top})")
    assert_true(abs(Ts[-1] - m.T_last) < 1e-9, f"Last stage = T_last ({Ts[-1]:.1f})")


def test_mass_conservation():
    print("\n[Test 5] Mass balance: feed = distillate + blowdown")
    m, _ = make_model()
    ss = m.steady_state()
    total = ss["D_total"] + ss["brine_out_kg_s"]
    assert_true(abs(total - m.M_brine) < 1e-6, f"{total:.3f} == feed {m.M_brine}")
    # salinity must concentrate (water removed, salt conserved)
    assert_true(ss["salinity_out_ppm"] > m.S_feed,
                f"blowdown {ss['salinity_out_ppm']:.0f} > feed {m.S_feed:.0f} ppm")
    # salt mass conservation: S_feed*Mb = S_out*Mb_out
    salt_in = m.S_feed * m.M_brine
    salt_out = ss["salinity_out_ppm"] * ss["brine_out_kg_s"]
    assert_true(abs(salt_in - salt_out) / salt_in < 1e-6, "Salt mass conserved")


def test_energy_conservation():
    print("\n[Test 6] Energy balance: Q_heater = M_steam * hfg_steam")
    m, _ = make_model()
    ss = m.steady_state()
    q = ss["M_steam"] * m.hfg_steam
    assert_true(abs(q - ss["Q_heater"]) < 1e-6, f"{q:.1f} == Q_heater {ss['Q_heater']:.1f}")
    # distillate latent heat <= heater duty + recovered (GOR > 1 means heat is recycled)
    assert_true(ss["GOR"] > 1.0, f"GOR={ss['GOR']:.2f} > 1 (heat recovery active)")


def test_distillate_positive_and_recovery():
    print("\n[Test 7] Distillate positive in every stage; recovery physical")
    m, _ = make_model()
    ss = m.steady_state()
    assert_true(np.all(ss["distillate_stage"] >= 0), "All stage distillate >= 0")
    assert_true(ss["D_total"] > 0, f"Total distillate {ss['D_total']:.2f} > 0")
    assert_true(0.0 < ss["recovery"] < 0.5,
                f"recovery {ss['recovery']:.3f} in (0, 0.5) (single-pass MSF)")


def test_nea_reduces_distillate():
    print("\n[Test 8] Non-equilibrium allowance reduces distillate vs equilibrium")
    import json
    _, cm = make_model()
    base = cm._raw
    p_eq = json.loads(json.dumps(base))
    p_eq["unit"]["NEA_coeff"]["value"] = 0.0   # equilibrium flashing
    D_eq = MSF_F2a(p_eq).steady_state()["D_total"]
    D_nea = MSF_F2a(base).steady_state()["D_total"]
    assert_true(D_nea < D_eq, f"D(NEA)={D_nea:.2f} < D(equil)={D_eq:.2f}")
    m, _ = make_model()
    assert_true(m.nea(m.steady_state()["dT_stage"]) >= 0, "NEA >= 0")


def test_bpe_physical():
    print("\n[Test 9] Boiling-point elevation positive and increases with salinity")
    m, _ = make_model()
    bpe_low = m.boiling_point_elevation(80.0, 35000.0)
    bpe_high = m.boiling_point_elevation(80.0, 70000.0)
    assert_true(bpe_low > 0, f"BPE>0 ({bpe_low:.3f})")
    assert_true(bpe_high > bpe_low, f"BPE rises with salinity ({bpe_high:.3f} > {bpe_low:.3f})")
    assert_true(bpe_high < 3.0, f"BPE physically small ({bpe_high:.3f} < 3 degC)")


def test_transient_converges_to_cascade():
    print("\n[Test 10] Lumped transient ODE relaxes to steady cascade from cold start")
    m, _ = make_model()
    r = m.simulate(duration_s=2500.0, n_eval=120)
    # starts cold (seawater), heats up
    assert_true(r["T_stages"][0].mean() < r["T_stages"][-1].mean(),
                "Stages heat up from cold start")
    err = np.abs(r["T_stages"][-1] - r["T_target"]).max()
    assert_true(err < 0.05, f"Converges to design cascade (max err {err:.4f} degC)")
    # transient cascade monotone at steady state
    assert_true(np.all(np.diff(r["T_stages"][-1]) < 1e-6), "Final cascade decreasing")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"duration_s": 800.0, "n_eval": 50})
    for key in ["t", "T_stages", "T_target", "GOR", "D_total", "M_steam",
                "recovery", "flash_range", "NEA"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(r["T_stages"].shape == (50, cm._model.N), "T_stages shape (n_eval, N)")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC212", "component_id EC212")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1500s transient sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(duration_s=1500.0, n_eval=200)
    elapsed = time.perf_counter() - t0
    print(f"  1500s transient in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_gor_realistic,
        test_gor_rises_with_tbt,
        test_gor_rises_with_stages,
        test_temperature_cascade_monotone,
        test_mass_conservation,
        test_energy_conservation,
        test_distillate_positive_and_recovery,
        test_nea_reduces_distillate,
        test_bpe_physical,
        test_transient_converges_to_cascade,
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
    print(f"EC212 MSF F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
