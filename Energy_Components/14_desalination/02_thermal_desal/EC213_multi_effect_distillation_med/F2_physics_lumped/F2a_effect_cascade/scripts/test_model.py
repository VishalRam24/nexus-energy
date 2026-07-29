"""
EC213 -- Multi-Effect Distillation (MED) -- F2a Physics-Lumped Effect Cascade
Test suite: mass/energy conservation, GOR~N, temperature cascade, lower energy
than MSF, transient ODE convergence, predict() interface, benchmark timing.

Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MED_F2a
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
def test_temperature_cascade():
    print("\n[Test 1] Temperatures form a strictly decreasing cascade")
    m, _ = make_model()
    T = m.effect_temperatures()
    assert_true(np.all(np.diff(T) < 0), "T_effect strictly decreasing across effects")
    assert_true(abs(T[0] - m.T_top) < 1e-9, f"T[0]={T[0]:.2f} == T_top={m.T_top}")
    assert_true(abs(T[-1] - m.T_last) < 1e-9, f"T[-1]={T[-1]:.2f} == T_last={m.T_last}")
    dT = m.temperature_drop_per_effect()
    assert_true(dT > 0, f"per-effect dT={dT:.3f} C > 0")


def test_mass_conservation():
    print("\n[Test 2] Mass balance per effect: feed = distillate + brine")
    m, _ = make_model()
    ss = m.steady_state()
    feed_per_effect = m.M_feed / m.N
    for i in range(m.N):
        lhs = ss["D_effect"][i] + ss["B_effect"][i]
        assert_true(abs(lhs - feed_per_effect) < 1e-6,
                    f"effect {i}: D+B={lhs:.4f} == feed_share={feed_per_effect:.4f}")


def test_salt_conservation():
    print("\n[Test 3] Salt balance: feed salt = brine salt (distillate is pure)")
    m, _ = make_model()
    ss = m.steady_state()
    feed_per_effect = m.M_feed / m.N
    for i in range(m.N):
        salt_in = feed_per_effect * m.X_feed
        salt_out = ss["B_effect"][i] * ss["X_brine"][i]
        assert_true(abs(salt_in - salt_out) / salt_in < 1e-6,
                    f"effect {i}: salt_in={salt_in:.1f} == salt_out={salt_out:.1f}")
    assert_true(np.all(ss["X_brine"] >= m.X_feed - 1e-6),
                "brine more saline than feed in every effect")
    assert_true(np.all(ss["X_brine"] <= m.X_brine_max + 1.0),
                "brine salinity within scaling limit")


def test_energy_conservation_reuse():
    print("\n[Test 4] Energy balance + latent-heat reuse: Q_i ~ D_{i-1}*hfg_{i-1}")
    m, _ = make_model()
    ss = m.steady_state()
    for i in range(1, m.N):
        hfg_prev = float(m.hfg(ss["T_effect"][i - 1]))
        expected = ss["D_effect"][i - 1] * hfg_prev
        assert_true(abs(ss["Q_effect"][i] - expected) / expected < 1e-6,
                    f"effect {i}: Q={ss['Q_effect'][i]:.1f} == D_prev*hfg={expected:.1f}")
    # each effect's vapor must equal its heat duty / its own latent heat
    for i in range(m.N):
        hfg_i = float(m.hfg(ss["T_effect"][i]))
        assert_true(abs(ss["D_effect"][i] - ss["Q_effect"][i] / hfg_i) < 1e-6,
                    f"effect {i}: D == Q/hfg (energy closes)")


def test_gor_near_n_effects():
    print("\n[Test 5] GOR ~ N_effects and scales linearly with N")
    m, _ = make_model()
    for N in [4, 8, 12, 16]:
        ss = m.steady_state(N=N)
        gor = ss["GOR"]
        # GOR should be a high fraction of N (latent-heat reused ~N times)
        assert_true(0.85 * N <= gor <= 1.05 * N,
                    f"N={N}: GOR={gor:.2f} within [0.85N, 1.05N]")
    # monotone in N
    gors = [m.steady_state(N=N)["GOR"] for N in [4, 8, 12, 16]]
    assert_true(all(np.diff(gors) > 0), f"GOR increases with N: {[round(g,1) for g in gors]}")


def test_distillate_production():
    print("\n[Test 6] Positive distillate production and recovery in (0,1)")
    m, _ = make_model()
    ss = m.steady_state()
    assert_true(ss["distillate_total_kg_s"] > 0, "total distillate > 0")
    assert_true(ss["distillate_total_m3_h"] > 0, "distillate m3/h > 0")
    assert_true(0 < ss["recovery"] < 1.0, f"recovery={ss['recovery']:.3f} in (0,1)")
    assert_true(np.all(ss["D_effect"] > 0), "every effect produces vapor")


def test_lower_energy_than_msf():
    print("\n[Test 7] Specific thermal energy lower than MSF (~250-290 kJ/kg)")
    m, _ = make_model()
    ss = m.steady_state()
    sec = ss["specific_thermal_kJ_kg"]
    MSF_TYPICAL = 250.0  # MSF GOR~8 => ~290 kJ/kg; MED at GOR~12 must beat it
    assert_true(sec < MSF_TYPICAL,
                f"MED SEC_thermal={sec:.1f} kJ/kg < MSF ~{MSF_TYPICAL} kJ/kg")
    assert_true(sec > 100.0, f"SEC_thermal={sec:.1f} kJ/kg physically reasonable")


def test_top_temp_below_msf():
    print("\n[Test 8] Top brine temperature lower than MSF (limits scaling)")
    m, _ = make_model()
    assert_true(m.T_top <= 80.0, f"T_top={m.T_top} C <= 80 C (MSF runs ~90-110 C)")
    assert_true(m.T_top > m.T_last, "T_top > T_last")


def test_transient_relaxes_to_steady():
    print("\n[Test 9] Transient ODE relaxes to the steady-state cascade")
    m, _ = make_model()
    ss = m.steady_state()
    r = m.simulate(T0_C=m.T_last, dt=20.0, duration_s=7200.0)
    T_end = r["T_effect"][:, -1]
    err = np.max(np.abs(T_end - ss["T_effect"]))
    assert_true(err < 0.5, f"max |T_end - T_steady| = {err:.3f} C < 0.5")
    assert_true(abs(r["GOR"][-1] - ss["GOR"]) < 0.5,
                f"transient GOR end={r['GOR'][-1]:.2f} ~ steady {ss['GOR']:.2f}")


def test_transient_heats_up():
    print("\n[Test 10] Cold start: effects heat up toward the cascade")
    m, _ = make_model()
    r = m.simulate(T0_C=m.T_last, dt=20.0, duration_s=3600.0)
    T0col = r["T_effect"][:, 0]
    Tend = r["T_effect"][:, -1]
    # first effect must rise from cold (T_last) toward T_top
    assert_true(Tend[0] > T0col[0] + 1.0, f"effect 1 heats: {T0col[0]:.1f}->{Tend[0]:.1f} C")
    assert_true(np.all(Tend <= m.T_top + 0.5), "no effect exceeds T_top")
    assert_true(np.all(Tend >= m.T_last - 0.5), "no effect below T_last")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + get_info()")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC213", "component_id == EC213")
    r = cm.predict({"N_effects": 10, "transient": True, "duration_s": 1800.0, "dt": 30.0})
    for key in ["T_effect", "D_effect", "B_effect", "X_brine", "Q_effect",
                "distillate_total_kg_s", "GOR", "recovery", "transient"]:
        assert_true(key in r, f"predict output has '{key}'")
    assert_true(len(r["T_effect"]) == 10, "N_effects override honored (10 effects)")


def test_benchmark():
    print("\n[Test 12] Benchmark: 2-hour transient sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(T0_C=40.0, dt=10.0, duration_s=7200.0)
    elapsed = time.perf_counter() - t0
    print(f"  7200 s transient in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_temperature_cascade,
        test_mass_conservation,
        test_salt_conservation,
        test_energy_conservation_reuse,
        test_gor_near_n_effects,
        test_distillate_production,
        test_lower_energy_than_msf,
        test_top_temp_below_msf,
        test_transient_relaxes_to_steady,
        test_transient_heats_up,
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
    print(f"EC213 MED F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
