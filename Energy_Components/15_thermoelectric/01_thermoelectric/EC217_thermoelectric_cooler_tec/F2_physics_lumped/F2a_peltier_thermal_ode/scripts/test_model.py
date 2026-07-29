"""
EC217 -- Thermoelectric Cooler (TEC) -- F2a Peltier + Thermal ODE -- Test suite.
Custom assert harness (NO pytest). Run: python3 scripts/test_model.py
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import TEC_F2a
from predict import ComponentModel

PASS, FAIL = "✓", "✗"


def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def make():
    cm = ComponentModel()
    return cm._model, cm


def test_energy_conservation():
    """W_input must equal Q_hot - Q_cold exactly (1st law)."""
    print("\n[Test 1] Energy conservation: W_in = Q_hot - Q_cold")
    m, _ = make()
    for (T_c, T_h, I) in [(280.0, 300.0, 2.0), (270.0, 310.0, 5.0), (290.0, 305.0, 1.0)]:
        jh = m.junction_heat(T_c, T_h, I)
        bal = jh["Q_hot_W"] - jh["Q_cold_W"]
        err = abs(jh["W_input_W"] - bal)
        assert_true(err < 1e-9 * max(abs(bal), 1.0) + 1e-9,
                    f"T_c={T_c},T_h={T_h},I={I}: W={jh['W_input_W']:.4f}, "
                    f"Qh-Qc={bal:.4f}, err={err:.2e}")


def test_cop_positive_and_below_carnot():
    """When cooling, 0 < COP < Carnot COP."""
    print("\n[Test 2] 0 < COP < Carnot")
    m, _ = make()
    T_c, T_h = 290.0, 300.0   # 10 K lift, well within dT_max
    # Operate at half the max-cooling current: positive Q_cold with good COP.
    I = 0.5 * m.current_for_max_cooling(T_c, T_h)
    jh = m.junction_heat(T_c, T_h, I)
    carnot = m.carnot_cop(T_c, T_h)
    assert_true(jh["Q_cold_W"] > 0, f"Q_cold={jh['Q_cold_W']:.3f} W > 0 (cooling)")
    assert_true(jh["COP"] > 0, f"COP={jh['COP']:.3f} > 0")
    assert_true(jh["COP"] < carnot, f"COP={jh['COP']:.3f} < Carnot={carnot:.3f}")


def test_cop_max_below_carnot():
    """Theoretical max COP (Ioffe) must stay below the Carnot COP."""
    print("\n[Test 3] COP_max (Ioffe) < Carnot")
    m, _ = make()
    for (T_c, T_h) in [(290.0, 300.0), (280.0, 310.0), (270.0, 320.0)]:
        cop_max = m.cop_max_theoretical(T_c, T_h)
        carnot = m.carnot_cop(T_c, T_h)
        assert_true(0.0 <= cop_max < carnot,
                    f"T_c={T_c},T_h={T_h}: COP_max={cop_max:.3f} < Carnot={carnot:.3f}")


def test_qc_max_at_optimal_current():
    """Q_cold is maximised at I = alpha_m*T_c/R (Goldsmid max-cooling current)."""
    print("\n[Test 4] Q_cold peaks at I_qmax = alpha_m*T_c/R")
    m, _ = make()
    T_c, T_h = 280.0, 305.0
    I_opt = m.current_for_max_cooling(T_c, T_h)
    Q_opt = m.junction_heat(T_c, T_h, I_opt)["Q_cold_W"]
    for frac in [0.6, 0.8, 0.9, 1.1, 1.2, 1.4]:
        Q = m.junction_heat(T_c, T_h, I_opt * frac)["Q_cold_W"]
        assert_true(Q <= Q_opt + 1e-9,
                    f"I={I_opt*frac:.3f}A: Q_cold={Q:.3f} <= Q_max={Q_opt:.3f}")


def test_zt_figure_of_merit():
    """ZT = alpha^2*sigma*T/k must be positive and O(1) for Bi2Te3 near 300 K."""
    print("\n[Test 5] ZT material figure of merit ~ O(1)")
    m, _ = make()
    zt = float(m.zt_local(300.0))
    assert_true(zt > 0, f"ZT={zt:.3f} > 0")
    assert_true(0.3 < zt < 2.0, f"ZT={zt:.3f} in Bi2Te3 range (0.3-2.0)")


def test_pulldown_cools_below_ambient():
    """Transient: with a sensible current the cold plate drops below ambient."""
    print("\n[Test 6] ODE pull-down cools cold plate below ambient")
    m, _ = make()
    out = m.simulate(I=3.0, duration_s=120.0)
    assert_true(out["success"], "solve_ivp integration succeeded")
    T_amb = m.T_amb
    assert_true(out["T_cold"][-1] < T_amb - 1.0,
                f"T_cold_ss={out['T_cold'][-1]:.2f} K < ambient {T_amb:.1f} K")
    assert_true(out["T_cold"][-1] < out["T_cold"][0],
                f"cold plate cooled: {out['T_cold'][0]:.1f} -> {out['T_cold'][-1]:.1f} K")
    assert_true(out["T_hot"][-1] >= out["T_hot"][0] - 1e-6,
                f"hot plate did not cool: {out['T_hot'][0]:.1f} -> {out['T_hot'][-1]:.1f} K")


def test_steady_state_zero_derivative():
    """At the end of a long run dT/dt ~ 0 (steady state reached)."""
    print("\n[Test 7] Steady state: dT/dt -> 0 at end of long run")
    m, _ = make()
    out = m.simulate(I=3.0, duration_s=400.0)
    T_c, T_h = out["T_cold"][-1], out["T_hot"][-1]
    dy = m._rhs(0.0, [T_c, T_h], 3.0, 0.0)
    assert_true(abs(dy[0]) < 1e-2, f"dT_cold/dt={dy[0]:.2e} ~ 0")
    assert_true(abs(dy[1]) < 1e-2, f"dT_hot/dt={dy[1]:.2e} ~ 0")


def test_higher_current_more_joule_heating():
    """Beyond I_qmax, raising current reduces cooling (Joule dominates)."""
    print("\n[Test 8] Excess current -> less cooling (Joule self-heating)")
    m, _ = make()
    out_lo = m.simulate(I=3.0, duration_s=200.0)
    out_hi = m.simulate(I=10.0, duration_s=200.0)
    assert_true(out_hi["T_cold"][-1] > out_lo["T_cold"][-1],
                f"I=10A T_cold={out_hi['T_cold'][-1]:.1f} > "
                f"I=3A T_cold={out_lo['T_cold'][-1]:.1f} K (over-driven)")


def test_zero_current_no_cooling():
    """At I=0 only Fourier conduction acts; cold plate cannot go below ambient."""
    print("\n[Test 9] Zero current -> no active cooling")
    m, _ = make()
    out = m.simulate(I=0.0, duration_s=120.0)
    assert_true(out["T_cold"][-1] >= m.T_amb - 0.5,
                f"I=0: T_cold_ss={out['T_cold'][-1]:.2f} K not below ambient")
    jh = m.junction_heat(290.0, 300.0, 0.0)
    assert_true(jh["Q_cold_W"] < 0,
                f"I=0: Q_cold={jh['Q_cold_W']:.3f} W <= 0 (only leak-back)")


def test_load_reduces_pulldown():
    """A larger cold-side heat load yields a warmer steady cold plate."""
    print("\n[Test 10] Higher Q_load -> warmer cold plate")
    m, _ = make()
    out0 = m.simulate(I=3.0, Q_load=0.0, duration_s=200.0)
    out5 = m.simulate(I=3.0, Q_load=5.0, duration_s=200.0)
    assert_true(out5["T_cold"][-1] > out0["T_cold"][-1],
                f"Q=5W T_cold={out5['T_cold'][-1]:.1f} > "
                f"Q=0W T_cold={out0['T_cold'][-1]:.1f} K")


def test_predict_interface():
    """ComponentModel predict()/get_info() return the expected shape."""
    print("\n[Test 11] predict() interface + metadata")
    _, cm = make()
    r = cm.predict({"current_A": 3.0, "duration_s": 60.0})
    for k in ["t", "T_cold", "T_hot", "Q_cold_W", "Q_hot_W", "W_input_W",
              "COP", "V_module_V", "steady_state", "success"]:
        assert_true(k in r, f"Key '{k}' present")
    for k in ["T_cold_ss_K", "T_hot_ss_K", "dT_ss_K", "COP", "carnot_COP", "ZT_avg"]:
        assert_true(k in r["steady_state"], f"steady_state['{k}'] present")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC217", "component_id = EC217")
    assert_true("F2a" in info["fidelity"], "fidelity contains F2a")


def test_benchmark():
    """Benchmark: a representative 120 s simulate() call must be fast."""
    print("\n[Test 12] Benchmark")
    m, _ = make()
    m.simulate(I=3.0, duration_s=120.0)  # warm-up
    t0 = time.perf_counter()
    n = 10
    for _ in range(n):
        m.simulate(I=3.0, duration_s=120.0)
    elapsed = (time.perf_counter() - t0) / n
    print(f"  Single 120 s simulate in {elapsed*1000:.2f} ms")
    assert_true(elapsed < 5.0, "< 5 s per representative simulate()")


if __name__ == "__main__":
    tests = [
        test_energy_conservation,
        test_cop_positive_and_below_carnot,
        test_cop_max_below_carnot,
        test_qc_max_at_optimal_current,
        test_zt_figure_of_merit,
        test_pulldown_cools_below_ambient,
        test_steady_state_zero_derivative,
        test_higher_current_more_joule_heating,
        test_zero_current_no_cooling,
        test_load_reduces_pulldown,
        test_predict_interface,
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
    print(f"\n{'=' * 60}")
    print(f"EC217 TEC F2a -- {p} passed, {f} failed")
    print(f"{'=' * 60}")
    sys.exit(0 if f == 0 else 1)
