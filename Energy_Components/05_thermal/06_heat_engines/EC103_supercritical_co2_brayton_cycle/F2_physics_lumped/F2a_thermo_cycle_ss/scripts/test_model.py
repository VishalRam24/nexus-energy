"""
EC103 -- Supercritical CO2 Brayton Cycle -- F2a Physics-Lumped
Test suite: cycle physics sanity, near-critical advantage, conservation,
Carnot bound, transient ODE, predict() interface, benchmark.
Custom harness (NO pytest). Run: python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SCO2BraytonF2a
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
def test_efficiency_band():
    print("\n[Test 1] Thermal efficiency in sCO2 target band 0.40-0.50")
    m, _ = make_model()
    c = m.cycle()
    eta = c["eta_thermal"]
    assert_true(0.40 <= eta <= 0.52, f"eta_thermal={eta:.4f} in ~[0.40, 0.50]")


def test_below_carnot():
    print("\n[Test 2] Efficiency strictly below Carnot bound (2nd law)")
    m, _ = make_model()
    for T4 in [823.15, 973.15, 1073.15]:
        c = m.cycle(T_turb_in=T4)
        assert_true(c["eta_thermal"] < c["eta_carnot"],
                    f"eta={c['eta_thermal']:.4f} < carnot={c['eta_carnot']:.4f} @T4={T4:.0f}K")


def test_near_critical_low_Z():
    print("\n[Test 3] Near-critical compressor inlet is liquid-like (low Z, high rho)")
    m, _ = make_model()
    Z = m.compressibility(305.15, 7.7e6)
    rho = m.density(305.15, 7.7e6)
    assert_true(0.15 < Z < 0.45, f"Z(305K,7.7MPa)={Z:.3f} liquid-like (SW ~0.28)")
    assert_true(rho > 350.0, f"rho={rho:.1f} kg/m3 >> ideal-gas value")
    Z_hot = m.compressibility(973.15, 25e6)
    assert_true(Z_hot > 0.9, f"Z(973K,25MPa)={Z_hot:.3f} -> near ideal at hot end")


def test_low_main_compressor_work():
    print("\n[Test 4] Main-compressor work is a small fraction of turbine work")
    m, _ = make_model()
    c = m.cycle()
    ratio = c["w_mc"] / c["w_turb"]
    assert_true(ratio < 0.25, f"w_mc/w_turb={ratio:.3f} < 0.25 (key sCO2 advantage)")
    assert_true(c["back_work_ratio"] < 0.55, f"overall BWR={c['back_work_ratio']:.3f} < 0.55")


def test_cp_near_critical_enhancement():
    print("\n[Test 5] Real-gas cp peaks near critical, decays at high T")
    m, _ = make_model()
    cp_near = m.cp_real(310.0, 8.0e6)
    cp_hot = m.cp_real(973.15, 25e6)
    assert_true(cp_near > 1.3 * m.cp_ideal, f"cp_near={cp_near:.0f} >> cp_ideal={m.cp_ideal:.0f}")
    assert_true(abs(cp_hot - m.cp_ideal) / m.cp_ideal < 0.1,
                f"cp_hot={cp_hot:.0f} ~ cp_ideal={m.cp_ideal:.0f}")


def test_energy_conservation():
    print("\n[Test 6] Energy balance: Q_in = P_net + Q_rej")
    m, _ = make_model()
    c = m.cycle()
    resid = abs(c["Q_in_W"] - (c["P_net_W"] + c["Q_rej_W"]))
    assert_true(resid < 1e-3 * c["Q_in_W"], f"|Q_in-(P_net+Q_rej)|={resid:.2e} W ~ 0")
    assert_true(c["w_net"] > 0 and c["q_in"] > 0, "w_net>0 and q_in>0")


def test_efficiency_monotone_in_Tin():
    print("\n[Test 7] Efficiency increases with turbine inlet temperature")
    m, _ = make_model()
    etas = [m.cycle(T_turb_in=T)["eta_thermal"] for T in [773.15, 873.15, 973.15, 1073.15]]
    for a, b in zip(etas[:-1], etas[1:]):
        assert_true(b >= a - 1e-9, f"eta non-decreasing: {a:.4f} -> {b:.4f}")


def test_recompression_helps():
    print("\n[Test 8] Recompression (at optimum split) beats no recompression")
    m, _ = make_model()
    eta_simple = m.cycle(f_rc=0.0)["eta_thermal"]
    # optimum split balances the recuperator capacity-rate mismatch
    f_opt = m.cycle()["f_balance"]
    eta_recomp = m.cycle(f_rc=f_opt)["eta_thermal"]
    assert_true(eta_recomp > eta_simple,
                f"recompression eta={eta_recomp:.4f} > simple eta={eta_simple:.4f} (f_opt={f_opt:.2f})")


def test_transient_ode_steady_state():
    print("\n[Test 9] Lumped thermal ODE warms hot section toward steady state")
    m, _ = make_model()
    r = m.simulate(T_metal0=400.0, dt=5.0, duration_s=2000.0)
    T = r["T_turbine_inlet"]
    assert_true(T[-1] > T[0], f"heats up: {T[0]:.1f} -> {T[-1]:.1f} K")
    dT = abs(T[-1] - T[-2])
    assert_true(dT < 1.0, f"near steady state: |dT|={dT:.4f} K between last steps")
    assert_true(T[-1] < 1100.0, f"T_final={T[-1]:.1f} K bounded/physical")


def test_efficiency_in_unit_interval():
    print("\n[Test 10] Transient efficiency series in (0,1) and rises with T")
    m, _ = make_model()
    r = m.simulate(T_metal0=400.0, dt=10.0, duration_s=1500.0)
    eta = r["efficiency"]
    assert_true(np.all((eta > 0) & (eta < 1.0)), "all efficiencies in (0,1)")
    assert_true(eta[-1] > eta[0], f"efficiency rises: {eta[0]:.4f} -> {eta[-1]:.4f}")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface keys")
    _, cm = make_model()
    r = cm.predict({"transient": True, "duration_s": 300.0, "dt": 10.0})
    for key in ["eta_thermal", "eta_carnot", "w_net_J_per_kg", "back_work_ratio",
                "P_net_W", "Q_in_W", "Q_rej_W", "states_K", "transient"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["transient"]["t"]) == len(r["transient"]["efficiency"]),
                "transient arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: transient sim < 5 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(T_metal0=400.0, dt=5.0, duration_s=600.0)
    elapsed = time.perf_counter() - t0
    print(f"  600s transient in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_efficiency_band,
        test_below_carnot,
        test_near_critical_low_Z,
        test_low_main_compressor_work,
        test_cp_near_critical_enhancement,
        test_energy_conservation,
        test_efficiency_monotone_in_Tin,
        test_recompression_helps,
        test_transient_ode_steady_state,
        test_efficiency_in_unit_interval,
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
    print(f"EC103 sCO2 Brayton F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
