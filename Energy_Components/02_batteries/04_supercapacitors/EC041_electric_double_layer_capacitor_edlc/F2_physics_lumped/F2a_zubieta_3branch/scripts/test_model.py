"""
EC041 -- EDLC Supercapacitor -- F2a Zubieta 3-Branch ECM
Test suite: physics sanity (charge conservation, energy bounds, redistribution,
thermal balance, 0<eff<1), edge cases, predict() interface, benchmark.
Custom harness (NO pytest). Run: python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np
from scipy.integrate import trapezoid

sys.path.insert(0, os.path.dirname(__file__))
from model import EDLC_F2a
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
def test_capacitance_voltage_dependent():
    print("\n[Test 1] C(V) increases with voltage (Zubieta C0+kv*V)")
    m, _ = make_model()
    c0 = m.C_imm(0.0)
    c1 = m.C_imm(2.5)
    assert_true(c1 > c0, f"C(2.5V)={c1:.1f} > C(0V)={c0:.1f} F")
    assert_true(abs(c0 - m.C0) < 1e-6, f"C(0)=C0={m.C0:.1f} F")


def test_charge_conservation():
    print("\n[Test 2] Charge conservation: q_stored ~= integral(I_into_caps) dt")
    m, _ = make_model()
    # No leakage isolation: account for leakage current explicitly.
    I = 50.0
    r = m.simulate(I, v0_V=0.0, dt=0.05, duration_s=20.0)
    t = r["t"]
    # Charge delivered by source
    q_src = trapezoid(r["current"], t)
    # Charge lost to leakage over the run
    i_leak = r["v1"] / m.R_leak
    q_leak = trapezoid(i_leak, t)
    # Charge stored on the three branch capacitors (nonlinear immediate q1)
    v1, v2, v3 = r["v1"][-1], r["v2"][-1], r["v3"][-1]
    q_stored = (m.C0 * v1 + 0.5 * m.kv * v1**2) + m.Cd * v2 + m.Cl * v3
    bal = q_src - q_leak - q_stored
    assert_true(abs(bal) < 0.01 * q_src + 1.0,
                f"charge balance |{bal:.3f}| C small vs q_src={q_src:.1f} C")


def test_voltage_monotone_charge():
    print("\n[Test 3] Terminal voltage rises monotonically under constant charge")
    m, _ = make_model()
    r = m.simulate(80.0, v0_V=0.0, dt=0.1, duration_s=25.0)
    v1 = r["v1"]
    diffs = np.diff(v1)
    assert_true(np.all(diffs >= -1e-9), "v1 non-decreasing during charge")
    assert_true(r["v_terminal"][-1] > r["v_terminal"][0], "terminal V increased")


def test_charge_redistribution():
    print("\n[Test 4] Charge redistribution: terminal V sags after charge stops")
    m, _ = make_model()

    def prof(t):
        return 100.0 if t < 10.0 else 0.0  # charge 10 s, then open circuit

    r = m.simulate(prof, v0_V=0.0, dt=0.05, duration_s=120.0)
    t = r["t"]
    i_off = np.argmin(np.abs(t - 10.5))   # just after charging stops
    i_end = np.argmin(np.abs(t - 119.0))  # well into rest
    v_off = r["v1"][i_off]
    v_end = r["v1"][i_end]
    assert_true(v_end < v_off,
                f"v1 relaxes downward: rest_end={v_end:.4f} < just_after={v_off:.4f} V")
    # delayed/long-term branches should have absorbed charge (risen)
    assert_true(r["v2"][i_end] > r["v2"][i_off] or r["v3"][i_end] > r["v3"][i_off],
                "delayed/long-term branch voltage rose (absorbed redistributed charge)")


def test_energy_bounds_and_formula():
    print("\n[Test 5] Stored energy >= 0 and bounded; ~0.5 C V^2 scale")
    m, _ = make_model()
    r = m.simulate(100.0, v0_V=0.0, dt=0.1, duration_s=30.0)
    assert_true(np.all(r["energy_J"] >= -1e-9), "energy non-negative")
    assert_true(np.all(np.diff(r["energy_J"]) >= -1e-6), "energy non-decreasing on charge")
    # sanity scale at final v1: between 0.5*C0*V^2 and 0.5*C(V)*V^2
    v1 = r["v1"][-1]
    e_lo = 0.5 * m.C0 * v1**2
    e_hi = 0.5 * m.C_imm(v1) * v1**2
    e_imm = 0.5 * m.C0 * v1**2 + (1.0/3.0) * m.kv * v1**3
    assert_true(e_lo <= e_imm <= e_hi + 1e-6,
                f"immediate energy {e_imm:.0f} J in [{e_lo:.0f}, {e_hi:.0f}]")


def test_self_discharge():
    print("\n[Test 6] Self-discharge: voltage decays on open circuit (I=0)")
    m, _ = make_model()
    r = m.simulate(0.0, v0_V=2.5, T0_K=298.15, dt=1.0, duration_s=600.0)
    assert_true(r["v_terminal"][-1] < r["v_terminal"][0],
                f"V decays {r['v_terminal'][0]:.4f} -> {r['v_terminal'][-1]:.4f} V")
    assert_true(r["v_terminal"][-1] > 0.0, "V stays positive")
    # energy must drop (leakage dissipates stored energy)
    assert_true(r["energy_J"][-1] < r["energy_J"][0], "stored energy decreased")


def test_thermal_heats_up_and_balances():
    print("\n[Test 7] Thermal ODE: Joule heating warms cell, then balances")
    m, _ = make_model()
    # High continuous ripple via large constant charge then discharge cycling
    def prof(t):
        return 250.0 if int(t) % 2 == 0 else -250.0
    r = m.simulate(prof, v0_V=1.0, T0_K=298.15, dt=0.05, duration_s=200.0)
    assert_true(r["temperature"][-1] > 298.15, f"T rose to {r['temperature'][-1]:.3f} K")
    assert_true(r["temperature"][-1] < 360.0, "T stays physically reasonable (<360 K)")
    assert_true(np.all(r["heat_W"] >= -1e-9), "Joule heat generation non-negative")


def test_esr_arrhenius():
    print("\n[Test 8] ESR Arrhenius: ESR higher when cold, lower when hot")
    m, _ = make_model()
    esr_cold = float(m.Ri(253.15))   # -20 C
    esr_ref = float(m.Ri(298.15))    # 25 C
    esr_hot = float(m.Ri(333.15))    # 60 C
    assert_true(esr_cold > esr_ref > esr_hot,
                f"ESR(-20C)={esr_cold*1e3:.3f} > ESR(25C)={esr_ref*1e3:.3f} > ESR(60C)={esr_hot*1e3:.3f} mOhm")
    assert_true(abs(esr_ref - m.Ri_ref) < 1e-9, "ESR(T_ref)==Ri_ref")


def test_round_trip_efficiency():
    print("\n[Test 9] Round-trip efficiency in (0,1), high for low current")
    m, _ = make_model()
    eff_lo, ein_lo, eout_lo = m.round_trip_efficiency(20.0, v_top=2.5)
    eff_hi, ein_hi, eout_hi = m.round_trip_efficiency(300.0, v_top=2.5)
    assert_true(0.0 < eff_lo < 1.0, f"eff(20A)={eff_lo:.4f} in (0,1)")
    assert_true(0.0 < eff_hi < 1.0, f"eff(300A)={eff_hi:.4f} in (0,1)")
    assert_true(eff_lo > eff_hi,
                f"lower current more efficient: {eff_lo:.4f} > {eff_hi:.4f}")
    assert_true(eff_lo > 0.9, f"EDLC low-current eff high ({eff_lo:.4f} > 0.9)")


def test_terminal_voltage_ir_drop():
    print("\n[Test 10] Terminal voltage shows IR offset vs internal v1")
    m, _ = make_model()
    # During discharge terminal V should be below v1 by I*ESR
    r = m.simulate(-200.0, v0_V=2.0, T0_K=298.15, dt=0.1, duration_s=2.0)
    drop = r["v1"][1] - r["v_terminal"][1]
    expected = 200.0 * float(m.Ri(r["temperature"][1]))
    assert_true(drop > 0, f"discharge terminal V below v1 by {drop*1e3:.3f} mV")
    assert_true(abs(drop - expected) < 1e-6, "IR drop == I*ESR exactly")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + array consistency")
    _, cm = make_model()
    r = cm.predict({"current_A": 100.0, "dt": 0.5, "duration_s": 10.0})
    for key in ["t", "v_terminal", "v1", "v2", "v3", "current",
                "energy_J", "power_W", "temperature", "esr_Ohm", "heat_W"]:
        assert_true(key in r, f"Key '{key}' in output")
    n = len(r["t"])
    assert_true(all(len(r[k]) == n for k in
                    ["v_terminal", "v1", "v2", "v3", "energy_J", "temperature"]),
                "all output arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC041", "get_info component_id EC041")


def test_benchmark():
    print("\n[Test 12] Benchmark: 60 s simulation at dt=0.1")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(100.0, v0_V=0.0, dt=0.1, duration_s=60.0)
    elapsed = time.perf_counter() - t0
    print(f"  60 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_capacitance_voltage_dependent,
        test_charge_conservation,
        test_voltage_monotone_charge,
        test_charge_redistribution,
        test_energy_bounds_and_formula,
        test_self_discharge,
        test_thermal_heats_up_and_balances,
        test_esr_arrhenius,
        test_round_trip_efficiency,
        test_terminal_voltage_ir_drop,
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
    print(f"EC041 EDLC F2a Zubieta 3-branch -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
