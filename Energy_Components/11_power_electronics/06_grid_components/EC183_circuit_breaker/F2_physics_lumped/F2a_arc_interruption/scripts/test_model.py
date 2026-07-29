"""
EC183 -- Circuit Breaker -- F2a Cassie-Mayr Arc Interruption
Test suite: arc physics sanity, current-zero interruption, TRV / dielectric
reignition, breaking-capacity limit, energy accounting, predict() interface.
Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import copy
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import CircuitBreakerArc_F2a
from predict import ComponentModel

PASS = "✓"
FAIL = "✗"

_PARAMS = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def make_model():
    cm = ComponentModel()
    return cm._model, cm


def load_raw():
    with open(_PARAMS) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
def test_arc_voltage_plateau():
    print("\n[Test 1] Cassie arc-voltage plateau ~ U_c during conduction")
    m, _ = make_model()
    r = m.simulate(I_fault_kA=20.0, duration_ms=3.0, dt_us=0.05)
    t, u = r["t"], r["arc_voltage"]
    mask = (t > 0.2e-3) & (t < 0.9e-3)        # high-current arcing phase
    umed = np.median(np.abs(u[mask]))
    assert_true(0.7 * m.U_c < umed < 1.3 * m.U_c,
                f"arc V median {umed:.0f} V near U_c={m.U_c:.0f} V")


def test_current_zero_clears():
    print("\n[Test 2] Successful interruption at/after a current zero")
    m, _ = make_model()
    r = m.simulate(I_fault_kA=20.0, duration_ms=15.0, dt_us=0.1)
    assert_true(r["n_current_zeros"] >= 1, "at least one current zero found")
    assert_true(r["interruption_success"], "20 kA fault interrupted (within rating)")
    assert_true(r["conductance"][-1] < 1e-3,
                f"post-arc conductance collapsed: g_final={r['conductance'][-1]:.2e} S")


def test_arc_extinguishes_only_if_trv_withstood():
    print("\n[Test 3] Arc reignites if TRV exceeds dielectric withstand")
    raw = load_raw()
    raw["unit"]["u_dielectric_kV"]["value"] = 2.0     # weak gap -> breakdown
    m = CircuitBreakerArc_F2a(raw)
    r = m.simulate(I_fault_kA=20.0, duration_ms=15.0, dt_us=0.1)
    assert_true(not r["trv_withstood"], "TRV NOT withstood with 2 kV gap")
    assert_true(not r["interruption_success"], "interruption fails on reignition")


def test_breaking_capacity_limit():
    print("\n[Test 4] Breaking-capacity limit enforced")
    m, _ = make_model()
    r_ok = m.simulate(I_fault_kA=25.0, duration_ms=15.0, dt_us=0.1)   # = rating
    r_no = m.simulate(I_fault_kA=40.0, duration_ms=15.0, dt_us=0.1)   # > rating
    assert_true(r_ok["within_capacity"], "25 kA within 25 kA breaking capacity")
    assert_true(r_ok["interruption_success"], "rated fault interrupts")
    assert_true(not r_no["within_capacity"], "40 kA exceeds breaking capacity")
    assert_true(not r_no["interruption_success"], "over-capacity fault fails")


def test_arc_energy_positive_and_monotone():
    print("\n[Test 5] Arc energy >= 0 and increases with fault current")
    m, _ = make_model()
    E = []
    for If in [10.0, 20.0, 30.0]:
        r = m.simulate(I_fault_kA=If, duration_ms=12.0, dt_us=0.1)
        E.append(r["arc_energy_total_J"])
        assert_true(r["arc_energy_total_J"] > 0, f"E_arc({If}kA)>0")
    assert_true(E[0] < E[1] < E[2],
                f"E_arc monotone in I_fault: {[round(x/1e3,1) for x in E]} kJ")


def test_arc_energy_integral_consistency():
    print("\n[Test 6] Reported E_arc matches trapezoidal integral of u*i")
    m, _ = make_model()
    r = m.simulate(I_fault_kA=20.0, duration_ms=8.0, dt_us=0.05)
    p = r["arc_voltage"] * r["current"]
    _trap = getattr(np, "trapezoid", np.trapz)
    E_trap = _trap(p, r["t"])
    E_rep = r["arc_energy_total_J"]
    rel = abs(E_trap - E_rep) / abs(E_rep)
    assert_true(rel < 0.05, f"integral matches state E within 5% (rel={rel:.3%})")


def test_trv_bounded_by_system_voltage():
    print("\n[Test 7] TRV peak bounded near system phase voltage (not fault-scaled)")
    m, _ = make_model()
    Vph_pk = m.V_rated * np.sqrt(2.0 / 3.0)
    r1 = m.simulate(I_fault_kA=10.0, duration_ms=12.0, dt_us=0.1)
    r2 = m.simulate(I_fault_kA=25.0, duration_ms=12.0, dt_us=0.1)
    # TRV must not scale up with fault current (physical: set by grid voltage)
    bound = 2.5 * Vph_pk
    assert_true(r1["trv_peak_V"] < bound, f"TRV(10kA)={r1['trv_peak_V']/1e3:.1f} kV bounded")
    assert_true(r2["trv_peak_V"] < bound, f"TRV(25kA)={r2['trv_peak_V']/1e3:.1f} kV bounded")


def test_fault_current_matches_request():
    print("\n[Test 8] Realised fault current matches requested value")
    m, _ = make_model()
    for If in [5.0, 15.0, 25.0]:
        r = m.simulate(I_fault_kA=If, duration_ms=4.0, dt_us=0.1)
        assert_true(abs(r["I_fault_kA"] - If) / If < 0.02,
                    f"I_fault realised {r['I_fault_kA']:.2f} kA ~ {If} kA")


def test_conductance_positive():
    print("\n[Test 9] Arc conductance stays positive throughout")
    m, _ = make_model()
    r = m.simulate(I_fault_kA=20.0, duration_ms=10.0, dt_us=0.1)
    assert_true(np.all(r["conductance"] > 0), "g(t) > 0 for all t (ln-g formulation)")


def test_crossover_current_positive():
    print("\n[Test 10] Cassie/Mayr crossover current is physical")
    m, _ = make_model()
    ic = m.crossover_current()
    assert_true(ic > 0, f"crossover current {ic:.2f} A = P0/U_c > 0")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"I_fault_kA": 20.0, "duration_ms": 8.0, "dt_us": 0.1})
    for key in ["t", "current", "arc_voltage", "conductance",
                "arc_energy_total_J", "trv_peak_V", "n_current_zeros",
                "interruption_success", "interruption_time_s", "within_capacity"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["current"]) == len(r["arc_voltage"]),
                "time-series arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC183", "get_info component_id == EC183")


def test_benchmark():
    print("\n[Test 12] Benchmark: 15 ms interruption sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(I_fault_kA=25.0, duration_ms=15.0, dt_us=0.05)
    elapsed = time.perf_counter() - t0
    print(f"  15 ms simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_arc_voltage_plateau,
        test_current_zero_clears,
        test_arc_extinguishes_only_if_trv_withstood,
        test_breaking_capacity_limit,
        test_arc_energy_positive_and_monotone,
        test_arc_energy_integral_consistency,
        test_trv_bounded_by_system_voltage,
        test_fault_current_matches_request,
        test_conductance_positive,
        test_crossover_current_positive,
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
    print(f"EC183 Circuit Breaker F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
