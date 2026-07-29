"""
EC151 -- Dry Steam Geothermal Plant -- F2a Physics-Lumped
Test suite: thermodynamic property sanity, energy conservation, efficiency
bounds (eta_util < Carnot), NCG effect, transient ODE behaviour, interface.
Run: python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import DrySteamGeothermalF2a
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
def test_steam_properties():
    print("\n[Test 1] Steam saturation properties match IAPWS steam tables")
    m, _ = make_model()
    # Tsat at atmospheric (0.101325 MPa) ~ 100 degC
    Ts = m.Tsat(0.101325) - 273.15
    assert_true(abs(Ts - 99.97) < 0.5, f"Tsat(1 atm)={Ts:.2f} C ~ 100 C")
    # h_g at 0.1 MPa ~ 2675 kJ/kg
    hg = m.hg_sat(0.1)
    assert_true(abs(hg - 2675.0) < 8, f"h_g(0.1MPa)={hg:.1f} ~ 2675 kJ/kg")
    # h_f at 0.5 MPa ~ 640 kJ/kg
    hf = m.hf_sat(0.5)
    assert_true(abs(hf - 640.1) < 8, f"h_f(0.5MPa)={hf:.1f} ~ 640 kJ/kg")
    # h_fg positive and decreasing with pressure
    assert_true(m.hfg_sat(0.1) > m.hfg_sat(1.0) > 0,
                f"h_fg decreases with P: {m.hfg_sat(0.1):.0f} > {m.hfg_sat(1.0):.0f}")


def test_tsat_monotone():
    print("\n[Test 2] Tsat increases monotonically with pressure")
    m, _ = make_model()
    P = np.linspace(0.01, 1.5, 40)
    T = m.Tsat(P)
    assert_true(np.all(np.diff(T) > 0), "Tsat strictly increasing in P")


def test_isentropic_drop_positive():
    print("\n[Test 3] Isentropic enthalpy drop > 0 across the turbine")
    m, _ = make_model()
    es = m.expansion_endstate(0.8, 0.012)
    assert_true(es["dh_isentropic"] > 0, f"dh_s={es['dh_isentropic']:.1f} > 0")
    assert_true(es["dh_actual"] < es["dh_isentropic"],
                f"dh_act={es['dh_actual']:.1f} < dh_s (irreversibility)")
    assert_true(0.0 <= es["x2s"] <= 1.0, f"exit quality x2s={es['x2s']:.3f} in [0,1]")


def test_efficiency_below_carnot():
    print("\n[Test 4] Utilization efficiency < Carnot (2nd law)")
    m, _ = make_model()
    for P_wh in [0.4, 0.6, 0.8, 1.0, 1.2]:
        r = m.power(50.0, P_wh, 0.012, 0.0)
        assert_true(0 < r["eta_utilization"] < r["eta_carnot"],
                    f"P={P_wh}: eta_util={r['eta_utilization']:.4f} < "
                    f"Carnot={r['eta_carnot']:.4f}")


def test_realistic_utilization():
    print("\n[Test 5] Utilization efficiency in realistic dry-steam band 10-25%")
    m, _ = make_model()
    r = m.power(50.0, 0.8, 0.012, 0.0)
    eta = r["eta_utilization"]
    assert_true(0.10 < eta < 0.25, f"eta_util={eta:.4f} in [0.10, 0.25] (DiPippo 2015)")
    assert_true(0 < r["eta_2nd_law"] < 1.0, f"eta_2nd={r['eta_2nd_law']:.4f} in (0,1)")


def test_energy_conservation():
    print("\n[Test 6] Energy balance: P_gross = m_dot*w_specific")
    m, _ = make_model()
    r = m.power(50.0, 0.8, 0.012, 0.0, x_ncg=0.0)
    P_expected = 50.0 * r["w_specific_kJ_kg"]   # kW (kJ/s), x_ncg=0
    assert_true(abs(r["P_gross_kW"] - P_expected) < 1e-6,
                f"P_gross={r['P_gross_kW']:.2f} == m_dot*w={P_expected:.2f}")
    # Enthalpy bookkeeping: h_in - h2_actual == dh_actual
    es = m.expansion_endstate(0.8, 0.012)
    assert_true(abs((es["h_in"] - es["h2_actual"]) - es["dh_actual"]) < 1e-6,
                "h_in - h2_act == dh_actual (1st law on turbine)")


def test_ncg_penalty():
    print("\n[Test 7] Non-condensable gas reduces net power monotonically")
    m, _ = make_model()
    P_prev = m.power(50.0, 0.8, 0.012, 0.0, x_ncg=0.0)["P_net_kW"]
    for x in [0.02, 0.05, 0.10]:
        r = m.power(50.0, 0.8, 0.012, 0.0, x_ncg=x)
        assert_true(r["P_net_kW"] < P_prev, f"x_ncg={x}: P_net={r['P_net_kW']:.0f} < prev {P_prev:.0f}")
        assert_true(r["P_parasitic_kW"] > 0, f"x_ncg={x}: parasitic>0")
        P_prev = r["P_net_kW"]


def test_power_scales_with_flow():
    print("\n[Test 8] Net power scales linearly with steam mass flow")
    m, _ = make_model()
    P1 = m.power(25.0, 0.8, 0.012, 0.0)["P_net_kW"]
    P2 = m.power(50.0, 0.8, 0.012, 0.0)["P_net_kW"]
    assert_true(abs(P2 - 2.0 * P1) < 1e-6, f"P(50)={P2:.0f} == 2*P(25)={2*P1:.0f}")


def test_transient_steady_hold():
    print("\n[Test 9] Transient ODE holds steady at fixed wellhead pressure")
    m, _ = make_model()
    r = m.simulate(0.8, m_dot0=m.m_dot_design, dt=2.0, duration_s=120.0)
    dm = abs(r["m_dot"][-1] - r["m_dot"][0])
    assert_true(dm < 1e-3, f"m_dot stays at design (dm={dm:.2e})")
    assert_true(np.all(r["P_net_kW"] > 0), "P_net > 0 throughout")


def test_transient_step_response():
    print("\n[Test 10] Wellhead pressure step -> first-order flow rise to new SS")
    m, _ = make_model()

    def P_step(t):
        return 0.6 if t < 60.0 else 1.0

    r = m.simulate(P_step, m_dot0=m._mdot_target(0.6), dt=2.0, duration_s=300.0)
    # before step: near m_target(0.6)
    i_before = np.argmin(np.abs(r["t"] - 58.0))
    # well after step: near m_target(1.0)
    m_final = r["m_dot"][-1]
    assert_true(r["m_dot"][i_before] < m_final,
                f"flow rises after pressure step: {r['m_dot'][i_before]:.1f} -> {m_final:.1f}")
    assert_true(abs(m_final - m._mdot_target(1.0)) < 0.5,
                f"settles near new target {m._mdot_target(1.0):.1f} (got {m_final:.1f})")
    # tau check: one time constant after step (t=60+tau) reaches ~63% of gap
    m0 = r["m_dot"][i_before]
    mtgt = m._mdot_target(1.0)
    i_tau = np.argmin(np.abs(r["t"] - (60.0 + m.tau_wh)))
    frac = (r["m_dot"][i_tau] - m0) / (mtgt - m0)
    assert_true(0.55 < frac < 0.72, f"~63% of step reached at 1 tau (frac={frac:.3f})")


def test_superheat_raises_work():
    print("\n[Test 11] Superheated inlet raises specific work vs dry saturated")
    m, _ = make_model()
    w_dry = m.specific_work(0.8, 0.012, T_superheat=0.0)
    w_sup = m.specific_work(0.8, 0.012, T_superheat=40.0)
    assert_true(w_sup > w_dry, f"superheat: w={w_sup:.1f} > dry w={w_dry:.1f} kJ/kg")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() / predict_steady() interface")
    _, cm = make_model()
    r = cm.predict({"P_wh_MPa": 0.8, "dt": 5.0, "duration_s": 30.0})
    for key in ["t", "m_dot", "T_casing", "P_net_kW", "P_gross_kW",
                "eta_utilization", "eta_carnot"]:
        assert_true(key in r, f"Key '{key}' in transient output")
    assert_true(len(r["t"]) == len(r["P_net_kW"]), "Arrays same length")
    ss = cm.predict_steady({})
    assert_true("P_net_kW" in ss and ss["P_net_kW"] > 0, "predict_steady returns P_net>0")


def test_benchmark():
    print("\n[Test 13] Benchmark: 300 s transient sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.8, dt=1.0, duration_s=300.0)
    elapsed = time.perf_counter() - t0
    print(f"  300 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_steam_properties,
        test_tsat_monotone,
        test_isentropic_drop_positive,
        test_efficiency_below_carnot,
        test_realistic_utilization,
        test_energy_conservation,
        test_ncg_penalty,
        test_power_scales_with_flow,
        test_transient_steady_hold,
        test_transient_step_response,
        test_superheat_raises_work,
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
    print(f"EC151 Dry Steam Geothermal F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
