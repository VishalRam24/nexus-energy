"""
EC039 -- Organic Flow Battery (OFB) -- F2a Physics-Lumped Stack Model
Test suite: physics sanity (Coulomb conservation, V_charge>V_disch,
efficiency bounds, thermal balance, fade monotonicity), edge cases,
predict() interface, benchmark timing. Custom harness (NO pytest).
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import OrganicFlowF2a, F_CONST
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
def test_nernst_soc_monotone():
    print("\n[Test 1] Nernst voltage rises monotonically with SOC")
    m, _ = make_model()
    socs = np.linspace(0.05, 0.95, 25)
    E_prev = m.e_nernst(socs[0], 298.15)
    for s in socs[1:]:
        E = m.e_nernst(s, 298.15)
        assert_true(E > E_prev - 1e-12, f"E({s:.2f})={E:.4f} > prev {E_prev:.4f}")
        E_prev = E
    E50 = m.e_nernst(0.5, 298.15)
    assert_true(abs(E50 - m.E0_ref) < 1e-6, f"E(SOC=0.5)={E50:.4f} == E0={m.E0_ref}")


def test_charge_above_discharge():
    print("\n[Test 2] V_charge > V_nernst > V_discharge (overpotential sign)")
    m, _ = make_model()
    for soc in [0.2, 0.5, 0.8]:
        for I in [5.0, 20.0, 40.0]:
            E = m.e_nernst(soc, 298.15)
            V_dis = m.cell_voltage(soc, I, 298.15)
            V_chg = m.cell_voltage(soc, -I, 298.15)
            assert_true(V_chg > E > V_dis,
                        f"soc={soc} I={I}: Vchg={V_chg:.4f} > E={E:.4f} > Vdis={V_dis:.4f}")


def test_overpotentials_grow_with_current():
    print("\n[Test 3] Activation+ohmic+conc overpotentials increase with |I|")
    m, _ = make_model()
    soc = 0.6
    prev = -1.0
    for I in [1.0, 5.0, 15.0, 30.0, 45.0]:
        eta = (m.activation_overpotential(I, 298.15)
               + m.ohmic_overpotential(I, 298.15)
               + m.concentration_overpotential(I, soc))
        assert_true(eta > prev, f"eta(|I|={I})={eta:.4f} > prev {prev:.4f}")
        prev = eta


def test_concentration_diverges():
    print("\n[Test 4] Concentration loss diverges as |I| -> j_L (low SOC worse)")
    m, _ = make_model()
    # j_L scales with SOC; at low SOC the same current is closer to the limit
    eta_highsoc = m.concentration_overpotential(20.0, 0.9)
    eta_lowsoc = m.concentration_overpotential(20.0, 0.15)
    assert_true(eta_lowsoc > eta_highsoc * 2,
                f"eta_conc(low SOC)={eta_lowsoc:.4f} >> eta_conc(high SOC)={eta_highsoc:.4f}")


def test_coulomb_conservation():
    print("\n[Test 5] Coulomb conservation: dSOC matches charge passed / capacity")
    m, _ = make_model()
    I = 20.0  # discharge
    dur = 600.0
    r = m._model_sim = m.simulate(I, 0.9, 298.15, 5.0, dur)
    dSOC = r["soc"][0] - r["soc"][-1]                 # SOC drop (discharge)
    # Coulomb-counting expectation: useful charge = I*dur; SOC drop = I*dur/(CE*Qcap)
    expected_dSOC = (I / m.CE) * dur / m.Q_cap_C
    rel_err = abs(dSOC - expected_dSOC) / expected_dSOC
    assert_true(rel_err < 0.02,
                f"dSOC={dSOC:.5f} vs expected {expected_dSOC:.5f} (rel err {rel_err*100:.2f}%)")


def test_coulomb_efficiency_asymmetry():
    print("\n[Test 6] CE<1 makes charge-in > discharge-out for same |SOC| swing")
    m, _ = make_model()
    assert_true(0.0 < m.CE < 1.0, f"CE={m.CE} in (0,1)")
    # charging by 0.1 SOC needs charge_in; discharging same swing yields charge_out
    # |dSOC/dt|_charge = |I|*CE/Q ; |dSOC/dt|_discharge = |I|/CE/Q  -> discharge depletes faster
    rate_chg = abs(m.dSOC_dt(0.5, -20.0))
    rate_dis = abs(m.dSOC_dt(0.5, 20.0))
    assert_true(rate_dis > rate_chg,
                f"discharge SOC-rate {rate_dis:.3e} > charge SOC-rate {rate_chg:.3e} (CE loss)")


def test_efficiency_bounds():
    print("\n[Test 7] 0 < voltage/energy efficiency < 1 (within operating envelope)")
    m, _ = make_model()
    # Physically valid operating points: high current only when SOC supports it
    # (current density must stay below the SOC-scaled transport limit j_L(SOC)).
    cases = [(0.5, 5.0), (0.5, 20.0), (0.8, 20.0), (0.8, 40.0), (0.9, 40.0)]
    for soc, I in cases:
        ve = m.voltage_efficiency(soc, I, 298.15)
        ee = m.energy_efficiency(soc, I, 298.15)
        assert_true(0.0 < ve < 1.0, f"VE(soc={soc},I={I})={ve:.4f} in (0,1)")
        assert_true(0.0 < ee < ve + 1e-12, f"EE={ee:.4f} <= VE={ve:.4f} (CE factor)")


def test_thermal_balance():
    print("\n[Test 8] Thermal ODE: heating from losses, bounded steady state")
    m, _ = make_model()
    # Modest discharge keeping SOC inside the valid window over the horizon.
    I = 10.0
    r = m.simulate(I, 0.9, 298.15, 30.0, 1800.0)
    T_final = r["temperature"][-1]
    soc_final = r["soc"][-1]
    assert_true(soc_final > 0.4, f"SOC stays in valid window: {soc_final:.3f}")
    assert_true(T_final > 298.15, f"Stack warms above ambient: T_final={T_final:.3f} K")
    assert_true(T_final < 360.0, f"Bounded by cooling: T_final={T_final:.3f} K < 360 K")
    # First-law energy balance over the run (conservation):
    #   integral(Q_gen) = m_cp*dT  +  integral(Q_cool)
    t = r["t"]
    Qg = np.array([m.heat_generation(s, I, T)
                   for s, T in zip(r["soc"], r["temperature"])])
    Qc = m.hA * (r["temperature"] - m.T_amb)
    _trap = getattr(np, "trapezoid", np.trapz)
    E_in = _trap(Qg, t)
    E_cool = _trap(Qc, t)
    E_store = m.m_cp * (r["temperature"][-1] - r["temperature"][0])
    resid = abs(E_in - E_store - E_cool) / max(E_in, 1e-6)
    assert_true(resid < 0.02,
                f"First-law balance: Q_in={E_in:.0f} = dStore({E_store:.0f}) "
                f"+ Q_rej({E_cool:.0f}) J, resid {resid*100:.2f}%")


def test_capacity_fade_monotone():
    print("\n[Test 9] Capacity fades monotonically; faster at higher T")
    m, _ = make_model()
    r = m.simulate(0.0, 0.5, 298.15, 3600.0, 30 * 24 * 3600.0)  # 30 days
    cap = r["capacity"]
    assert_true(np.all(np.diff(cap) <= 1e-12), "capacity non-increasing")
    assert_true(cap[-1] < cap[0], f"capacity fades: {cap[0]:.5f} -> {cap[-1]:.5f}")
    # Arrhenius: fade rate higher at elevated T
    k_cold = m.k_fade_ref * m._arrhenius_up_fade(298.15)
    k_hot = m.k_fade_ref * m._arrhenius_up_fade(313.15)
    assert_true(k_hot > k_cold, f"fade k(40C)={k_hot:.3e} > k(25C)={k_cold:.3e}")


def test_discharge_depletes_soc():
    print("\n[Test 10] Discharge lowers SOC, charge raises it")
    m, _ = make_model()
    r_dis = m.simulate(20.0, 0.8, 298.15, 10.0, 300.0)
    r_chg = m.simulate(-20.0, 0.5, 298.15, 10.0, 300.0)
    assert_true(r_dis["soc"][-1] < r_dis["soc"][0], "discharge: SOC falls")
    assert_true(r_chg["soc"][-1] > r_chg["soc"][0], "charge: SOC rises")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + array consistency")
    _, cm = make_model()
    r = cm.predict({"current_A": 20.0, "soc0": 0.9, "dt": 30.0, "duration_s": 300.0})
    for key in ["t", "soc", "capacity", "voltage", "cell_voltage",
                "power", "temperature", "efficiency", "overpotentials"]:
        assert_true(key in r, f"Key '{key}' in output")
    n = len(r["t"])
    assert_true(all(len(r[k]) == n for k in
                    ["soc", "capacity", "voltage", "temperature", "efficiency"]),
                "All time-series arrays equal length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC039", "get_info component_id == EC039")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h sim at dt=1 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(20.0, 0.9, 298.15, 1.0, 3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_nernst_soc_monotone,
        test_charge_above_discharge,
        test_overpotentials_grow_with_current,
        test_concentration_diverges,
        test_coulomb_conservation,
        test_coulomb_efficiency_asymmetry,
        test_efficiency_bounds,
        test_thermal_balance,
        test_capacity_fade_monotone,
        test_discharge_depletes_soc,
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
    print(f"EC039 Organic Flow Battery F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
