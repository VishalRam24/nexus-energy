"""
EC024 -- Silicon-Anode Li-ion Battery (Si/NMC) -- F2a Thevenin ECM
Test suite: Coulomb conservation, OCV monotonicity per branch, Si hysteresis,
Arrhenius R(T), thermal balance, efficiency bounds, RC transient, interface.
Run: python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SiAnodeECM_F2a
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
def test_ocv_monotone_per_branch():
    print("\n[Test 1] OCV monotonically increasing with SOC (each branch)")
    m, _ = make_model()
    soc = np.linspace(0.0, 1.0, 200)
    for name, fn in [("charge", m.ocv_charge), ("discharge", m.ocv_discharge)]:
        v = fn(soc)
        assert_true(np.all(np.diff(v) >= -1e-9),
                    f"{name} branch OCV nondecreasing over SOC")
    # endpoints sane
    assert_true(2.5 < m.ocv_discharge(0.0) < 3.3, "OCV(0) low-SOC sane")
    assert_true(3.9 < m.ocv_charge(1.0) < 4.3, "OCV(1) high-SOC sane")


def test_hysteresis_gap_positive():
    print("\n[Test 2] Si hysteresis: charge branch above discharge branch")
    m, _ = make_model()
    soc = np.linspace(0.05, 0.95, 50)
    gap = m.hysteresis_gap(soc)
    assert_true(np.all(gap >= -1e-9), "charge OCV >= discharge OCV everywhere")
    assert_true(np.mean(gap) > 0.02,
                f"mean hysteresis gap {np.mean(gap)*1000:.1f} mV (large, Si-like)")


def test_hysteresis_state_branches():
    print("\n[Test 3] Hysteresis state drives correct OCV branch")
    m, _ = make_model()
    # discharge then OCV should approach discharge branch (h->-1)
    rd = m.simulate(3.5, 0.9, 298.15, 5.0, 600.0)
    assert_true(rd["hysteresis"][-1] < -0.5,
                f"discharge -> h={rd['hysteresis'][-1]:.2f} (toward -1)")
    # charge -> h -> +1
    rc = m.simulate(-3.5, 0.1, 298.15, 5.0, 600.0)
    assert_true(rc["hysteresis"][-1] > 0.5,
                f"charge -> h={rc['hysteresis'][-1]:.2f} (toward +1)")


def test_coulomb_conservation():
    print("\n[Test 4] Coulomb conservation: dSOC = -eta*I*t/(Q*3600)")
    m, _ = make_model()
    I, dur = 3.5, 300.0          # discharge, eta=1
    r = m.simulate(I, 0.9, 298.15, 1.0, dur)
    expected = -I * dur / (m.Q * 3600.0)
    actual = r["soc"][-1] - r["soc"][0]
    assert_true(abs(actual - expected) < 1e-4,
                f"discharge dSOC actual={actual:.5f} vs expected={expected:.5f}")
    # charge: efficiency <1 means less SOC gained per coulomb
    Ic = -3.5
    rc = m.simulate(Ic, 0.1, 298.15, 1.0, dur)
    expected_c = -m.eta_c * Ic * dur / (m.Q * 3600.0)
    actual_c = rc["soc"][-1] - rc["soc"][0]
    assert_true(abs(actual_c - expected_c) < 1e-4,
                f"charge dSOC actual={actual_c:.5f} vs expected={expected_c:.5f}")
    assert_true(actual_c < -m.eta_c * Ic * dur / (m.Q * 3600.0) + 1e-6
                and m.eta_c < 1.0, "coulombic efficiency <1 reduces gained SOC")


def test_efficiency_bounds():
    print("\n[Test 5] 0 < efficiency < 1 over a discharge")
    m, _ = make_model()
    r = m.simulate(3.5, 0.8, 298.15, 2.0, 400.0)
    eta = r["efficiency"]
    assert_true(np.all(eta > 0.0) and np.all(eta < 1.0),
                f"eta in (0,1): min={eta.min():.3f} max={eta.max():.3f}")


def test_coulombic_eff_param():
    print("\n[Test 6] Coulombic efficiency parameter in (0,1)")
    m, _ = make_model()
    assert_true(0.0 < m.eta_c < 1.0, f"eta_coulombic={m.eta_c} in (0,1)")
    assert_true(m.coulombic_eff(-3.5) == m.eta_c, "charge uses eta_c")
    assert_true(m.coulombic_eff(3.5) == 1.0, "discharge eta=1")


def test_arrhenius_resistance():
    print("\n[Test 7] Arrhenius R(T): resistance falls as T rises")
    m, _ = make_model()
    R_cold = m.R0(273.15)
    R_ref = m.R0(298.15)
    R_hot = m.R0(323.15)
    assert_true(R_cold > R_ref > R_hot,
                f"R0: cold={R_cold*1e3:.2f} > ref={R_ref*1e3:.2f} > hot={R_hot*1e3:.2f} mOhm")
    assert_true(abs(R_ref - m.R0_ref) < 1e-9, "R0(T_ref)=R0_ref")


def test_voltage_below_ocv_on_discharge():
    print("\n[Test 8] Terminal V < OCV on discharge, V > OCV on charge")
    m, _ = make_model()
    rd = m.simulate(3.5, 0.7, 298.15, 2.0, 200.0)
    assert_true(np.all(rd["voltage"][1:] <= rd["ocv"][1:] + 1e-6),
                "discharge: V_t <= OCV (IR + RC drop)")
    rc = m.simulate(-3.5, 0.3, 298.15, 2.0, 200.0)
    assert_true(np.all(rc["voltage"][1:] >= rc["ocv"][1:] - 1e-6),
                "charge: V_t >= OCV (IR + RC rise)")


def test_thermal_balance():
    print("\n[Test 9] Thermal ODE: heats under load, relaxes to ambient at rest")
    m, _ = make_model()
    rh = m.simulate(14.0, 0.8, 298.15, 1.0, 300.0)   # ~4C discharge
    assert_true(rh["temperature"][-1] > 298.15,
                f"heats up under load: T={rh['temperature'][-1]:.2f} K")
    assert_true(rh["temperature"][-1] < 333.15,
                f"stays in valid range: T={rh['temperature'][-1]:.2f} K")
    # rest from a hot start relaxes toward ambient
    rr = m.simulate(0.0, 0.5, 310.0, 5.0, 1200.0)
    assert_true(rr["temperature"][-1] < 310.0 and rr["temperature"][-1] > 297.0,
                f"rest relaxes toward ambient: T={rr['temperature'][-1]:.2f} K")


def test_rc_transient():
    print("\n[Test 10] RC transient: voltage relaxes after current step to rest")
    m, _ = make_model()
    def step(t):
        return 7.0 if t < 100.0 else 0.0
    r = m.simulate(step, 0.7, 298.15, 1.0, 300.0)
    # immediately after step-down there is an instantaneous IR recovery, then
    # the RC overpotential decays -> voltage keeps rising over the rest period
    i_after = np.argmin(np.abs(r["t"] - 105.0))
    i_end = np.argmin(np.abs(r["t"] - 290.0))
    assert_true(r["voltage"][i_end] > r["voltage"][i_after],
                "voltage relaxes upward during rest (RC decay)")
    assert_true(abs(r["v_rc1"][i_end]) < abs(r["v_rc1"][i_after]) + 1e-6,
                "fast RC overpotential decays toward 0 at rest")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + info")
    _, cm = make_model()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC024", "component_id EC024")
    r = cm.predict({"current_A": 3.5, "soc0": 0.9, "dt": 5.0, "duration_s": 100.0})
    for key in ["t", "soc", "voltage", "ocv", "current", "power",
                "v_rc1", "v_rc2", "hysteresis", "temperature",
                "swelling_strain", "efficiency", "components"]:
        assert_true(key in r, f"key '{key}' in output")
    assert_true(len(r["t"]) == len(r["voltage"]) == len(r["soc"]),
                "arrays same length")


def test_swelling_and_benchmark():
    print("\n[Test 12] Si swelling diagnostic + benchmark timing")
    m, _ = make_model()
    assert_true(m.swelling_strain(0.0) == 0.0, "zero strain at SOC=0")
    assert_true(m.swelling_strain(1.0) > 0.1,
                f"large strain at full SOC: {m.swelling_strain(1.0)*100:.0f}%")
    t0 = time.perf_counter()
    m.simulate(3.5, 0.9, 298.15, 1.0, 600.0)
    elapsed = time.perf_counter() - t0
    print(f"  600s sim (dt=1) in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_ocv_monotone_per_branch,
        test_hysteresis_gap_positive,
        test_hysteresis_state_branches,
        test_coulomb_conservation,
        test_efficiency_bounds,
        test_coulombic_eff_param,
        test_arrhenius_resistance,
        test_voltage_below_ocv_on_discharge,
        test_thermal_balance,
        test_rc_transient,
        test_predict_interface,
        test_swelling_and_benchmark,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"EC024 Si-Anode Li-ion F2a ECM -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
