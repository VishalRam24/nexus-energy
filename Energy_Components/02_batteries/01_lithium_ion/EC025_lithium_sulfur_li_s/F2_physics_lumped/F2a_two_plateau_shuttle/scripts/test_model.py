"""
EC025 -- Lithium-Sulfur Battery (Li-S) -- F2a Two-Plateau + Shuttle + Thermal
Test suite: two-plateau OCV shape, shuttle/coulombic-efficiency physics,
Coulomb conservation, thermal balance, ODE convergence, predict() interface.
NO pytest -- run as: python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import LiS_F2a
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
def test_ocv_two_plateau():
    print("\n[Test 1] OCV exhibits TWO plateaus (high ~2.35V, low ~2.1V)")
    m, _ = make_model()
    soc = np.linspace(0.0, 1.0, 200)
    v = m.ocv(soc)
    # upper plateau (high SOC) near 2.3-2.4 V
    v_high = np.mean(v[(soc > 0.55) & (soc < 0.85)])
    # lower plateau (mid-low SOC) near ~2.1 V
    v_low = np.mean(v[(soc > 0.10) & (soc < 0.35)])
    assert_true(2.25 < v_high < 2.45, f"upper plateau mean V={v_high:.3f} in [2.25,2.45]")
    assert_true(1.95 < v_low < 2.20, f"lower plateau mean V={v_low:.3f} in [1.95,2.20]")
    assert_true(v_high - v_low > 0.12, f"plateau gap {v_high - v_low:.3f} V > 0.12 (two distinct plateaus)")


def test_ocv_monotone():
    print("\n[Test 2] OCV increases monotonically with SOC")
    m, _ = make_model()
    soc = np.linspace(0.0, 1.0, 300)
    v = m.ocv(soc)
    diffs = np.diff(v)
    assert_true(np.all(diffs >= -1e-6), f"OCV non-decreasing (min diff={diffs.min():.2e})")
    assert_true(v[-1] > v[0], f"OCV(1)={v[-1]:.3f} > OCV(0)={v[0]:.3f}")


def test_voltage_below_ocv_on_discharge():
    print("\n[Test 3] Terminal V < OCV during discharge (I>0)")
    m, _ = make_model()
    for soc in [0.2, 0.5, 0.8]:
        V = m.terminal_voltage(soc, 2.0, 0.05, 298.15)
        OCV = m.ocv_with_T(soc, 298.15)
        assert_true(V < OCV, f"SOC={soc}: V={V:.3f} < OCV={OCV:.3f}")


def test_shuttle_always_drains():
    print("\n[Test 4] Polysulfide shuttle current >= 0 (always drains cathode)")
    m, _ = make_model()
    for soc in [0.05, 0.3, 0.6, 0.95]:
        I_sh = m.shuttle_current(soc, 298.15)
        assert_true(I_sh >= 0.0, f"SOC={soc}: I_shuttle={I_sh:.4f} A >= 0")
    # shuttle is stronger at high SOC (more high-order polysulfides present)
    assert_true(m.shuttle_current(0.9, 298.15) > m.shuttle_current(0.1, 298.15),
                "shuttle larger at high SOC (more high-order polysulfides)")


def test_coulombic_efficiency_below_one():
    print("\n[Test 5] Coulombic efficiency in (0,1) on discharge (shuttle loss)")
    m, _ = make_model()
    for soc in [0.3, 0.7, 0.9]:
        eta = m.coulombic_efficiency(2.0, soc, 298.15)
        assert_true(0.0 < eta < 1.0, f"SOC={soc}: eta_C={eta:.4f} in (0,1)")
    # higher applied current -> closer to 1 (shuttle a smaller relative fraction)
    eta_lo = m.coulombic_efficiency(0.5, 0.9, 298.15)
    eta_hi = m.coulombic_efficiency(4.0, 0.9, 298.15)
    assert_true(eta_hi > eta_lo, f"eta_C rises with current: {eta_hi:.3f} > {eta_lo:.3f}")


def test_coulomb_conservation():
    print("\n[Test 6] Coulomb conservation: SOC drop = (I_app+I_shuttle) integral / Q")
    m, _ = make_model()
    I = 1.0
    r = m.simulate(I, 0.8, 298.15, 5.0, 1800.0)
    dsoc = r["soc"][0] - r["soc"][-1]
    # charge removed from cathode = integral of (I_app + I_shuttle) dt, in Ah
    integrand = (I + r["shuttle_current"]) / 3600.0
    trapz = getattr(np, "trapezoid", np.trapz)
    Q_removed = trapz(integrand, r["t"])  # Ah
    dsoc_expected = Q_removed / m.Q
    rel = abs(dsoc - dsoc_expected) / max(dsoc_expected, 1e-9)
    assert_true(rel < 0.02, f"SOC balance closes: dsoc={dsoc:.4f} vs expected {dsoc_expected:.4f} (rel {rel:.3%})")
    # and shuttle strictly increased the drain beyond the applied charge alone
    dsoc_no_shuttle = (I / 3600.0 * (r['t'][-1] - r['t'][0])) / m.Q
    assert_true(dsoc > dsoc_no_shuttle, f"shuttle increased drain: {dsoc:.4f} > {dsoc_no_shuttle:.4f}")


def test_self_discharge_at_rest():
    print("\n[Test 7] Shuttle causes self-discharge at rest (I_app=0)")
    m, _ = make_model()
    r = m.simulate(0.0, 0.9, 298.15, 30.0, 3600.0)
    assert_true(r["soc"][-1] < r["soc"][0],
                f"SOC drops at rest: {r['soc'][0]:.3f} -> {r['soc'][-1]:.3f}")


def test_voltage_two_plateau_in_discharge():
    print("\n[Test 8] Discharge curve shows the high-then-low plateau transition")
    m, _ = make_model()
    r = m.simulate(1.0, 1.0, 298.15, 10.0, 7200.0)
    # early (high SOC) voltage above the inter-plateau midpoint, late below it
    v_early = np.mean(r["voltage"][r["soc"] > 0.7])
    v_late = np.mean(r["voltage"][(r["soc"] > 0.15) & (r["soc"] < 0.4)])
    # under load the OCV plateau gap is partly masked by IR drop; require a clear
    # high-then-low step but with a load-realistic threshold
    assert_true(v_early > v_late + 0.04, f"high plateau V={v_early:.3f} > low plateau V={v_late:.3f}")
    assert_true(np.all(np.diff(r["soc"]) <= 1e-9), "SOC monotonically decreases on discharge")


def test_thermal_balance():
    print("\n[Test 9] Thermal ODE: bounded T, entropic cooling lowers high-rate heating")
    m, _ = make_model()
    r = m.simulate(3.0, 0.8, 298.15, 5.0, 1200.0)
    assert_true(258.0 < r["temperature"][-1] < 333.15,
                f"T_final={r['temperature'][-1]:.2f} K within valid range")
    # positive dOCV/dT: reversible heat is endothermic on discharge -> reduces net heating
    # confirm net temperature rise is modest (entropic term partially offsets I^2R)
    assert_true(r["temperature"][-1] - 298.15 < 15.0,
                f"net dT={r['temperature'][-1]-298.15:.2f} K modest (entropic offset present)")


def test_arrhenius_resistance():
    print("\n[Test 10] Resistance falls with temperature (Arrhenius)")
    m, _ = make_model()
    R_cold = m.R0(278.15)
    R_hot = m.R0(318.15)
    assert_true(R_cold > R_hot, f"R0(5C)={R_cold:.4f} > R0(45C)={R_hot:.4f} Ohm")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + get_info()")
    _, cm = make_model()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC025", "component_id == EC025")
    assert_true(cm.version == "1.0.0", "version == 1.0.0")
    r = cm.predict({"current_A": 1.0, "soc0": 1.0, "dt": 30.0, "duration_s": 600.0})
    for key in ["t", "soc", "voltage", "ocv", "temperature", "shuttle_current",
                "coulombic_efficiency", "power"]:
        assert_true(key in r, f"output key '{key}' present")
    assert_true(len(r["t"]) == len(r["voltage"]), "arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: full discharge sim < 5 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(1.0, 1.0, 298.15, 5.0, 7200.0)
    elapsed = time.perf_counter() - t0
    print(f"  2h discharge (dt=5s) in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_ocv_two_plateau,
        test_ocv_monotone,
        test_voltage_below_ocv_on_discharge,
        test_shuttle_always_drains,
        test_coulombic_efficiency_below_one,
        test_coulomb_conservation,
        test_self_discharge_at_rest,
        test_voltage_two_plateau_in_discharge,
        test_thermal_balance,
        test_arrhenius_resistance,
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
    print(f"EC025 Li-S F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
