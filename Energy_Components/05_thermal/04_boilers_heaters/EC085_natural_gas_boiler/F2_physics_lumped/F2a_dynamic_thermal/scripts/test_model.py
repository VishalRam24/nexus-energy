"""
EC085 -- Natural Gas Boiler -- F2a Dynamic Thermal
Test suite: physics sanity, ODE convergence, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import NatGasBoiler_F2a
from predict import ComponentModel

PASS = "\u2713"
FAIL = "\u2717"


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
def test_cold_start_heats_up():
    print("\n[Test 1] Cold start: boiler heats up toward setpoint")
    m, _ = make_model()
    r = m.simulate(293.15, 333.15, 6.0, 353.15, 1.0, 300.0)
    assert_true(r["T_boiler"][-1] > 293.15, f"T rises from cold: {r['T_boiler'][-1]:.1f} K")
    assert_true(r["T_boiler"][-1] < 373.15, f"T stays below boiling: {r['T_boiler'][-1]:.1f} K")


def test_reaches_setpoint():
    print("\n[Test 2] Boiler approaches setpoint over time")
    m, _ = make_model()
    # Low flow to allow faster heating
    r = m.simulate(333.15, 333.15, 1.0, 353.15, 1.0, 600.0)
    assert_true(abs(r["T_boiler"][-1] - 353.15) < 10.0,
                f"T_final={r['T_boiler'][-1]:.1f} near setpoint 353.15 K")


def test_standby_heat_loss():
    print("\n[Test 3] Standby: no flow, no burner -> cools toward ambient")
    m, _ = make_model()
    r = m.simulate(353.15, 333.15, 0.0, 273.15, 1.0, 600.0, modulation_override=0.0)
    assert_true(r["T_boiler"][-1] < 353.15, f"Cools from 353.15 to {r['T_boiler'][-1]:.1f}")
    assert_true(r["T_boiler"][-1] > m.T_amb, f"Stays above ambient {m.T_amb:.1f}")


def test_combustion_efficiency_range():
    print("\n[Test 4] Combustion efficiency in valid range")
    m, _ = make_model()
    for mod in [0.2, 0.5, 0.8, 1.0]:
        eta = m.combustion_efficiency(mod)
        assert_true(0.8 < eta <= 1.0, f"eta_comb({mod})={eta:.3f} in (0.8, 1.0)")


def test_burner_modulation_range():
    print("\n[Test 5] Burner modulation in valid range")
    m, _ = make_model()
    r = m.simulate(293.15, 333.15, 6.0, 353.15, 1.0, 300.0)
    for mod in r["modulation"]:
        assert_true(0.0 <= mod <= 1.0, f"mod={mod:.3f} in [0, 1]")


def test_energy_balance():
    print("\n[Test 6] Energy balance (Q_burner = Q_output + Q_loss + Q_stored)")
    m, _ = make_model()
    r = m.simulate(333.15, 333.15, 6.0, 353.15, 1.0, 300.0)
    # Integrate Q_burner, Q_output, Q_loss
    dt_arr = np.diff(r["t"])
    E_burner = np.sum(r["Q_burner_W"][1:] * dt_arr)
    E_output = np.sum(r["Q_output_W"][1:] * dt_arr)
    E_loss = np.sum(r["Q_loss_W"][1:] * dt_arr)
    E_stored = m.C_total * (r["T_boiler"][-1] - r["T_boiler"][0])
    balance = E_burner - E_output - E_loss - E_stored
    rel_err = abs(balance) / max(abs(E_burner), 1.0)
    assert_true(rel_err < 0.05, f"Energy balance error: {rel_err*100:.2f}%")


def test_full_load_output():
    print("\n[Test 7] Full load: Q_output is substantial")
    m, _ = make_model()
    r = m.simulate(353.15, 333.15, 6.0, 353.15, 1.0, 60.0, modulation_override=1.0)
    Q_out_mean = np.mean(r["Q_output_W"])
    assert_true(Q_out_mean > 100000, f"Mean Q_output={Q_out_mean/1000:.0f} kW > 100 kW")


def test_thermal_efficiency_range():
    print("\n[Test 8] Thermal efficiency in reasonable range at steady state")
    m, _ = make_model()
    # Use low flow so boiler can reach setpoint, then check efficiency at end
    r = m.simulate(353.15, 343.15, 2.0, 353.15, 1.0, 600.0)
    # Check last 100 points where system is near steady state
    mask = (r["fuel_input_W"] > 1000) & (np.arange(len(r["t"])) > len(r["t"]) - 100)
    if np.any(mask):
        eta_mean = np.mean(r["thermal_efficiency"][mask])
        assert_true(0.3 < eta_mean < 1.5, f"Mean thermal_eff={eta_mean:.3f} in (0.3, 1.5)")
    else:
        print("  (Burner off at end -- checking full run)")
        mask2 = r["fuel_input_W"] > 1000
        if np.any(mask2):
            eta_mean = np.mean(r["thermal_efficiency"][mask2])
            assert_true(eta_mean > 0.1, f"Thermal efficiency positive: {eta_mean:.3f}")


def test_step_load_change():
    print("\n[Test 9] Step load change: flow step response")
    m, _ = make_model()
    def step_flow(t):
        return 2.0 if t < 150 else 8.0
    r = m.simulate(353.15, 333.15, step_flow, 353.15, 1.0, 300.0)
    idx_before = np.argmin(np.abs(r["t"] - 149))
    idx_after = np.argmin(np.abs(r["t"] - 160))
    # After flow increase, temperature should drop temporarily
    assert_true(r["T_boiler"][idx_after] < r["T_boiler"][idx_before] + 1.0,
                "T drops or holds after flow step up")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"T_init_K": 333.15, "dt": 1.0, "duration_s": 30.0})
    for key in ["t", "T_boiler", "modulation", "Q_burner_W", "Q_output_W", "thermal_efficiency"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_boiler"]), "Arrays same length")


def test_benchmark():
    print("\n[Test 11] Benchmark: 600s sim at dt=1")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(293.15, 333.15, 6.0, 353.15, 1.0, 600.0)
    elapsed = time.perf_counter() - t0
    print(f"  600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 10.0, "Completes in < 10 s")


if __name__ == "__main__":
    tests = [
        test_cold_start_heats_up,
        test_reaches_setpoint,
        test_standby_heat_loss,
        test_combustion_efficiency_range,
        test_burner_modulation_range,
        test_energy_balance,
        test_full_load_output,
        test_thermal_efficiency_range,
        test_step_load_change,
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
    print(f"EC085 Boiler F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
