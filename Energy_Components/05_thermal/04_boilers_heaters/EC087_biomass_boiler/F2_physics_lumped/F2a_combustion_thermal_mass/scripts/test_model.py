"""
EC087 -- Biomass Boiler -- F2a Combustion + Thermal-Mass ODE
Test suite: conservation, physics sanity, moisture coupling, edge cases, interface.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import BiomassBoilerF2a
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
def test_lhv_moisture_coupling():
    print("\n[Test 1] Moisture lowers effective LHV (preserved F1 coupling)")
    m, _ = make_model()
    lhv_dry = m.effective_lhv(0.0)
    lhv_wet = m.effective_lhv(0.30)
    assert_true(lhv_dry > lhv_wet, f"LHV(0%)={lhv_dry:.0f} > LHV(30%)={lhv_wet:.0f} kJ/kg")
    # equals dry LHV at zero moisture
    assert_true(abs(lhv_dry - m.LHV_dry) < 1e-6, "LHV_eff == LHV_dry at w=0")
    # monotone decreasing
    ws = np.linspace(0.0, 0.5, 11)
    lhvs = [m.effective_lhv(w) for w in ws]
    assert_true(all(lhvs[i+1] < lhvs[i] for i in range(len(lhvs)-1)),
                "LHV_eff strictly decreasing in moisture")


def test_energy_conservation():
    print("\n[Test 2] Energy balance: Q_comb == Q_useful + Q_stack + Q_casing + storage")
    m, _ = make_model()
    r = m.simulate(1.0, 333.15, 323.15, dt=2.0, duration_s=3600.0)
    # at steady state, storage term ~ 0
    i = -1
    storage = m.C * (r["T_water_C"][i] - r["T_water_C"][i-1]) / (r["t"][i] - r["t"][i-1])
    lhs = r["Q_comb"][i]
    rhs = r["Q_useful"][i] + r["Q_stack"][i] + r["Q_casing"][i] + storage
    rel = abs(lhs - rhs) / max(lhs, 1e-6)
    assert_true(rel < 0.02, f"Energy closes: Q_comb={lhs:.2f} vs sum={rhs:.2f} (rel {rel:.4f})")


def test_mass_conservation():
    print("\n[Test 3] Mass balance: m_fuel + m_air == m_flue (ash neglected)")
    m, _ = make_model()
    r = m.simulate(0.8, 333.15, 323.15, dt=2.0, duration_s=600.0)
    i = -1
    inflow = r["m_fuel"][i] + r["m_air"][i]
    outflow = r["m_flue"][i]
    rel = abs(inflow - outflow) / max(inflow, 1e-9)
    assert_true(rel < 1e-6, f"m_in={inflow:.5f} == m_flue={outflow:.5f} kg/s")


def test_efficiency_range():
    print("\n[Test 4] Water-side efficiency in (0,1)")
    m, _ = make_model()
    r = m.simulate(1.0, 333.15, 323.15, dt=5.0, duration_s=2400.0)
    eta_ss = r["efficiency"][-1]
    assert_true(0.0 < eta_ss < 1.0, f"eta_ss={eta_ss:.3f} in (0,1)")
    assert_true(eta_ss > 0.6, f"eta_ss={eta_ss:.3f} physically reasonable (>0.6)")


def test_efficiency_drops_with_moisture():
    print("\n[Test 5] Higher moisture -> lower efficiency")
    _, cm_dry = make_model()
    cm_dry = ComponentModel(params={"moisture_content": 0.05})
    cm_wet = ComponentModel(params={"moisture_content": 0.40})
    r_dry = cm_dry.predict({"PLR": 1.0, "dt": 10.0, "duration_s": 3000.0})
    r_wet = cm_wet.predict({"PLR": 1.0, "dt": 10.0, "duration_s": 3000.0})
    eta_dry = r_dry["efficiency"][-1]
    eta_wet = r_wet["efficiency"][-1]
    assert_true(eta_wet < eta_dry, f"eta(40%)={eta_wet:.3f} < eta(5%)={eta_dry:.3f}")


def test_stack_loss_positive_and_load_dependent():
    print("\n[Test 6] Stack loss > 0 and rises with load")
    m, _ = make_model()
    r_lo = m.simulate(0.3, 333.15, 323.15, dt=5.0, duration_s=900.0)
    r_hi = m.simulate(1.0, 333.15, 323.15, dt=5.0, duration_s=900.0)
    assert_true(r_hi["Q_stack"][-1] > 0, f"Q_stack>0 ({r_hi['Q_stack'][-1]:.2f} kW)")
    assert_true(r_hi["Q_stack"][-1] > r_lo["Q_stack"][-1],
                f"Stack@PLR1={r_hi['Q_stack'][-1]:.2f} > Stack@PLR0.3={r_lo['Q_stack'][-1]:.2f} kW")


def test_thermal_warmup():
    print("\n[Test 7] Cold start: block heats up toward steady state")
    m, _ = make_model()
    r = m.simulate(1.0, 293.15, 323.15, dt=5.0, duration_s=3600.0)
    assert_true(r["T_water_C"][-1] > r["T_water_C"][0],
                f"T rises {r['T_water_C'][0]:.1f} -> {r['T_water_C'][-1]:.1f} degC")
    dT = abs(r["T_water_C"][-1] - r["T_water_C"][-2])
    assert_true(dT < 0.05, f"Near steady state: dT={dT:.4f} degC/step")


def test_partload_monotone_output():
    print("\n[Test 8] Steady useful output increases with PLR")
    m, _ = make_model()
    outs = []
    for plr in [0.2, 0.5, 0.8, 1.0]:
        r = m.simulate(plr, 333.15, 323.15, dt=10.0, duration_s=3000.0)
        outs.append(r["Q_useful"][-1])
    assert_true(all(outs[i+1] > outs[i] for i in range(len(outs)-1)),
                f"Q_useful monotone in PLR: {[f'{q:.1f}' for q in outs]} kW")


def test_fuel_lag():
    print("\n[Test 9] Fuel feed lags a PLR step (tau_fuel dynamics)")
    m, _ = make_model()
    def step(t):
        return 0.3 if t < 100 else 1.0
    r = m.simulate(step, 333.15, 323.15, dt=2.0, duration_s=400.0)
    i_step = int(np.argmin(np.abs(r["t"] - 100.0)))
    i_just = int(np.argmin(np.abs(r["t"] - 110.0)))
    i_late = int(np.argmin(np.abs(r["t"] - 360.0)))
    # just after step fuel has not yet reached the new demand
    assert_true(r["m_fuel"][i_just] < r["m_fuel"][i_late],
                "Fuel flow ramps up gradually after step (first-order lag)")


def test_zero_load_idle():
    print("\n[Test 10] PLR=0 -> no heat release, block cools to ambient-bounded")
    m, _ = make_model()
    r = m.simulate(0.0, 333.15, 323.15, dt=10.0, duration_s=3600.0)
    assert_true(r["Q_comb"][-1] < 1e-6, f"Q_comb~0 at PLR=0 ({r['Q_comb'][-1]:.4e} kW)")
    assert_true(r["T_water_C"][-1] <= r["T_water_C"][0] + 1e-6,
                "Block does not heat with no fuel")
    assert_true(r["T_water_C"][-1] >= m.T_air - 1e-6, "Block stays >= ambient")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"PLR": 0.7, "dt": 10.0, "duration_s": 300.0})
    for key in ["t", "T_water", "PLR", "m_fuel", "m_air", "m_flue",
                "Q_comb", "Q_stack", "Q_useful", "efficiency"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_water"]) == len(r["efficiency"]),
                "Output arrays equal length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC087" and info["version"] == "1.0.0",
                "get_info() metadata correct")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h sim at dt=2 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(1.0, 293.15, 323.15, dt=2.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_lhv_moisture_coupling,
        test_energy_conservation,
        test_mass_conservation,
        test_efficiency_range,
        test_efficiency_drops_with_moisture,
        test_stack_loss_positive_and_load_dependent,
        test_thermal_warmup,
        test_partload_monotone_output,
        test_fuel_lag,
        test_zero_load_idle,
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
    print(f"EC087 Biomass Boiler F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
