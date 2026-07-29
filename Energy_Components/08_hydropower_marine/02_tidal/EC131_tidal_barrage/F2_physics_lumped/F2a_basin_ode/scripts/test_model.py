"""
EC131 -- Tidal Barrage -- F2a Physics-Lumped Basin ODE
Test suite: mass/energy conservation, flow laws, ODE behaviour, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import TidalBarrageF2a
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
def test_sea_level_sinusoid():
    print("\n[Test 1] Sea level is a bounded sinusoid")
    m, _ = make_model()
    t = np.linspace(0, m.T, 500)
    z = m.sea_level(t)
    assert_true(np.max(z) <= m.z0 + m.a + 1e-9, f"max sea {np.max(z):.3f} <= amplitude")
    assert_true(np.min(z) >= m.z0 - m.a - 1e-9, f"min sea {np.min(z):.3f} >= -amplitude")
    # peak near quarter period
    assert_true(abs(m.sea_level(m.T / 4.0) - (m.z0 + m.a)) < 1e-6, "peak at T/4")


def test_orifice_flow_sqrt_head():
    print("\n[Test 2] Turbine flow follows sqrt(head) and sign of head")
    m, _ = make_model()
    Q1 = m.turbine_flow(1.0)
    Q4 = m.turbine_flow(4.0)
    # Q ~ sqrt(H): doubling sqrt -> 4x head gives 2x flow
    assert_true(abs(Q4 / Q1 - 2.0) < 1e-6, f"Q(4)/Q(1)={Q4/Q1:.4f} ~ 2 (sqrt law)")
    assert_true(m.turbine_flow(-2.0) < 0, "Negative head -> negative (emptying) flow")
    assert_true(abs(m.turbine_flow(0.0)) < 1e-12, "Zero head -> zero flow")


def test_power_formula():
    print("\n[Test 3] P = eta * rho * g * Q * H (positive, below eta bound)")
    m, _ = make_model()
    # craft a generating state: basin well above sea (ebb), |H| >= h_start
    t = m.T * 0.30  # falling sea past peak
    z_sea = m.sea_level(t)
    z_basin = z_sea + m.h_start + 1.0  # head = -(h_start+1), magnitude > h_start
    P = m.power_elec(t, z_basin)
    H = z_sea - z_basin
    Q, _ = m.flow(t, z_basin)
    P_expect = m.eta_t * m.rho * m.g * abs(Q) * abs(H)
    assert_true(P > 0, f"Generating power P={P/1e6:.2f} MW > 0")
    assert_true(abs(P - P_expect) < 1e-3 * max(P, 1.0), "P matches rho g Q H eta")
    assert_true(P < m.rho * m.g * abs(Q) * abs(H) + 1e-6, "P below hydraulic (eta<1)")


def test_no_generation_below_hmin():
    print("\n[Test 4] No power generated below minimum head")
    m, _ = make_model()
    t = m.T * 0.30
    z_sea = m.sea_level(t)
    z_basin = z_sea + 0.5 * m.h_min  # head magnitude < h_min
    P = m.power_elec(t, z_basin)
    assert_true(P == 0.0, f"P={P} W == 0 below h_min={m.h_min} m")


def test_efficiency_bound():
    print("\n[Test 5] Turbine efficiency strictly in (0,1)")
    m, _ = make_model()
    assert_true(0.0 < m.eta_t < 1.0, f"eta_turbine={m.eta_t} in (0,1)")


def test_mass_conservation():
    print("\n[Test 6] Mass conservation: basin level periodic over full cycles")
    m, cm = make_model()
    r = cm.predict({"n_cycles": 4, "n_eval": 4000})
    z = r["z_basin"]
    # Over integer cycles the basin returns near its periodic band:
    # compare level at end vs one period earlier (limit-cycle convergence).
    t = r["t"]
    idx_end = -1
    idx_prev = np.argmin(np.abs(t - (t[-1] - m.T)))
    drift = abs(z[idx_end] - z[idx_prev])
    assert_true(drift < 0.5, f"Per-cycle basin drift {drift:.3f} m < 0.5 m (limit cycle)")
    # Basin stays within sea tidal envelope plus a margin
    assert_true(np.max(np.abs(z)) < m.a + m.h_start + 2.0,
                f"Basin level bounded: max|z|={np.max(np.abs(z)):.2f} m")


def test_volume_balance():
    print("\n[Test 7] Volume in/out balance (incompressible, periodic basin)")
    m, cm = make_model()
    r = cm.predict({"n_cycles": 4, "n_eval": 4000})
    V_in, V_out = r["volume_in_m3"], r["volume_out_m3"]
    rel = abs(V_in - V_out) / max(V_in, V_out, 1.0)
    assert_true(rel < 0.10, f"|V_in-V_out|/V = {rel*100:.1f}% < 10% (mass balance)")


def test_energy_positive_and_bounded():
    print("\n[Test 8] Energy/cycle > 0 and below thermodynamic max bound")
    m, cm = make_model()
    r = cm.predict({"n_cycles": 2})
    E = r["energy_per_cycle_MWh"]
    E_max = r["max_energy_per_cycle_MWh"]          # 2 rho g A a^2 (Baker bound)
    E_prandle = r["theoretical_energy_per_cycle_MWh"]  # 0.5 rho g A a^2 (avg-power)
    assert_true(E > 0, f"E/cycle = {E:.1f} MWh > 0")
    assert_true(E < E_max, f"E/cycle {E:.1f} < hard max {E_max:.1f} MWh (eta<1, finite flow)")
    # Should be the same order of magnitude as the Prandle single-effect figure
    assert_true(0.3 * E_prandle < E < 6.0 * E_prandle,
                f"E/cycle {E:.1f} same order as Prandle {E_prandle:.1f} MWh")


def test_energy_scales_with_amplitude():
    print("\n[Test 9] Larger tidal amplitude -> more energy (monotone)")
    _, cm = make_model()
    r_small = cm.predict({"n_cycles": 2, "tidal_amplitude_m": 3.0})
    cm2 = ComponentModel()
    r_big = cm2.predict({"n_cycles": 2, "tidal_amplitude_m": 6.0})
    assert_true(r_big["energy_per_cycle_MWh"] > r_small["energy_per_cycle_MWh"],
                f"E(6m)={r_big['energy_per_cycle_MWh']:.0f} > "
                f"E(3m)={r_small['energy_per_cycle_MWh']:.0f} MWh")


def test_flood_gen_adds_energy():
    print("\n[Test 10] Two-way (flood+ebb) generation >= ebb-only")
    _, cm = make_model()
    r_ebb = cm.predict({"n_cycles": 3, "flood_gen": False})
    cm2 = ComponentModel()
    r_two = cm2.predict({"n_cycles": 3, "flood_gen": True})
    assert_true(r_two["energy_per_cycle_MWh"] >= r_ebb["energy_per_cycle_MWh"] - 1e-6,
                f"two-way {r_two['energy_per_cycle_MWh']:.0f} >= "
                f"ebb {r_ebb['energy_per_cycle_MWh']:.0f} MWh")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(cm.component_id == "EC131", "component_id == EC131")
    r = cm.predict({"n_cycles": 1, "n_eval": 500})
    for key in ["t", "z_sea", "z_basin", "head", "flow", "power",
                "energy_per_cycle_MWh", "avg_power_MW"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["power"]), "Arrays same length")
    assert_true(r["solver_success"], "ODE solver succeeded")


def test_benchmark():
    print("\n[Test 12] Benchmark: 2-cycle simulation timing")
    _, cm = make_model()
    t0 = time.perf_counter()
    cm.predict({"n_cycles": 2, "n_eval": 2000})
    elapsed = time.perf_counter() - t0
    print(f"  2-cycle simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_sea_level_sinusoid,
        test_orifice_flow_sqrt_head,
        test_power_formula,
        test_no_generation_below_hmin,
        test_efficiency_bound,
        test_mass_conservation,
        test_volume_balance,
        test_energy_positive_and_bounded,
        test_energy_scales_with_amplitude,
        test_flood_gen_adds_energy,
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
    print(f"EC131 Tidal Barrage F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
