"""
EC129 -- Run-of-River Hydropower -- F2a Physics-Lumped Headpond Transient
Test suite: physics sanity (energy conservation, monotonicity, known limits),
edge cases, predict() interface, and a benchmark timing test. No pytest.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import RunOfRiverF2a
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
def test_efficiency_range():
    print("\n[Test 1] 0 < efficiency < 1 over operating flows")
    m, _ = make_model()
    Q = np.linspace(0.3 * m.Q_design, m.Q_design, 25)
    eta = m.overall_efficiency(Q)
    for e in eta:
        assert_true(0.0 < e < 1.0, f"eta={e:.4f} in (0,1)")
    eta_peak_pt = m.overall_efficiency(m.Q_design)
    assert_true(np.isclose(eta_peak_pt, m.eta_peak * m.eta_gen, atol=1e-9),
                f"eta at BEP = eta_peak*eta_gen = {eta_peak_pt:.4f}")


def test_efficiency_peaks_at_design():
    print("\n[Test 2] Hill chart: efficiency maximal at design flow")
    m, _ = make_model()
    eta_des = m.turbine_efficiency(m.Q_design)
    for q in [0.5, 0.7, 0.9, 1.1]:
        eta_off = m.turbine_efficiency(q * m.Q_design)
        assert_true(eta_off <= eta_des + 1e-12,
                    f"eta(q={q}) {eta_off:.4f} <= eta_BEP {eta_des:.4f}")


def test_head_loss_grows_with_flow():
    print("\n[Test 3] Darcy-Weisbach head loss is >=0 and grows as Q^2")
    m, _ = make_model()
    Q1, Q2 = 20.0, 40.0
    h1, h2 = m.head_loss(Q1), m.head_loss(Q2)
    assert_true(h1 >= 0.0 and h2 >= 0.0, f"h_loss>=0: {h1:.4f}, {h2:.4f}")
    ratio = h2 / h1
    assert_true(np.isclose(ratio, (Q2 / Q1) ** 2, rtol=1e-9),
                f"h_loss ~ Q^2: ratio={ratio:.3f} vs {(Q2/Q1)**2:.3f}")


def test_head_loss_reduces_net_head():
    print("\n[Test 4] Net head < gross head (loss strictly reduces it)")
    m, _ = make_model()
    Hg = m.H_design
    Hn = float(m.net_head(m.Q_design, Hg))
    hl = float(m.head_loss(m.Q_design))
    assert_true(Hn < Hg, f"H_net={Hn:.3f} < H_gross={Hg:.3f}")
    assert_true(np.isclose(Hn, Hg - hl, atol=1e-9),
                f"H_net = H_gross - h_loss ({Hn:.3f} = {Hg:.3f}-{hl:.3f})")


def test_power_scales_with_Q_and_H():
    print("\n[Test 5] P scales with Q*H (energy-conservation form)")
    m, _ = make_model()
    # Fix efficiency by using design flow; vary gross head linearly.
    P1 = float(m.power_kw(m.Q_design, 6.0))
    P2 = float(m.power_kw(m.Q_design, 12.0))
    assert_true(P2 > P1, f"P rises with head: {P2:.1f} > {P1:.1f}")
    # Direct rho*g*Q*H_net*eta check (no rated clip at these values).
    Q, Hg = 30.0, 8.0
    eta = float(m.overall_efficiency(Q))
    Hn = float(m.net_head(Q, Hg))
    P_expect = eta * m.rho * m.g * Q * Hn / 1000.0
    P_model = float(m.power_kw(Q, Hg))
    assert_true(np.isclose(P_model, P_expect, rtol=1e-9),
                f"P = eta*rho*g*Q*H_net: {P_model:.2f} = {P_expect:.2f} kW")


def test_energy_conservation_efficiency():
    print("\n[Test 6] Extracted power <= available hydraulic power")
    m, _ = make_model()
    Q, Hg = 45.0, 8.0
    P_avail = m.rho * m.g * Q * Hg / 1000.0          # gross hydraulic kW
    P_elec = float(m.power_kw(Q, Hg))
    assert_true(P_elec < P_avail,
                f"P_elec={P_elec:.1f} < P_hydraulic_gross={P_avail:.1f} kW")


def test_pond_mass_balance_steady():
    print("\n[Test 7] Headpond ODE: inflow=turbine demand -> level steady")
    m, _ = make_model()
    r = m.simulate(Q_inflow=40.0, Q_demand=40.0, z0=8.0, dt=30.0, duration_s=3600.0)
    drift = abs(r["z"][-1] - r["z"][0])
    assert_true(drift < 1e-3, f"level steady when balanced: drift={drift:.2e} m")


def test_pond_drawdown_and_refill():
    print("\n[Test 8] Headpond ODE: over-draft lowers level, under-draft raises it")
    m, _ = make_model()
    # Turbine draws more than inflow -> forebay falls.
    r_dn = m.simulate(Q_inflow=20.0, Q_demand=50.0, z0=8.0, dt=30.0, duration_s=3600.0)
    assert_true(r_dn["z"][-1] < r_dn["z"][0],
                f"draft>inflow lowers level: {r_dn['z'][-1]:.3f} < {r_dn['z'][0]:.3f}")
    # Inflow exceeds draft -> forebay rises.
    r_up = m.simulate(Q_inflow=55.0, Q_demand=30.0, z0=7.0, dt=30.0, duration_s=1800.0)
    assert_true(r_up["z"][-1] > r_up["z"][0],
                f"inflow>draft raises level: {r_up['z'][-1]:.3f} > {r_up['z'][0]:.3f}")


def test_mass_balance_volume_closure():
    print("\n[Test 9] Volume balance closes: dV = (inflow-out)*dt integral")
    m, _ = make_model()
    r = m.simulate(Q_inflow=55.0, Q_demand=30.0, z0=7.0, dt=10.0, duration_s=1800.0)
    dV_storage = (r["z"][-1] - r["z"][0]) * m.A_pond
    net_in = r["Q_inflow"] - r["Q_turbine"] - r["Q_spill"]
    dV_flux = np.trapz(net_in, r["t"])
    rel = abs(dV_storage - dV_flux) / max(abs(dV_flux), 1.0)
    assert_true(rel < 0.02, f"storage vs flux balance closes: rel err={rel:.4f}")


def test_spill_caps_level():
    print("\n[Test 10] Spillway: large inflow spills, bounding the level")
    m, _ = make_model()
    r = m.simulate(Q_inflow=80.0, Q_demand=50.0, z0=9.0, dt=20.0, duration_s=7200.0)
    assert_true(np.any(r["Q_spill"] > 0.0), "spill activates above crest")
    assert_true(r["z"][-1] < m.z_max + 1.0,
                f"level bounded near crest: z={r['z'][-1]:.3f} (crest={m.z_max})")


def test_turbine_shut_below_zmin():
    print("\n[Test 11] No power when forebay below minimum operating level")
    m, _ = make_model()
    r = m.simulate(Q_inflow=2.0, Q_demand=50.0, z0=m.z_min - 0.1, dt=30.0, duration_s=600.0)
    assert_true(np.all(r["power_kw"] == 0.0), "P=0 below z_min")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() interface + info")
    _, cm = make_model()
    r = cm.predict({"Q_inflow_m3s": 50.0, "dt": 60.0, "duration_s": 1800.0})
    for key in ["t", "z", "H_gross", "H_net", "head_loss",
                "Q_inflow", "Q_turbine", "Q_spill", "eta", "power_kw"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["power_kw"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC129", "component_id EC129")
    assert_true("Gulliver" in info["source"], "Gulliver & Arndt cited in source")


def test_benchmark():
    print("\n[Test 13] Benchmark: 24h sim at dt=60s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(Q_inflow=50.0, Q_demand=48.0, z0=8.0, dt=60.0, duration_s=86400.0)
    elapsed = time.perf_counter() - t0
    print(f"  24h simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_efficiency_range,
        test_efficiency_peaks_at_design,
        test_head_loss_grows_with_flow,
        test_head_loss_reduces_net_head,
        test_power_scales_with_Q_and_H,
        test_energy_conservation_efficiency,
        test_pond_mass_balance_steady,
        test_pond_drawdown_and_refill,
        test_mass_balance_volume_closure,
        test_spill_caps_level,
        test_turbine_shut_below_zmin,
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
    print(f"EC129 RoR Hydro F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
