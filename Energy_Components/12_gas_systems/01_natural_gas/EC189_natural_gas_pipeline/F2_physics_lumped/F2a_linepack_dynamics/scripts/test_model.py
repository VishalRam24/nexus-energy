"""
EC189 -- Natural Gas Pipeline -- F2a Line-Pack Dynamics
Test suite: physics sanity (mass conservation, pressure-drop-with-flow,
line-pack storage, friction), edge cases, predict() interface, benchmark.
NO pytest -- custom assert harness, run with system python3.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import NGPipelineF2a
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
def test_no_flow_no_drop():
    print("\n[Test 1] Zero flow iff equal pressures")
    m, _ = make_model()
    P = 60e5
    assert_true(abs(m.flow_rate_std_m3_day(P, P)) < 1e-9,
                "Q = 0 when P1 = P2")
    Q = m.flow_rate_std_m3_day(70e5, 50e5)
    assert_true(Q > 0, f"Q > 0 when P1 > P2 (Q={Q:.3e} std m3/day)")


def test_flow_monotone_in_dp2():
    print("\n[Test 2] Flow increases with (P1^2 - P2^2)")
    m, _ = make_model()
    P1 = 70e5
    Q_prev = -1.0
    for P2 in [69e5, 65e5, 60e5, 50e5, 40e5]:
        Q = m.flow_rate_std_m3_day(P1, P2)
        assert_true(Q > Q_prev, f"Q(P2={P2/1e5:.0f}bar)={Q:.3e} > prev")
        Q_prev = Q


def test_flow_sqrt_scaling():
    print("\n[Test 3] Q ~ sqrt(P1^2 - P2^2) signature of isothermal compressible flow")
    m, _ = make_model()
    # Fix f by using the explicit Weymouth f so scaling is exact.
    f = 0.032 / (m.D_m * 1000.0) ** (1.0 / 3.0)
    Q1 = m.flow_rate_std_m3_day(70e5, 50e5, f=f)
    Q2 = m.flow_rate_std_m3_day(90e5, np.sqrt(90e5**2 - 2*(70e5**2 - 50e5**2)), f=f)
    # Q2 has 2x the (P1^2-P2^2) -> ratio sqrt(2)
    ratio = Q2 / Q1
    assert_true(abs(ratio - np.sqrt(2.0)) < 1e-6,
                f"Q ratio={ratio:.5f} ~ sqrt(2)={np.sqrt(2):.5f}")


def test_reverse_flow():
    print("\n[Test 4] Reverse flow when P2 > P1 (signed)")
    m, _ = make_model()
    Qf = m.flow_rate_std_m3_day(70e5, 50e5)
    Qr = m.flow_rate_std_m3_day(50e5, 70e5)
    assert_true(Qr < 0, f"Reverse Q={Qr:.3e} < 0")
    assert_true(abs(Qf + Qr) < 1e-6 * abs(Qf), "Forward/reverse antisymmetric")


def test_friction_physical():
    print("\n[Test 5] Friction factor positive & Colebrook in turbulent range")
    m, _ = make_model()
    Q = m.flow_rate_std_m3_day(70e5, 50e5)
    Re = m.reynolds(Q)
    f = m.friction_factor(Q)
    assert_true(Re > 4000, f"Transmission flow is turbulent (Re={Re:.3e})")
    assert_true(0.005 < f < 0.05, f"Darcy f={f:.5f} in physical range")
    # rougher wall -> higher f
    m.eps_m *= 5.0
    f2 = m.friction_factor(Q)
    assert_true(f2 > f, f"Rougher pipe raises f ({f2:.5f} > {f:.5f})")


def test_linepack_mass_rises_with_pressure():
    print("\n[Test 6] Line-pack mass monotone increasing in mean pressure")
    m, _ = make_model()
    masses = [m.linepack_mass_kg(P * 1e5) for P in [40, 50, 60, 70, 80]]
    for a, b in zip(masses, masses[1:]):
        assert_true(b > a, f"line-pack {b/1e3:.1f}t > {a/1e3:.1f}t")
    # sanity: 100 km of 24-in at 60 bar is ~ hundreds of tonnes
    m60 = m.linepack_mass_kg(60e5)
    assert_true(1e5 < m60 < 1e7, f"line-pack at 60bar = {m60/1e3:.0f} t (order check)")


def test_packing_raises_pressure():
    print("\n[Test 7] Line-pack storage: surplus inflow charges pipe (P_avg up)")
    _, cm = make_model()
    r = cm.predict({"P_avg0_bar": 55.0, "P_out_bar": 50.0,
                    "m_in_kg_s": 200.0, "dt": 60.0, "duration_s": 3600.0})
    assert_true(r["P_avg"][-1] > r["P_avg"][0],
                f"P_avg rose {r['P_avg'][0]:.2f}->{r['P_avg'][-1]:.2f} bar")
    assert_true(r["linepack_mass"][-1] > r["linepack_mass"][0],
                "Stored mass increased")


def test_drafting_lowers_pressure():
    print("\n[Test 8] Deficit inflow discharges pipe (P_avg down)")
    _, cm = make_model()
    r = cm.predict({"P_avg0_bar": 70.0, "P_out_bar": 50.0,
                    "m_in_kg_s": 1.0, "dt": 60.0, "duration_s": 3600.0})
    assert_true(r["P_avg"][-1] < r["P_avg"][0],
                f"P_avg fell {r['P_avg'][0]:.2f}->{r['P_avg'][-1]:.2f} bar")


def test_mass_conservation():
    print("\n[Test 9] Mass conservation: d(line-pack) = integral(m_in - m_out)")
    m, cm = make_model()
    r = cm.predict({"P_avg0_bar": 58.0, "P_out_bar": 50.0,
                    "m_in_kg_s": 120.0, "dt": 30.0, "duration_s": 7200.0})
    dM = r["linepack_mass"][-1] - r["linepack_mass"][0]
    _trap = getattr(np, "trapezoid", np.trapz)
    net = _trap(r["m_in"] - r["m_out"], r["t"])  # kg
    rel = abs(dM - net) / max(abs(dM), 1.0)
    assert_true(rel < 1e-3, f"dM={dM:.1f} kg vs integral={net:.1f} kg (rel {rel:.2e})")


def test_steady_state_balance():
    print("\n[Test 10] Driven mode reaches steady state (m_in ~ m_out)")
    _, cm = make_model()
    r = cm.predict({"P_avg0_bar": 55.0, "P_out_bar": 50.0,
                    "P_in_bar": 70.0, "dt": 60.0, "duration_s": 21600.0})
    imbalance = abs(r["m_in"][-1] - r["m_out"][-1]) / max(abs(r["m_out"][-1]), 1e-9)
    assert_true(imbalance < 1e-2,
                f"Near SS: |m_in-m_out|/m_out = {imbalance:.2e}")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"P_avg0_bar": 60.0, "P_out_bar": 50.0,
                    "m_in_kg_s": 100.0, "dt": 120.0, "duration_s": 1800.0})
    for key in ["t", "P_avg", "linepack_mass", "m_in", "m_out", "Q_out",
                "friction_factor", "reynolds"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["P_avg"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC189", "Info id EC189")


def test_benchmark():
    print("\n[Test 12] Benchmark: 24 h sim at dt=60 s")
    _, cm = make_model()
    t0 = time.perf_counter()
    cm.predict({"P_avg0_bar": 60.0, "P_out_bar": 50.0,
                "m_in_kg_s": 100.0, "dt": 60.0, "duration_s": 86400.0})
    elapsed = time.perf_counter() - t0
    print(f"  24h simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_no_flow_no_drop,
        test_flow_monotone_in_dp2,
        test_flow_sqrt_scaling,
        test_reverse_flow,
        test_friction_physical,
        test_linepack_mass_rises_with_pressure,
        test_packing_raises_pressure,
        test_drafting_lowers_pressure,
        test_mass_conservation,
        test_steady_state_balance,
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
    print(f"EC189 NG Pipeline F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
