"""
EC187 -- HVDC Converter Station -- F2a Physics-Lumped (LCC + DC-link ODE)
Test suite: converter physics sanity, energy conservation, ODE convergence,
reactive consumption, edge cases, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import HVDC_LCC_F2a
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
def test_vd_firing_relation():
    print("\n[Test 1] Vd = Vd0 cos(alpha) - (3/pi)Xc Id  (firing-angle relation)")
    m, _ = make_model()
    Id = 2000.0
    # cos(alpha) dependence: larger alpha => lower rectifier voltage
    V0 = m.Vd_rectifier(Id, np.deg2rad(0.0))
    V15 = m.Vd_rectifier(Id, np.deg2rad(15.0))
    V45 = m.Vd_rectifier(Id, np.deg2rad(45.0))
    assert_true(V0 > V15 > V45, f"Vd falls with alpha: {V0/1e3:.0f}>{V15/1e3:.0f}>{V45/1e3:.0f} kV")
    # Check exact cosine form at Id=0
    lhs = m.Vd_rectifier(0.0, np.deg2rad(30.0))
    rhs = m.n * m.Vd0_r * np.cos(np.deg2rad(30.0))
    assert_true(abs(lhs - rhs) < 1.0, "At Id=0, Vd == n*Vd0*cos(alpha)")
    # Commutation drop is linear in Id
    drop = m.Vd_rectifier(0.0, m.alpha_rect) - m.Vd_rectifier(2000.0, m.alpha_rect)
    expected = m.n * (3.0 / np.pi) * m.Xc * 2000.0
    assert_true(abs(drop - expected) < 1.0, "Commutation drop = n*(3/pi)*Xc*Id")


def test_alpha_90_zero_voltage():
    print("\n[Test 2] alpha -> 90 deg gives ~zero (sign-reversal) DC voltage")
    m, _ = make_model()
    V90 = m.Vd_rectifier(0.0, np.deg2rad(90.0))
    assert_true(abs(V90) < 1.0, f"Vd(alpha=90, Id=0) ~ 0: {V90:.2f} V")
    V100 = m.Vd_rectifier(0.0, np.deg2rad(100.0))
    assert_true(V100 < 0, "Vd < 0 for alpha > 90 (inverter operation)")


def test_steady_state_current_rated():
    print("\n[Test 3] Steady-state current near rated at nominal angles")
    m, _ = make_model()
    Id = m.steady_state_current()
    assert_true(1.8e3 < Id < 2.2e3, f"Id_ss={Id/1e3:.3f} kA near rated 2.0 kA")
    pb = m.power_balance(Id)
    assert_true(450e3 < pb["Vd_inv_V"] < 540e3, f"Vd_inv={pb['Vd_inv_V']/1e3:.0f} kV near 500 kV pole")


def test_ode_converges_to_steady_state():
    print("\n[Test 4] DC-link ODE converges to analytic steady-state current")
    m, _ = make_model()
    r = m.simulate(m.alpha_rect, Id0=0.0, dt=2e-3, duration_s=1.5)
    Id_final = r["Id_A"][-1]
    Id_ss = m.steady_state_current()
    rel = abs(Id_final - Id_ss) / Id_ss
    assert_true(rel < 0.01, f"ODE final {Id_final/1e3:.3f} kA vs analytic {Id_ss/1e3:.3f} kA (rel {rel:.4f})")
    # Current builds up monotonically from zero (no overshoot for over-damped RL link)
    assert_true(np.all(np.diff(r["Id_A"]) >= -1.0), "Id rises monotonically toward steady state")


def test_energy_conservation_dc():
    print("\n[Test 5] DC energy conservation at operating points: "
          "P_dc_rect - P_dc_inv = line loss")
    m, _ = make_model()
    # Conservation is a property of the DC-link equilibrium: at each operating
    # firing angle the steady current makes (Vd_rect - Vd_inv) = R_line*Id.
    for P_MW in [250.0, 500.0, 1000.0]:
        a = m.alpha_for_power(P_MW * 1e6)
        Id = m.steady_state_current(alpha=a)
        pb = m.power_balance(Id, alpha=a)
        bal = pb["P_dc_rect_W"] - pb["P_dc_inv_W"] - pb["P_line_loss_W"]
        scale = max(pb["P_dc_rect_W"], 1.0)
        assert_true(abs(bal) / scale < 1e-6,
                    f"P~{P_MW:.0f}MW (Id={Id/1e3:.2f}kA): residual {bal:.3e} W ~ 0")


def test_efficiency_bounds():
    print("\n[Test 6] Link efficiency in (0,1) and high (~0.95-0.99) at load")
    m, _ = make_model()
    Id = m.steady_state_current()
    pb = m.power_balance(Id)
    eta = pb["efficiency"]
    assert_true(0.0 < eta < 1.0, f"eta={eta:.4f} strictly in (0,1)")
    assert_true(0.93 < eta < 0.995, f"eta={eta:.4f} in realistic HVDC band")


def test_reactive_consumption_lcc():
    print("\n[Test 7] LCC consumes reactive power Q>0 at both ends, Q/P ~ 0.4-0.6")
    m, _ = make_model()
    Id = m.steady_state_current()
    pb = m.power_balance(Id)
    assert_true(pb["Q_rect_VAR"] > 0, f"Q_rect={pb['Q_rect_VAR']/1e6:.0f} MVAR > 0 (consumed)")
    assert_true(pb["Q_inv_VAR"] > 0, f"Q_inv={pb['Q_inv_VAR']/1e6:.0f} MVAR > 0 (consumed)")
    qp = pb["Q_rect_VAR"] / pb["P_dc_rect_W"]
    assert_true(0.35 < qp < 0.65, f"Q/P={qp:.3f} in textbook LCC range 0.4-0.6")
    # power factor below unity, increases reactive demand at higher firing angle
    pf_low = m.power_factor(Id, np.deg2rad(15.0), "rect")
    pf_high = m.power_factor(Id, np.deg2rad(40.0), "rect")
    assert_true(pf_high < pf_low < 1.0, f"pf drops with firing angle: {pf_high:.3f}<{pf_low:.3f}<1")


def test_power_increases_with_alpha_reduction():
    print("\n[Test 8] Lower firing angle (closer to 0) raises power transfer")
    m, _ = make_model()
    P_lo_a = m.power_balance(m.steady_state_current(alpha=np.deg2rad(10.0)),
                             alpha=np.deg2rad(10.0))["P_dc_rect_W"]
    P_hi_a = m.power_balance(m.steady_state_current(alpha=np.deg2rad(30.0)),
                             alpha=np.deg2rad(30.0))["P_dc_rect_W"]
    assert_true(P_lo_a > P_hi_a, f"P(alpha=10)={P_lo_a/1e6:.0f} > P(alpha=30)={P_hi_a/1e6:.0f} MW")
    # alpha_for_power round-trips
    a = m.alpha_for_power(800e6)
    P = m.power_balance(m.steady_state_current(alpha=a), alpha=a)["P_dc_rect_W"]
    assert_true(abs(P - 800e6) / 800e6 < 0.10, f"alpha_for_power(800MW) -> {P/1e6:.0f} MW (<10% err)")


def test_zero_power_edge():
    print("\n[Test 9] Edge case: alpha=gamma => near-zero current and power")
    m, _ = make_model()
    # Force rectifier voltage to match inverter back-emf -> no driving voltage
    a = m.alpha_for_power(0.0)
    Id = m.steady_state_current(alpha=a)
    assert_true(Id < 50.0, f"Id={Id:.1f} A ~ 0 at zero power order")
    pb = m.power_balance(max(Id, 0.0), alpha=a)
    assert_true(pb["P_dc_rect_W"] < 30e6, f"P={pb['P_dc_rect_W']/1e6:.2f} MW ~ 0")


def test_line_temperature_effect():
    print("\n[Test 10] Hotter DC line => higher resistance => lower current")
    m, _ = make_model()
    Id_cold = m.steady_state_current(T_line_degC=-10.0)
    Id_hot = m.steady_state_current(T_line_degC=70.0)
    assert_true(Id_hot < Id_cold, f"Id_hot={Id_hot/1e3:.3f} < Id_cold={Id_cold/1e3:.3f} kA")
    R_hot = m.line_resistance(70.0)
    R_cold = m.line_resistance(-10.0)
    assert_true(R_hot > R_cold, f"R(70C)={R_hot:.3f} > R(-10C)={R_cold:.3f} Ohm")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + units")
    _, cm = make_model()
    r = cm.predict({"P_order_MW": 1000.0, "dt": 5e-3, "duration_s": 0.4})
    for key in ["t", "Id_kA", "Vd_rect_kV", "Vd_inv_kV", "P_dc_rect_MW",
                "P_line_loss_MW", "efficiency", "Q_rect_MVAR", "steady_state"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["Id_kA"]), "Time-series arrays same length")
    ss = r["steady_state"]
    assert_true(0.0 < ss["efficiency"] < 1.0, f"SS eta={ss['efficiency']:.4f} in (0,1)")
    assert_true(900 < ss["P_transfer_MW"] < 1100, f"SS P={ss['P_transfer_MW']:.0f} MW near rated")


def test_benchmark():
    print("\n[Test 12] Benchmark: 0.5 s DC-link transient at dt=1e-4")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(m.alpha_rect, Id0=0.0, dt=1e-4, duration_s=0.5)
    elapsed = time.perf_counter() - t0
    print(f"  0.5 s transient (5000 steps) in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_vd_firing_relation,
        test_alpha_90_zero_voltage,
        test_steady_state_current_rated,
        test_ode_converges_to_steady_state,
        test_energy_conservation_dc,
        test_efficiency_bounds,
        test_reactive_consumption_lcc,
        test_power_increases_with_alpha_reduction,
        test_zero_power_edge,
        test_line_temperature_effect,
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
    print(f"EC187 HVDC LCC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
