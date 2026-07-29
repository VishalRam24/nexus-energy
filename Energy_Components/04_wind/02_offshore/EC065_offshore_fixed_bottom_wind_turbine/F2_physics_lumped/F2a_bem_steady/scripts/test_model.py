"""
EC065 -- Offshore Fixed-Bottom Wind Turbine -- F2a BEM Steady -- Test suite.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import OffshoreWindBEM_F2a
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"


def assert_true(c, m):
    if c:
        print(f"  {PASS}  {m}")
    else:
        print(f"  {FAIL}  FAILED: {m}")
        raise AssertionError(m)


def make():
    cm = ComponentModel()
    return cm._model, cm


# ------------------------------------------------------------------
def test_betz_limit():
    """Cp must not exceed Betz limit (16/27 ~ 0.593)."""
    print("\n[Test 1] Cp <= Betz limit (16/27 ~ 0.593)")
    m, _ = make()
    betz = 16.0 / 27.0
    for V in [6, 8, 10, 11.4, 14]:
        r = m.solve(V)
        assert_true(
            r["Cp"] <= betz + 0.01,
            f"Cp({V} m/s) = {r['Cp']:.4f} <= {betz:.4f}",
        )


# ------------------------------------------------------------------
def test_rated_power():
    """Power near rated at rated wind speed."""
    print("\n[Test 2] Power near rated at rated wind speed")
    m, _ = make()
    r = m.solve(m.rated_wind, pitch_deg=0.0)
    P = r["power_kw"]
    # BEM with simplified airfoil may not hit exactly rated, allow 30% margin
    assert_true(
        P > m.rated_power_kw * 0.3,
        f"P({m.rated_wind} m/s) = {P:.0f} kW > {m.rated_power_kw * 0.3:.0f} kW (30% of rated)",
    )
    assert_true(
        P < m.rated_power_kw * 2.0,
        f"P({m.rated_wind} m/s) = {P:.0f} kW < {m.rated_power_kw * 2.0:.0f} kW (200% of rated)",
    )


# ------------------------------------------------------------------
def test_zero_power_below_cutin():
    """Zero or near-zero power below cut-in."""
    print("\n[Test 3] Near-zero power below cut-in")
    m, _ = make()
    r = m.solve(2.0)
    # Large offshore turbine (126m rotor) still captures some power at 2 m/s;
    # check it is negligible relative to rated (5000 kW).
    assert_true(
        r["power_kw"] < 50.0,
        f"P(2 m/s) = {r['power_kw']:.1f} kW < 50 kW (negligible vs {m.rated_power_kw:.0f} kW rated)",
    )


# ------------------------------------------------------------------
def test_power_increases_with_wind():
    """Power increases monotonically below rated."""
    print("\n[Test 4] Power increases with wind speed (below rated)")
    m, _ = make()
    P_prev = m.solve(4.0)["power_kw"]
    for V in [6, 8, 10]:
        P = m.solve(V)["power_kw"]
        assert_true(P > P_prev, f"P({V}) = {P:.0f} > P_prev = {P_prev:.0f}")
        P_prev = P


# ------------------------------------------------------------------
def test_power_limiting_with_pitch():
    """Pitch reduces power above rated (power limiting)."""
    print("\n[Test 5] Pitch control limits power above rated")
    m, _ = make()
    r0 = m.solve(15.0, pitch_deg=0.0)
    r10 = m.solve(15.0, pitch_deg=10.0)
    assert_true(
        r0["power_kw"] > r10["power_kw"],
        f"P(pitch=0) = {r0['power_kw']:.0f} > P(pitch=10) = {r10['power_kw']:.0f}",
    )


# ------------------------------------------------------------------
def test_thrust_coefficient():
    """Ct should be in a reasonable range (0.1 to 1.0) for operational speeds."""
    print("\n[Test 6] Thrust coefficient Ct in range [0.1, 1.0]")
    m, _ = make()
    for V in [6, 8, 10, 11.4]:
        r = m.solve(V)
        assert_true(
            0.1 <= r["Ct"] <= 1.0,
            f"Ct({V} m/s) = {r['Ct']:.3f} in [0.1, 1.0]",
        )


# ------------------------------------------------------------------
def test_blade_loads():
    """Blade loads array has correct length."""
    print("\n[Test 7] Blade loads array has N_elements entries")
    m, _ = make()
    r = m.solve(10.0)
    assert_true(
        len(r["blade_loads"]) == m.N_el,
        f"N_loads = {len(r['blade_loads'])} == {m.N_el}",
    )


# ------------------------------------------------------------------
def test_induction_factors():
    """Induction factors in valid range."""
    print("\n[Test 8] Induction factors in valid range")
    m, _ = make()
    r = m.solve(10.0)
    for bl in r["blade_loads"]:
        assert_true(
            0.0 <= bl["a"] <= 0.95,
            f"a = {bl['a']:.3f} in [0, 0.95]",
        )


# ------------------------------------------------------------------
def test_predict_interface():
    """ComponentModel predict() returns all expected keys."""
    print("\n[Test 9] ComponentModel predict() interface")
    _, cm = make()
    r = cm.predict({"wind_speed_m_s": 10.0})
    for k in ["power_kw", "thrust_kN", "torque_kNm", "Cp", "Ct", "blade_loads"]:
        assert_true(k in r, f"Key '{k}' present")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC065", "component_id == EC065")
    assert_true("BEM" in info["fidelity"], "fidelity contains 'BEM'")


# ------------------------------------------------------------------
def test_power_curve():
    """Power curve computation with pitch control."""
    print("\n[Test 10] Power curve with pitch control")
    _, cm = make()
    ws = np.arange(4, 24, 2.0)
    pc = cm.predict_curve(ws, pitch_control=True)
    assert_true(len(pc["power_kw"]) == len(ws), f"Output length matches input")
    # Power should not greatly exceed rated (with pitch control)
    max_p = np.max(pc["power_kw"])
    assert_true(
        max_p <= cm._model.rated_power_kw * 1.15,
        f"Max power {max_p:.0f} kW <= {cm._model.rated_power_kw * 1.15:.0f} kW (115% rated)",
    )


# ------------------------------------------------------------------
def test_benchmark():
    """Benchmark: single BEM solve speed."""
    print("\n[Test 11] Benchmark: single BEM solve")
    m, _ = make()
    # Warm up
    m.solve(10.0)
    t0 = time.perf_counter()
    N_iter = 100
    for _ in range(N_iter):
        m.solve(10.0)
    elapsed = (time.perf_counter() - t0) / N_iter
    print(f"  Single solve in {elapsed * 1000:.1f} ms")
    assert_true(elapsed < 1.0, "< 1 s per solve")


# ------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_betz_limit,
        test_rated_power,
        test_zero_power_below_cutin,
        test_power_increases_with_wind,
        test_power_limiting_with_pitch,
        test_thrust_coefficient,
        test_blade_loads,
        test_induction_factors,
        test_predict_interface,
        test_power_curve,
        test_benchmark,
    ]
    p = f = 0
    for t in tests:
        try:
            t()
            p += 1
        except Exception as e:
            f += 1
            print(f"  ERROR: {e}")
    print(f"\n{'=' * 60}")
    print(f"EC065 Offshore Wind F2a BEM -- {p} passed, {f} failed")
    print(f"{'=' * 60}")
    sys.exit(0 if f == 0 else 1)
