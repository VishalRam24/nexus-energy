"""
EC054 -- Parabolic Trough CSP -- F2a HCE Thermal Model -- Test Suite
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import ParabolicTroughF2a
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"


def assert_true(condition, msg):
    if condition:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def make():
    cm = ComponentModel()
    return cm._model, cm


# ------------------------------------------------------------------
# Test 1: Thermal efficiency in expected range at typical conditions
# ------------------------------------------------------------------
def test_thermal_efficiency_typical():
    print("\n[Test 1] Thermal efficiency 60-75% at typical conditions")
    m, _ = make()
    for dni in [800.0, 900.0, 1000.0]:
        r = m.solve(dni=dni, theta_deg=0.0, T_htf_in_C=300.0,
                    m_dot=6.0, T_amb_C=25.0)
        eta = r["eta_thermal"]
        assert_true(0.55 <= eta <= 0.80,
                    f"DNI={dni}: eta_thermal={eta:.3f} in [0.55, 0.80]")
        assert_true(r["converged"], f"DNI={dni}: solver converged")


# ------------------------------------------------------------------
# Test 2: Zero DNI -> zero output
# ------------------------------------------------------------------
def test_zero_dni():
    print("\n[Test 2] Zero DNI -> zero output")
    m, _ = make()
    r = m.solve(dni=0.0, theta_deg=0.0, T_htf_in_C=300.0, m_dot=6.0)
    assert_true(r["Q_useful_W"] == 0.0, f"Q_useful={r['Q_useful_W']:.1f} == 0")
    assert_true(r["T_htf_out_C"] == 300.0, f"T_out={r['T_htf_out_C']:.1f} == T_in")
    assert_true(r["eta_thermal"] == 0.0, "eta_thermal == 0")


# ------------------------------------------------------------------
# Test 3: Higher inlet temp -> lower efficiency (more losses)
# ------------------------------------------------------------------
def test_higher_temp_lower_efficiency():
    print("\n[Test 3] Higher inlet temperature -> lower efficiency")
    m, _ = make()
    r_low = m.solve(dni=900.0, theta_deg=0.0, T_htf_in_C=200.0, m_dot=6.0)
    r_mid = m.solve(dni=900.0, theta_deg=0.0, T_htf_in_C=300.0, m_dot=6.0)
    r_high = m.solve(dni=900.0, theta_deg=0.0, T_htf_in_C=380.0, m_dot=6.0)
    assert_true(r_low["eta_thermal"] > r_mid["eta_thermal"],
                f"eta(200C)={r_low['eta_thermal']:.3f} > eta(300C)={r_mid['eta_thermal']:.3f}")
    assert_true(r_mid["eta_thermal"] > r_high["eta_thermal"],
                f"eta(300C)={r_mid['eta_thermal']:.3f} > eta(380C)={r_high['eta_thermal']:.3f}")


# ------------------------------------------------------------------
# Test 4: Q_useful > 0 for reasonable conditions
# ------------------------------------------------------------------
def test_positive_useful_heat():
    print("\n[Test 4] Q_useful > 0 for reasonable conditions")
    m, _ = make()
    r = m.solve(dni=800.0, theta_deg=10.0, T_htf_in_C=250.0, m_dot=5.0)
    assert_true(r["Q_useful_W"] > 0, f"Q_useful={r['Q_useful_W']/1000:.1f} kW > 0")
    assert_true(r["Q_abs_W"] > 0, f"Q_abs={r['Q_abs_W']/1000:.1f} kW > 0")


# ------------------------------------------------------------------
# Test 5: Energy conservation: Q_abs ~ Q_useful + Q_loss
# ------------------------------------------------------------------
def test_energy_conservation():
    print("\n[Test 5] Energy balance: Q_abs = Q_useful + Q_loss")
    m, _ = make()
    r = m.solve(dni=900.0, theta_deg=5.0, T_htf_in_C=300.0, m_dot=6.0)
    residual = abs(r["Q_abs_W"] - r["Q_useful_W"] - r["Q_loss_W"])
    rel_err = residual / max(r["Q_abs_W"], 1.0)
    assert_true(rel_err < 0.01,
                f"Q_abs={r['Q_abs_W']/1000:.1f}, Q_use+Q_loss="
                f"{(r['Q_useful_W']+r['Q_loss_W'])/1000:.1f}, "
                f"rel_err={rel_err:.4f}")


# ------------------------------------------------------------------
# Test 6: T_out > T_in
# ------------------------------------------------------------------
def test_outlet_temp():
    print("\n[Test 6] HTF outlet temperature > inlet temperature")
    m, _ = make()
    r = m.solve(dni=900.0, theta_deg=0.0, T_htf_in_C=300.0, m_dot=6.0)
    assert_true(r["T_htf_out_C"] > 300.0,
                f"T_out={r['T_htf_out_C']:.1f} > T_in=300.0")
    dT = r["T_htf_out_C"] - 300.0
    assert_true(1.0 < dT < 100.0,
                f"dT={dT:.1f}C is physically reasonable (1-100 range)")


# ------------------------------------------------------------------
# Test 7: ComponentModel predict() interface
# ------------------------------------------------------------------
def test_predict_interface():
    print("\n[Test 7] ComponentModel predict() interface")
    _, cm = make()
    r = cm.predict({
        "dni": 850.0, "incidence_angle": 10.0,
        "T_htf_in": 280.0, "m_dot": 5.0, "T_ambient": 30.0,
    })
    required_keys = ["Q_useful_W", "Q_abs_W", "Q_loss_W", "T_htf_out_C",
                     "T_abs_C", "T_glass_C", "eta_thermal", "eta_optical",
                     "h_htf", "converged"]
    for k in required_keys:
        assert_true(k in r, f"Key '{k}' present in output")
    assert_true(r["Q_useful_W"] > 0, f"Q_useful={r['Q_useful_W']/1000:.1f} kW > 0")


# ------------------------------------------------------------------
# Test 8: Incidence angle effect
# ------------------------------------------------------------------
def test_incidence_angle():
    print("\n[Test 8] Higher incidence angle -> lower output")
    m, _ = make()
    r0 = m.solve(dni=900.0, theta_deg=0.0, T_htf_in_C=300.0, m_dot=6.0)
    r30 = m.solve(dni=900.0, theta_deg=30.0, T_htf_in_C=300.0, m_dot=6.0)
    r60 = m.solve(dni=900.0, theta_deg=60.0, T_htf_in_C=300.0, m_dot=6.0)
    assert_true(r0["Q_useful_W"] > r30["Q_useful_W"],
                f"Q(0deg)={r0['Q_useful_W']/1000:.0f} > Q(30deg)={r30['Q_useful_W']/1000:.0f}")
    assert_true(r30["Q_useful_W"] > r60["Q_useful_W"],
                f"Q(30deg)={r30['Q_useful_W']/1000:.0f} > Q(60deg)={r60['Q_useful_W']/1000:.0f}")


# ------------------------------------------------------------------
# Test 9: Glass temperature between ambient and absorber
# ------------------------------------------------------------------
def test_glass_temperature():
    print("\n[Test 9] Glass temperature between ambient and absorber")
    m, _ = make()
    r = m.solve(dni=900.0, theta_deg=0.0, T_htf_in_C=300.0, m_dot=6.0)
    assert_true(r["T_glass_C"] > 25.0,
                f"T_glass={r['T_glass_C']:.1f} > T_amb=25")
    assert_true(r["T_glass_C"] < r["T_abs_C"],
                f"T_glass={r['T_glass_C']:.1f} < T_abs={r['T_abs_C']:.1f}")


# ------------------------------------------------------------------
# Test 10: Benchmark timing
# ------------------------------------------------------------------
def test_benchmark():
    print("\n[Test 10] Benchmark: single solve timing")
    m, _ = make()
    t0 = time.perf_counter()
    N = 100
    for _ in range(N):
        m.solve(dni=900.0, theta_deg=5.0, T_htf_in_C=300.0, m_dot=6.0)
    elapsed = (time.perf_counter() - t0) / N
    print(f"  Single solve in {elapsed*1000:.2f} ms")
    assert_true(elapsed < 1.0, f"< 1 s per solve ({elapsed*1000:.2f} ms)")


# ==================================================================
if __name__ == "__main__":
    tests = [
        test_thermal_efficiency_typical,
        test_zero_dni,
        test_higher_temp_lower_efficiency,
        test_positive_useful_heat,
        test_energy_conservation,
        test_outlet_temp,
        test_predict_interface,
        test_incidence_angle,
        test_glass_temperature,
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
    print(f"\n{'='*60}")
    print(f"EC054 Parabolic Trough F2a -- {p} passed, {f} failed")
    print(f"{'='*60}")
    sys.exit(0 if f == 0 else 1)
