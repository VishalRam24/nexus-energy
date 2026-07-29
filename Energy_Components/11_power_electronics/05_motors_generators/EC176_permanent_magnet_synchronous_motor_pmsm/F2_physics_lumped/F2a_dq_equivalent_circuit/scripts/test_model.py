"""
EC176 -- PMSM -- F2a dq-Frame Dynamic Model -- Test suite.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import PMSMF2a
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"
def assert_true(c, m):
    if c: print(f"  {PASS}  {m}")
    else: print(f"  {FAIL}  FAILED: {m}"); raise AssertionError(m)

def make():
    cm = ComponentModel()
    return cm._model, cm


def test_speed_control():
    print("\n[Test 1] Speed control: command 1000 rpm, verify convergence within 5%")
    m, _ = make()
    # Use 1000 rpm (well within voltage limit: back-EMF ~ 125.7 V < 173 V)
    r = m.simulate_speed_control(1000.0, T_load_Nm=5.0, dt=1e-4, duration_s=3.0)
    final_speed = r["speed_rpm"][-1]
    target = 1000.0
    error_pct = abs(final_speed - target) / target * 100.0
    assert_true(error_pct < 5.0,
                f"Final speed={final_speed:.1f} rpm within 5% of {target} (err={error_pct:.2f}%)")


def test_direct_voltage():
    print("\n[Test 2] Direct voltage: apply v_q=50V, motor spins")
    m, _ = make()
    r = m.simulate_direct(v_d=0.0, v_q=50.0, T_load_Nm=0.0, dt=1e-4, duration_s=1.0)
    final_speed = r["speed_rpm"][-1]
    assert_true(final_speed > 10.0,
                f"Motor spinning at {final_speed:.1f} rpm > 10 rpm")


def test_torque_equation():
    print("\n[Test 3] Torque: T_e = 1.5*P*Phi_m*i_q for surface-mount PMSM")
    m, _ = make()
    r = m.simulate_speed_control(1000.0, T_load_Nm=10.0, dt=1e-4, duration_s=2.0)
    # At steady state the torque should match the formula
    idx = -1  # last sample
    i_q = r["i_q"][idx]
    i_d = r["i_d"][idx]
    T_e_sim = r["torque"][idx]
    T_e_calc = 1.5 * m.P * (m.Phi_m * i_q + (m.Ld - m.Lq) * i_d * i_q)
    err = abs(T_e_sim - T_e_calc)
    assert_true(err < 0.01,
                f"T_e_sim={T_e_sim:.4f} matches formula={T_e_calc:.4f} (err={err:.6f})")


def test_zero_load():
    print("\n[Test 4] Zero load: motor accelerates freely")
    m, _ = make()
    r = m.simulate_direct(v_d=0.0, v_q=50.0, T_load_Nm=0.0, dt=1e-4, duration_s=2.0)
    final_speed = r["speed_rpm"][-1]
    # With v_q=50V and no load, motor should reach substantial speed
    assert_true(final_speed > 100.0,
                f"No-load speed={final_speed:.1f} rpm > 100 rpm")
    # Speed should be increasing or near steady state (not zero)
    mid = len(r["speed_rpm"]) // 2
    assert_true(r["speed_rpm"][-1] >= r["speed_rpm"][mid] * 0.95,
                "Speed non-decreasing in second half")


def test_load_step():
    print("\n[Test 5] Load step: speed drops then recovers under PI control")
    m, _ = make()
    # Use 1000 rpm (within voltage limit). Load step at t=2.0s (after settling)
    def T_load(t):
        return 2.0 if t < 2.0 else 8.0

    r = m.simulate_speed_control(1000.0, T_load_Nm=T_load, dt=1e-4, duration_s=5.0)
    # Find speed just before and after load step
    t = r["t"]
    speed = r["speed_rpm"]
    idx_pre = np.searchsorted(t, 1.9)
    idx_dip = np.searchsorted(t, 2.3)
    idx_final = -1

    speed_pre = speed[idx_pre]
    speed_dip = speed[idx_dip]
    speed_final = speed[idx_final]

    # Speed should dip after load increase
    assert_true(speed_dip < speed_pre,
                f"Speed dips: {speed_dip:.1f} < {speed_pre:.1f} rpm after load step")
    # Speed should recover close to reference
    error_pct = abs(speed_final - 1000.0) / 1000.0 * 100.0
    assert_true(error_pct < 5.0,
                f"Speed recovers to {speed_final:.1f} rpm (err={error_pct:.2f}%)")


def test_predict_interface():
    print("\n[Test 6] ComponentModel predict() interface")
    _, cm = make()
    r = cm.predict({"mode": "speed_control", "speed_ref_rpm": 1000.0,
                     "T_load_Nm": 2.0, "dt": 1e-4, "duration_s": 0.5})
    for k in ["t", "speed_rpm", "torque", "i_d", "i_q", "power", "omega_m"]:
        assert_true(k in r, f"Key '{k}' present")
    assert_true(len(r["t"]) > 100, f"Sufficient data points: {len(r['t'])}")


def test_get_info():
    print("\n[Test 7] ComponentModel get_info()")
    _, cm = make()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC176", f"component_id={info['component_id']}")
    assert_true("PMSM" in info["component_name"], f"component_name={info['component_name']}")
    assert_true("inputs" in info, "inputs dict present")
    assert_true("outputs" in info, "outputs dict present")


def test_benchmark():
    print("\n[Test 8] Benchmark: speed control simulation")
    m, _ = make()
    t0 = time.perf_counter()
    N = 3
    for _ in range(N):
        m.simulate_speed_control(1500.0, T_load_Nm=5.0, dt=5e-4, duration_s=0.5)
    elapsed = (time.perf_counter() - t0) / N
    print(f"  Single simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 30, "< 30 s per simulation")


if __name__ == "__main__":
    tests = [test_speed_control, test_direct_voltage, test_torque_equation,
             test_zero_load, test_load_step, test_predict_interface,
             test_get_info, test_benchmark]
    p = f = 0
    for t in tests:
        try: t(); p += 1
        except Exception as e: f += 1; print(f"  ERROR: {e}")
    print(f"\n{'='*60}\nEC176 PMSM F2a -- {p} passed, {f} failed\n{'='*60}")
    sys.exit(0 if f == 0 else 1)
