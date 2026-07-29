"""EC075 -- Finned-Tube HX -- F1a -- Test Suite"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"
def assert_true(c, m):
    if c: print(f"  {PASS}  {m}")
    else: print(f"  {FAIL}  FAILED: {m}"); raise AssertionError(m)


def test_predict_keys():
    print("\n[Test 1] predict() returns expected keys")
    m = ComponentModel()
    r = m.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 1.0, "m_dot_cold": 2.0})
    for k in ["Q_kw", "T_h_out", "T_c_out", "effectiveness", "NTU"]:
        assert_true(k in r, f"Key '{k}' present")


def test_get_info():
    print("\n[Test 2] get_info() metadata")
    m = ComponentModel()
    info = m.get_info()
    assert_true(info["ec_id"] == "EC075", "ec_id == EC075")


def test_positive_heat_transfer():
    print("\n[Test 3] Q > 0 when T_h > T_c")
    m = ComponentModel()
    r = m.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 1.0, "m_dot_cold": 2.0})
    assert_true(float(r["Q_kw"]) > 0, f"Q={r['Q_kw']:.2f} kW > 0")


def test_effectiveness_bounded():
    print("\n[Test 4] Effectiveness in (0, 1]")
    m = ComponentModel()
    r = m.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 1.0, "m_dot_cold": 2.0})
    eps = float(r["effectiveness"])
    assert_true(0.0 < eps <= 1.0, f"eps={eps:.4f} in (0,1]")


def test_energy_balance():
    print("\n[Test 5] Energy balance: Q_hot ~ Q_cold")
    m = ComponentModel()
    r = m.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 1.0, "m_dot_cold": 2.0})
    Q_h = 1.0 * 4180.0 * (80.0 - float(r["T_h_out"])) / 1000.0
    Q_c = 2.0 * 1006.0 * (float(r["T_c_out"]) - 20.0) / 1000.0
    assert_true(abs(Q_h - Q_c) < 0.1, f"Q_h={Q_h:.2f} ~ Q_c={Q_c:.2f}")


def test_q_increases_with_flow():
    print("\n[Test 6] Q increases with cold-side flow")
    m = ComponentModel()
    flows = [0.5, 1.0, 2.0, 4.0]
    qs = []
    for f in flows:
        r = m.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 1.0, "m_dot_cold": f})
        qs.append(float(r["Q_kw"]))
    assert_true(all(qs[i] <= qs[i+1] for i in range(len(qs)-1)),
                f"Q monotonic: {[f'{q:.1f}' for q in qs]}")


def test_benchmark():
    print("\n[Test 7] Benchmark: 1000 evaluations")
    m = ComponentModel()
    t0 = time.perf_counter()
    for _ in range(1000):
        m.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 1.0, "m_dot_cold": 2.0})
    elapsed = time.perf_counter() - t0
    print(f"  1000 calls in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "< 5 s for 1000 calls")


if __name__ == "__main__":
    tests = [test_predict_keys, test_get_info, test_positive_heat_transfer,
             test_effectiveness_bounded, test_energy_balance,
             test_q_increases_with_flow, test_benchmark]
    p = f = 0
    for t in tests:
        try: t(); p += 1
        except Exception as e: f += 1; print(f"  ERROR: {e}")
    print(f"\n{'='*50}\nEC075 Finned-Tube HX F1a -- {p} passed, {f} failed\n{'='*50}")
    sys.exit(0 if f == 0 else 1)
