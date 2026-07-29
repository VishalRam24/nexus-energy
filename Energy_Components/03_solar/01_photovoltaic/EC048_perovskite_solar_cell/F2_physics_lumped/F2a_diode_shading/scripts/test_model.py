"""
EC048 -- Perovskite Solar Cell -- F2a Diode Shading -- Test suite.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import PerovskitePV_DiodeShading_F2a
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"
def assert_true(c, m):
    if c: print(f"  {PASS}  {m}")
    else: print(f"  {FAIL}  FAILED: {m}"); raise AssertionError(m)

N_S = 24  # perovskite module cell count

def make():
    cm = ComponentModel()
    return cm._model, cm

def test_uniform_stc():
    print("\n[Test 1] Uniform STC: P_mp in reasonable range for perovskite module")
    m, _ = make()
    G = np.full(N_S, 1000.0)
    r = m.iv_curve(G, 25.0)
    # 24 cells * ~1.0V * ~1.8A ~ 43 W; expect 30-60 W range
    assert_true(20 < r["P_mp"] < 80, f"P_mp={r['P_mp']:.1f} W in [20,80]")
    assert_true(r["num_local_maxima"] <= 1, "Single MPP for uniform irradiance")
    # Check Voc is consistent with perovskite (24 cells * ~1.1 V ~ 26 V)
    assert_true(r["V"][0] > 15, f"Voc={r['V'][0]:.1f} V > 15 V (perovskite Ns=24)")

def test_zero_irradiance():
    print("\n[Test 2] Zero irradiance -> ~0 power")
    m, _ = make()
    G = np.full(N_S, 0.0)
    r = m.iv_curve(G, 25.0)
    assert_true(r["P_mp"] < 0.1, f"P_mp={r['P_mp']:.4f} ~ 0")

def test_partial_shading_loss():
    print("\n[Test 3] Partial shading causes loss")
    m, _ = make()
    G_uniform = np.full(N_S, 1000.0)
    G_shaded = np.full(N_S, 1000.0)
    G_shaded[0:6] = 200.0  # shade 6 of 24 cells (25%)
    r_u = m.iv_curve(G_uniform, 25.0)
    r_s = m.iv_curve(G_shaded, 25.0)
    assert_true(r_s["P_mp"] < r_u["P_mp"], "Shaded P < Uniform P")
    assert_true(r_s["shading_loss_pct"] > 0, f"Shading loss={r_s['shading_loss_pct']:.1f}%")

def test_multiple_mpp():
    print("\n[Test 4] Heavy shading may create multiple local maxima")
    m, _ = make()
    G = np.full(N_S, 1000.0)
    G[0:8] = 100.0  # shade 1/3 of cells heavily
    r = m.iv_curve(G, 25.0, N_points=500)
    print(f"  Local maxima found: {r['num_local_maxima']}")
    assert_true(r["P_mp"] > 0, "P_mp > 0 even with shading")

def test_temperature_effect():
    print("\n[Test 5] Higher T reduces power")
    m, _ = make()
    G = np.full(N_S, 1000.0)
    r_cool = m.iv_curve(G, 15.0)
    r_hot = m.iv_curve(G, 55.0)
    assert_true(r_cool["P_mp"] > r_hot["P_mp"],
                f"P(15C)={r_cool['P_mp']:.1f} > P(55C)={r_hot['P_mp']:.1f}")

def test_iv_shape():
    print("\n[Test 6] I-V curve shape: V decreases as I increases")
    m, _ = make()
    G = np.full(N_S, 1000.0)
    r = m.iv_curve(G, 25.0)
    # V should generally decrease (V at I=0 > V at I=Isc)
    assert_true(r["V"][0] > r["V"][-1], "V decreases from I=0 to I=Isc")

def test_predict_interface():
    print("\n[Test 7] ComponentModel predict()")
    _, cm = make()
    r = cm.predict({"irradiance_per_cell": 800.0, "temperature_degC": 30.0})
    for k in ["I", "V", "P", "P_mp", "num_local_maxima", "shading_loss_pct"]:
        assert_true(k in r, f"Key '{k}'")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC048", f"component_id={info['component_id']}")
    assert_true("F2a" in info["fidelity"], f"fidelity={info['fidelity']}")

def test_benchmark():
    print("\n[Test 8] Benchmark: uniform I-V curve")
    m, _ = make()
    G = np.full(N_S, 1000.0)
    t0 = time.perf_counter()
    for _ in range(10):
        m.iv_curve(G, 25.0)
    elapsed = (time.perf_counter() - t0) / 10
    print(f"  Single I-V curve in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 30, "< 30 s per curve")

if __name__ == "__main__":
    tests = [test_uniform_stc, test_zero_irradiance, test_partial_shading_loss,
             test_multiple_mpp, test_temperature_effect, test_iv_shape,
             test_predict_interface, test_benchmark]
    p = f = 0
    for t in tests:
        try: t(); p += 1
        except Exception as e: f += 1; print(f"  ERROR: {e}")
    print(f"\n{'='*60}\nEC048 Perovskite PV F2a -- {p} passed, {f} failed\n{'='*60}")
    sys.exit(0 if f == 0 else 1)
