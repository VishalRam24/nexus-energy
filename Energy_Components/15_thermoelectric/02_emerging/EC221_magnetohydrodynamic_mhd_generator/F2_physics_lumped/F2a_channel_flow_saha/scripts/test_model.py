"""
EC221 -- MHD Generator -- F2a Physics-Lumped Channel
Test suite: physics sanity, conservation, known limits, edge cases, interface.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MHD_F2a
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
def test_emf_equals_uBd():
    print("\n[Test 1] EMF = u*B*d (motional EMF across electrodes)")
    m, _ = make_model()
    r = m.simulate(K=0.5)
    emf = r["EMF_terminal"][0]
    expect = r["u"][0] * r["B"] * m.w
    assert_true(abs(emf - expect) < 1e-6 * expect,
                f"EMF_terminal={emf:.2f} V == u*B*d={expect:.2f} V")
    # field-level identity too
    assert_true(abs(r["EMF_field"][0] - r["u"][0] * r["B"]) < 1e-9,
                "EMF_field == u*B at inlet")


def test_power_max_at_K_half():
    print("\n[Test 2] Electrical power maximised near load factor K=0.5")
    m, _ = make_model()
    Ks = np.linspace(0.1, 0.9, 17)
    Ps = np.array([m.simulate(K=k)["P_elec_W"] for k in Ks])
    k_star = Ks[int(np.argmax(Ps))]
    assert_true(abs(k_star - 0.5) < 0.06, f"argmax_K P = {k_star:.3f} ~ 0.5")
    assert_true(Ps[int(np.argmax(Ps))] > Ps[0] and Ps[int(np.argmax(Ps))] > Ps[-1],
                "Peak power exceeds endpoints")


def test_eta_electric_equals_K():
    print("\n[Test 3] Faraday channel electrical efficiency = K")
    m, _ = make_model()
    for K in [0.3, 0.5, 0.7]:
        r = m.simulate(K=K)
        assert_true(abs(r["eta_electric"] - K) < 0.02,
                    f"eta_electric={r['eta_electric']:.3f} ~ K={K}")


def test_energy_conservation():
    print("\n[Test 4] Energy conservation: stagnation enthalpy drop == electrical work")
    m, _ = make_model()
    r = m.simulate(K=0.5)
    assert_true(r["dH_W"] > 0, f"dH={r['dH_W']/1e6:.3f} MW > 0 (flow loses enthalpy)")
    ratio = r["P_elec_W"] / r["dH_W"]
    assert_true(abs(ratio - 1.0) < 0.05,
                f"P_elec/dH = {ratio:.4f} ~ 1 (first law)")


def test_efficiency_bounds():
    print("\n[Test 5] Efficiencies in physical bounds")
    m, _ = make_model()
    r = m.simulate(K=0.5)
    assert_true(0.0 < r["eta_electric"] < 1.0, f"0 < eta_electric={r['eta_electric']:.3f} < 1")
    assert_true(0.0 < r["eta_enthalpy_extraction"] < 0.5,
                f"enthalpy extraction={r['eta_enthalpy_extraction']:.4f} in (0, 0.5)")


def test_conductivity_temperature():
    print("\n[Test 6] Saha conductivity rises with temperature")
    m, _ = make_model()
    s1 = m.conductivity(2300.0, 5e5)
    s2 = m.conductivity(2800.0, 5e5)
    s3 = m.conductivity(3300.0, 5e5)
    assert_true(s1 < s2 < s3, f"sigma(T): {s1:.2f} < {s2:.2f} < {s3:.2f} S/m")
    assert_true(1.0 < s2 < 100.0, f"sigma={s2:.2f} S/m in realistic seeded range")


def test_conductivity_seeding():
    print("\n[Test 7] Conductivity rises with seed fraction")
    import json
    base = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")))
    lo = json.loads(json.dumps(base)); lo["unit"]["seed_fraction"]["value"] = 0.002
    hi = json.loads(json.dumps(base)); hi["unit"]["seed_fraction"]["value"] = 0.02
    m_lo, m_hi = MHD_F2a(lo), MHD_F2a(hi)
    s_lo = m_lo.conductivity(2800.0, 5e5)
    s_hi = m_hi.conductivity(2800.0, 5e5)
    assert_true(s_hi > s_lo, f"sigma(seed 2%)={s_hi:.2f} > sigma(seed 0.2%)={s_lo:.2f}")


def test_hall_reduces_conductivity():
    print("\n[Test 8] Hall effect reduces effective Faraday conductivity")
    m, _ = make_model()
    sigma = m.conductivity(2800.0, 5e5)
    sigma_eff = m.sigma_effective(sigma, 5.0)
    beta = m.hall_parameter(5.0)
    assert_true(sigma_eff < sigma, f"sigma_eff={sigma_eff:.2f} < sigma={sigma:.2f}")
    assert_true(abs(sigma_eff - sigma / (1 + beta**2)) < 1e-9,
                f"sigma_eff == sigma/(1+beta^2), beta={beta:.2f}")
    # higher B -> larger beta -> stronger reduction
    assert_true(m.hall_parameter(7.0) > beta, "beta grows with B")


def test_flow_decelerates_and_extracts():
    print("\n[Test 9] MHD load decelerates the flow and extracts power")
    m, _ = make_model()
    r = m.simulate(K=0.5)
    assert_true(r["u"][-1] < r["u"][0],
                f"u: {r['u'][0]:.0f} -> {r['u'][-1]:.0f} m/s (decelerates)")
    assert_true(np.all(r["J"] > 0), "Current density J > 0 along channel")
    assert_true(r["P_elec_W"] > 0, f"P_elec={r['P_elec_W']/1e6:.2f} MW > 0")
    # short-circuit limit K->0: voltage->0 so delivered power ->0
    r0 = m.simulate(K=0.02)
    assert_true(r0["P_elec_W"] < r["P_elec_W"],
                "Near short-circuit delivers less power than K=0.5")
    # open-circuit limit K->1: current->0 so power->0
    r1 = m.simulate(K=0.98)
    assert_true(r1["P_elec_W"] < r["P_elec_W"],
                "Near open-circuit delivers less power than K=0.5")


def test_field_scaling():
    print("\n[Test 10] Power scales strongly with B (sigma_eff*u^2*B^2 trend)")
    m, _ = make_model()
    p_lo = m.simulate(B=3.0, K=0.5)["P_elec_W"]
    p_hi = m.simulate(B=6.0, K=0.5)["P_elec_W"]
    assert_true(p_hi > p_lo, f"P(B=6)={p_hi/1e6:.2f} > P(B=3)={p_lo/1e6:.2f} MW")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"K_load": 0.5, "n_points": 80})
    for key in ["x", "u", "T", "p", "sigma", "J", "P_elec_W",
                "eta_electric", "beta_hall", "power_density"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["x"]) == len(r["u"]) == len(r["T"]),
                "Profile arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC221" and info["version"] == "1.0.0",
                "get_info() id/version correct")


def test_benchmark():
    print("\n[Test 12] Benchmark: full channel integration time")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(K=0.5, n_points=200)
    elapsed = time.perf_counter() - t0
    print(f"  Channel integration in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_emf_equals_uBd,
        test_power_max_at_K_half,
        test_eta_electric_equals_K,
        test_energy_conservation,
        test_efficiency_bounds,
        test_conductivity_temperature,
        test_conductivity_seeding,
        test_hall_reduces_conductivity,
        test_flow_decelerates_and_extracts,
        test_field_scaling,
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

    print(f"\n{'='*62}")
    print(f"EC221 MHD Generator F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*62}")
    sys.exit(0 if failed == 0 else 1)
