"""EC073 — Shell-and-Tube HX — F1b Fouling — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_h_in": 90.0, "T_c_in": 20.0,
                       "m_dot_hot": 5.0, "m_dot_cold": 5.0})
    for k in ["Q_kw", "T_h_out", "T_c_out", "effectiveness", "ntu",
              "UA_effective", "cleanliness_factor", "effectiveness_reduction"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC073"
    assert info["fidelity"] == "F1b"


def test_clean_case_no_fouling(model):
    """At Rf=0, effectiveness_reduction must be 0 and cleanliness_factor=1."""
    r = model.predict({"T_h_in": 90.0, "T_c_in": 20.0,
                       "m_dot_hot": 5.0, "m_dot_cold": 5.0,
                       "Rf_shell": 0.0, "Rf_tube": 0.0})
    np.testing.assert_allclose(float(r["effectiveness_reduction"]), 0.0, atol=1e-10)
    np.testing.assert_allclose(float(r["cleanliness_factor"]), 1.0, atol=1e-10)


def test_fouling_reduces_Q(model):
    """Fouling must reduce heat transfer rate."""
    r_clean = model.predict({"T_h_in": 90.0, "T_c_in": 20.0,
                             "m_dot_hot": 5.0, "m_dot_cold": 5.0,
                             "Rf_shell": 0.0, "Rf_tube": 0.0})
    r_fouled = model.predict({"T_h_in": 90.0, "T_c_in": 20.0,
                              "m_dot_hot": 5.0, "m_dot_cold": 5.0,
                              "Rf_shell": 0.001, "Rf_tube": 0.001})
    assert float(r_fouled["Q_kw"]) < float(r_clean["Q_kw"]), \
        "Fouling must reduce Q"


def test_fouling_reduces_effectiveness_monotonically(model):
    """Effectiveness must decrease monotonically as fouling resistance increases."""
    Rf_arr = np.linspace(0.0, 0.002, 20)
    r = model.predict({"T_h_in": 90.0, "T_c_in": 20.0,
                       "m_dot_hot": 5.0, "m_dot_cold": 5.0,
                       "Rf_shell": Rf_arr, "Rf_tube": Rf_arr})
    assert np.all(np.diff(r["effectiveness"]) <= 0), \
        f"Effectiveness not monotonically decreasing with Rf: {r['effectiveness']}"


def test_cleanliness_factor_decreases_with_fouling(model):
    """Cleanliness factor must be < 1 for any fouling."""
    r = model.predict({"T_h_in": 90.0, "T_c_in": 20.0,
                       "m_dot_hot": 5.0, "m_dot_cold": 5.0,
                       "Rf_shell": 0.0002, "Rf_tube": 0.0002})
    assert float(r["cleanliness_factor"]) < 1.0


def test_UA_effective_formula(model):
    """1/UA_eff = 1/UA_0 + Rf_shell + Rf_tube — verify numerically."""
    from model import ShellTubeHXF1b
    import json
    params_path = Path(__file__).parent.parent / "data" / "parameters.json"
    with open(params_path) as f:
        params = json.load(f)
    m = ShellTubeHXF1b(params)
    Rf_s, Rf_t = 0.0003, 0.0002
    UA_0   = m.U_clean * m.A
    UA_exp = UA_0 / (1.0 + UA_0 * (Rf_s + Rf_t))
    UA_got = float(m.UA_effective(Rf_s, Rf_t))
    np.testing.assert_allclose(UA_got, UA_exp, rtol=1e-10)


def test_energy_balance(model):
    """Q from hot side must equal Q from cold side (within 0.1%)."""
    r = model.predict({"T_h_in": 90.0, "T_c_in": 20.0,
                       "m_dot_hot": 5.0, "m_dot_cold": 6.0,
                       "Rf_shell": 0.0002, "Rf_tube": 0.0001})
    Q   = float(r["Q_kw"]) * 1000.0
    T_h_out = float(r["T_h_out"])
    T_c_out = float(r["T_c_out"])
    Q_hot  = 5.0 * 4186.0 * (90.0 - T_h_out)
    Q_cold = 6.0 * 4186.0 * (T_c_out - 20.0)
    np.testing.assert_allclose(Q_hot, Q_cold, rtol=1e-5)
    np.testing.assert_allclose(Q, Q_hot, rtol=1e-5)


def test_hot_outlet_below_inlet(model):
    """Hot outlet temperature must be less than hot inlet."""
    r = model.predict({"T_h_in": 90.0, "T_c_in": 20.0,
                       "m_dot_hot": 5.0, "m_dot_cold": 5.0})
    assert float(r["T_h_out"]) < 90.0
    assert float(r["T_c_out"]) > 20.0


def test_second_law_not_violated(model):
    """Hot outlet must always be above cold inlet; cold outlet below hot inlet."""
    T_h_in, T_c_in = 90.0, 20.0
    r = model.predict({"T_h_in": T_h_in, "T_c_in": T_c_in,
                       "m_dot_hot": 5.0, "m_dot_cold": 5.0})
    assert float(r["T_h_out"]) >= T_c_in, \
        f"T_h_out={float(r['T_h_out']):.1f} below T_c_in={T_c_in}"
    assert float(r["T_c_out"]) <= T_h_in, \
        f"T_c_out={float(r['T_c_out']):.1f} above T_h_in={T_h_in}"


def test_higher_fouling_gives_larger_effectiveness_reduction(model):
    """More fouling → larger effectiveness_reduction."""
    r_low  = model.predict({"T_h_in": 90.0, "T_c_in": 20.0,
                            "m_dot_hot": 5.0, "m_dot_cold": 5.0,
                            "Rf_shell": 0.0001, "Rf_tube": 0.0001})
    r_high = model.predict({"T_h_in": 90.0, "T_c_in": 20.0,
                            "m_dot_hot": 5.0, "m_dot_cold": 5.0,
                            "Rf_shell": 0.001, "Rf_tube": 0.001})
    assert float(r_high["effectiveness_reduction"]) > float(r_low["effectiveness_reduction"])


def test_benchmark(model):
    rng = np.random.default_rng(7)
    T_h  = rng.uniform(50, 180, 1000)
    T_c  = rng.uniform(10, 60, 1000)
    mh   = rng.uniform(0.5, 30, 1000)
    mc   = rng.uniform(0.5, 30, 1000)
    Rfs  = rng.uniform(0.0, 0.002, 1000)
    Rft  = rng.uniform(0.0, 0.002, 1000)
    start = time.perf_counter()
    model.predict({"T_h_in": T_h, "T_c_in": T_c,
                   "m_dot_hot": mh, "m_dot_cold": mc,
                   "Rf_shell": Rfs, "Rf_tube": Rft})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
