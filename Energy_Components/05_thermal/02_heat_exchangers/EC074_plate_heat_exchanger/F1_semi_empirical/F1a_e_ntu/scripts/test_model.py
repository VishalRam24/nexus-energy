"""EC074 — Plate Heat Exchanger — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 1.0, "m_dot_cold": 1.0})
    for k in ["Q_kw", "T_h_out", "T_c_out", "effectiveness", "ntu"]:
        assert k in r


def test_get_info(model):
    assert model.get_info()["ec_id"] == "EC074"


def test_effectiveness_between_0_and_1(model):
    """Effectiveness must be in [0, 1] for all valid inputs."""
    m_vals = np.linspace(0.1, 4.0, 20)
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": m_vals, "m_dot_cold": m_vals})
    assert np.all(r["effectiveness"] >= 0.0)
    assert np.all(r["effectiveness"] <= 1.0)


def test_energy_balance(model):
    """Q_hot = Q_cold within numerical tolerance."""
    T_h, T_c = 75.0, 25.0
    mh, mc = 1.5, 2.0
    r = model.predict({"T_h_in": T_h, "T_c_in": T_c, "m_dot_hot": mh, "m_dot_cold": mc})
    cp = model._model.cp_h
    Q_hot  = mh * cp * (T_h - float(r["T_h_out"])) / 1000.0
    Q_cold = mc * cp * (float(r["T_c_out"]) - T_c)  / 1000.0
    assert abs(Q_hot - Q_cold) < 0.01, f"Energy imbalance: Q_hot={Q_hot:.4f}, Q_cold={Q_cold:.4f} kW"


def test_hot_outlet_above_cold_inlet(model):
    """Counter-flow: T_h_out >= T_c_in (2nd law)."""
    T_c_in = 20.0
    r = model.predict({"T_h_in": 80.0, "T_c_in": T_c_in, "m_dot_hot": 1.0, "m_dot_cold": 1.0})
    assert float(r["T_h_out"]) >= T_c_in - 0.01, \
        f"T_h_out={float(r['T_h_out']):.2f} < T_c_in={T_c_in} — violates 2nd law"


def test_heat_transfer_positive(model):
    """Q must be positive when T_h_in > T_c_in."""
    r = model.predict({"T_h_in": 70.0, "T_c_in": 30.0, "m_dot_hot": 1.0, "m_dot_cold": 1.0})
    assert float(r["Q_kw"]) > 0.0


def test_hot_outlet_below_hot_inlet(model):
    """Hot fluid must cool down."""
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 1.0, "m_dot_cold": 1.0})
    assert float(r["T_h_out"]) < 80.0


def test_cold_outlet_above_cold_inlet(model):
    """Cold fluid must heat up."""
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 1.0, "m_dot_cold": 1.0})
    assert float(r["T_c_out"]) > 20.0


def test_q_increases_with_delta_t(model):
    """Higher temperature difference → higher heat transfer."""
    r1 = model.predict({"T_h_in": 60.0, "T_c_in": 20.0, "m_dot_hot": 1.0, "m_dot_cold": 1.0})
    r2 = model.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 1.0, "m_dot_cold": 1.0})
    assert float(r2["Q_kw"]) > float(r1["Q_kw"])


def test_ntu_positive(model):
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 1.0, "m_dot_cold": 1.0})
    assert float(r["ntu"]) > 0.0


def test_equal_capacity_rates(model):
    """C_r = 1 edge case: epsilon = NTU/(1+NTU)."""
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 1.0, "m_dot_cold": 1.0})
    NTU = float(r["ntu"])
    eps_expected = NTU / (1.0 + NTU)
    eps_actual   = float(r["effectiveness"])
    assert abs(eps_actual - eps_expected) < 1e-6, \
        f"C_r=1 case: expected eps={eps_expected:.6f}, got {eps_actual:.6f}"


def test_zero_hot_flow(model):
    """Zero hot flow rate: no heat exchange, outlets equal inlets."""
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 0.0, "m_dot_cold": 1.0})
    assert float(r["Q_kw"]) == 0.0
    assert float(r["T_h_out"]) == 80.0
    assert float(r["T_c_out"]) == 20.0
    assert float(r["effectiveness"]) == 0.0
    assert float(r["ntu"]) == 0.0


def test_zero_cold_flow(model):
    """Zero cold flow rate: no heat exchange, outlets equal inlets."""
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 1.0, "m_dot_cold": 0.0})
    assert float(r["Q_kw"]) == 0.0
    assert float(r["T_h_out"]) == 80.0
    assert float(r["T_c_out"]) == 20.0


def test_zero_both_flows(model):
    """Both flow rates zero: no heat exchange, outlets equal inlets."""
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 0.0, "m_dot_cold": 0.0})
    assert float(r["Q_kw"]) == 0.0
    assert float(r["T_h_out"]) == 80.0
    assert float(r["T_c_out"]) == 20.0
    assert not np.isnan(float(r["effectiveness"]))
    assert not np.isnan(float(r["ntu"]))


def test_zero_flow_in_array(model):
    """Zero flow mixed with nonzero in array input: no NaN in output."""
    m_hot = np.array([0.0, 1.0, 2.0])
    m_cold = np.array([1.0, 0.0, 2.0])
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": m_hot, "m_dot_cold": m_cold})
    assert not np.any(np.isnan(r["Q_kw"]))
    assert not np.any(np.isnan(r["T_h_out"]))
    assert not np.any(np.isnan(r["T_c_out"]))
    # First two entries have a zero flow, so Q must be 0
    assert float(r["Q_kw"][0]) == 0.0
    assert float(r["Q_kw"][1]) == 0.0
    # Third entry has nonzero flow and dT>0, so Q must be positive
    assert float(r["Q_kw"][2]) > 0.0


def test_benchmark(model):
    n = 1000
    Th   = np.random.uniform(50.0, 100.0, n)
    Tc   = np.random.uniform(10.0, 40.0,  n)
    mh   = np.random.uniform(0.1,  4.0,   n)
    mc   = np.random.uniform(0.1,  4.0,   n)
    start = time.perf_counter()
    model.predict({"T_h_in": Th, "T_c_in": Tc, "m_dot_hot": mh, "m_dot_cold": mc})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
