"""EC160 -- Isolated DC-DC (Flyback/Forward) -- F1b Detailed Loss Model -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": 5.0})
    expected = ["duty_cycle", "v_out", "efficiency", "p_loss_w", "T_j_degC",
                "p_mosfet_cond_w", "p_diode_cond_w", "p_switching_w",
                "p_transformer_pri_w", "p_transformer_sec_w"]
    for k in expected:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC160"
    assert info["fidelity"] == "F1b"


def test_efficiency_bounds(model):
    """Efficiency must be in (0, 1) for positive load."""
    i_range = np.linspace(0.5, 10.0, 50)
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": i_range})
    assert np.all(r["efficiency"] < 1.0)
    assert np.all(r["efficiency"] > 0.0)


def test_duty_cycle_formula(model):
    """D = Vout*n / (Vin + Vout*n) for flyback."""
    n = model.params["unit"]["n_turns"]["value"]
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": 5.0})
    D_expected = 12.0 * n / (48.0 + 12.0 * n)
    assert abs(float(r["duty_cycle"]) - D_expected) < 1e-9


def test_loss_breakdown_sums_to_total(model):
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": 5.0})
    component_sum = (float(r["p_mosfet_cond_w"]) + float(r["p_diode_cond_w"]) +
                     float(r["p_switching_w"]) + float(r["p_transformer_pri_w"]) +
                     float(r["p_transformer_sec_w"]))
    total = float(r["p_loss_w"])
    assert abs(component_sum - total) < 1e-6


def test_zero_load_zero_conduction(model):
    """Zero load -> zero conduction and diode losses."""
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": 0.0})
    assert float(r["p_mosfet_cond_w"]) < 1e-9
    assert float(r["p_diode_cond_w"]) < 1e-9


def test_thermal_balance(model):
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": 8.0})
    T_a = model.params["unit"]["T_a"]["value"]
    assert float(r["T_j_degC"]) >= T_a


def test_rds_on_tempco(model):
    """Hotter ambient -> higher MOSFET conduction loss (positive tempco)."""
    import json
    from model import IsolatedDCDCF1b
    p_cold = json.loads(json.dumps(model.params))
    p_hot = json.loads(json.dumps(model.params))
    p_cold["unit"]["T_a"]["value"] = 25.0
    p_hot["unit"]["T_a"]["value"] = 75.0
    m_cold = IsolatedDCDCF1b(p_cold)
    m_hot = IsolatedDCDCF1b(p_hot)
    bd_cold = m_cold.loss_breakdown(48.0, 12.0, 5.0)
    bd_hot = m_hot.loss_breakdown(48.0, 12.0, 5.0)
    assert float(bd_hot["p_mosfet_cond_w"]) > float(bd_cold["p_mosfet_cond_w"])


def test_vectorized(model):
    v_in = np.array([24.0, 36.0, 48.0, 60.0])
    r = model.predict({"v_in": v_in, "v_out_target": 12.0, "i_load": 5.0})
    assert r["efficiency"].shape == (4,)
    assert np.all(r["efficiency"] > 0)


def test_benchmark(model):
    v_in = np.random.uniform(24, 72, 1000)
    i_load = np.random.uniform(0.1, 12.0, 1000)
    start = time.perf_counter()
    model.predict({"v_in": v_in, "v_out_target": 12.0, "i_load": i_load})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 2.0
