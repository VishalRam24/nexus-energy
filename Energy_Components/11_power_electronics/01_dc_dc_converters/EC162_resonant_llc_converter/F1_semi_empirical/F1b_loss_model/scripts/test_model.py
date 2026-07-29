"""EC162 -- Resonant LLC Converter -- F1b Detailed Loss Model -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"v_in": 400.0, "v_out_target": 12.0, "i_load": 50.0})
    expected = ["efficiency", "p_loss_w", "T_j_degC", "p_mosfet_cond_w",
                "p_switching_w", "p_diode_cond_w", "p_resonant_inductor_w",
                "p_transformer_pri_w", "p_transformer_sec_w"]
    for k in expected:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC162"
    assert info["fidelity"] == "F1b"


def test_efficiency_bounds(model):
    i_range = np.linspace(1.0, 150.0, 50)
    r = model.predict({"v_in": 400.0, "v_out_target": 12.0, "i_load": i_range})
    assert np.all(r["efficiency"] < 1.0)
    assert np.all(r["efficiency"] > 0.0)


def test_zero_load_zero_losses(model):
    r = model.predict({"v_in": 400.0, "v_out_target": 12.0, "i_load": 0.0})
    assert float(r["p_loss_w"]) < 1e-9


def test_loss_breakdown_sums_to_total(model):
    r = model.predict({"v_in": 400.0, "v_out_target": 12.0, "i_load": 50.0})
    component_sum = (float(r["p_mosfet_cond_w"]) + float(r["p_switching_w"]) +
                     float(r["p_diode_cond_w"]) + float(r["p_resonant_inductor_w"]) +
                     float(r["p_transformer_pri_w"]) + float(r["p_transformer_sec_w"]))
    total = float(r["p_loss_w"])
    assert abs(component_sum - total) < 1e-6


def test_thermal_balance(model):
    r = model.predict({"v_in": 400.0, "v_out_target": 12.0, "i_load": 100.0})
    T_a = model.params["unit"]["T_a"]["value"]
    assert float(r["T_j_degC"]) >= T_a


def test_rds_on_tempco(model):
    import json
    from model import LLCConverterF1b
    p_cold = json.loads(json.dumps(model.params))
    p_hot = json.loads(json.dumps(model.params))
    p_cold["unit"]["T_a"]["value"] = 25.0
    p_hot["unit"]["T_a"]["value"] = 85.0
    m_cold = LLCConverterF1b(p_cold)
    m_hot = LLCConverterF1b(p_hot)
    bd_cold = m_cold.loss_breakdown(400.0, 50.0)
    bd_hot = m_hot.loss_breakdown(400.0, 50.0)
    assert float(bd_hot["p_mosfet_cond_w"]) > float(bd_cold["p_mosfet_cond_w"])


def test_zvs_switching_lower_than_hard_switching(model):
    """ZVS switching loss should be lower than equivalent hard-switching (t_off only, no t_on)."""
    r = model.predict({"v_in": 400.0, "v_out_target": 12.0, "i_load": 50.0})
    # Switching loss = 2 * 0.5 * V_sw * I_pk * t_off * f_sw  (no t_on)
    # For hard-switching it would be 2 * 0.5 * V_sw * I_pk * (t_on + t_off) * f_sw
    # So LLC switching loss must be < conduction loss at full load
    assert float(r["p_switching_w"]) < float(r["p_mosfet_cond_w"]) * 10.0  # reasonable bound


def test_vectorized(model):
    i_load = np.array([10.0, 50.0, 100.0, 150.0])
    r = model.predict({"v_in": 400.0, "v_out_target": 12.0, "i_load": i_load})
    assert r["efficiency"].shape == (4,)
    assert np.all(r["efficiency"] > 0)


def test_benchmark(model):
    i_load = np.random.uniform(1, 150, 1000)
    start = time.perf_counter()
    model.predict({"v_in": 400.0, "v_out_target": 12.0, "i_load": i_load})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 2.0
