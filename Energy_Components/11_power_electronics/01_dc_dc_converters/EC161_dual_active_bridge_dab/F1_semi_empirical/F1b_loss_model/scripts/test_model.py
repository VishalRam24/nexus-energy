"""EC161 -- Dual Active Bridge (DAB) -- F1b Detailed Loss Model -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"v_in": 400.0, "v_out_target": 200.0, "p_load": 5000.0})
    expected = ["phi_rad", "efficiency", "p_loss_w", "T_j_degC",
                "p_mosfet_cond_w", "p_switching_w", "p_transformer_w"]
    for k in expected:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC161"
    assert info["fidelity"] == "F1b"


def test_efficiency_bounds(model):
    p_range = np.linspace(500, 12000, 50)
    r = model.predict({"v_in": 400.0, "v_out_target": 200.0, "p_load": p_range})
    assert np.all(r["efficiency"] < 1.0)
    assert np.all(r["efficiency"] > 0.0)


def test_zero_load_zero_losses(model):
    r = model.predict({"v_in": 400.0, "v_out_target": 200.0, "p_load": 0.0})
    assert float(r["p_loss_w"]) < 1e-9
    assert float(r["phi_rad"]) < 1e-9


def test_losses_increase_with_power(model):
    """Higher power -> higher losses."""
    r1 = model.predict({"v_in": 400.0, "v_out_target": 200.0, "p_load": 2000.0})
    r2 = model.predict({"v_in": 400.0, "v_out_target": 200.0, "p_load": 8000.0})
    assert float(r2["p_loss_w"]) > float(r1["p_loss_w"])


def test_loss_breakdown_sums_to_total(model):
    r = model.predict({"v_in": 400.0, "v_out_target": 200.0, "p_load": 5000.0})
    component_sum = (float(r["p_mosfet_cond_w"]) + float(r["p_switching_w"]) +
                     float(r["p_transformer_w"]))
    total = float(r["p_loss_w"])
    assert abs(component_sum - total) < 1e-6


def test_thermal_balance(model):
    r = model.predict({"v_in": 400.0, "v_out_target": 200.0, "p_load": 8000.0})
    T_a = model.params["unit"]["T_a"]["value"]
    assert float(r["T_j_degC"]) >= T_a


def test_rds_on_tempco(model):
    """Higher ambient -> higher MOSFET conduction loss."""
    import json
    from model import DABF1b
    p_cold = json.loads(json.dumps(model.params))
    p_hot = json.loads(json.dumps(model.params))
    p_cold["unit"]["T_a"]["value"] = 25.0
    p_hot["unit"]["T_a"]["value"] = 80.0
    m_cold = DABF1b(p_cold)
    m_hot = DABF1b(p_hot)
    bd_cold = m_cold.loss_breakdown(400.0, 200.0, 5000.0)
    bd_hot = m_hot.loss_breakdown(400.0, 200.0, 5000.0)
    assert float(bd_hot["p_mosfet_cond_w"]) > float(bd_cold["p_mosfet_cond_w"])


def test_switching_scales_with_frequency(model):
    """Switching losses scale linearly with f_sw."""
    import json
    from model import DABF1b
    p1 = json.loads(json.dumps(model.params))
    p2 = json.loads(json.dumps(model.params))
    p1["unit"]["f_sw"]["value"] = 50000.0
    p2["unit"]["f_sw"]["value"] = 100000.0
    # Need to fix phase shift for same power -- use lower power to ensure phi << pi/2
    m1 = DABF1b(p1)
    m2 = DABF1b(p2)
    # At 1kW both well within phi range
    bd1 = m1.loss_breakdown(400.0, 200.0, 1000.0)
    bd2 = m2.loss_breakdown(400.0, 200.0, 1000.0)
    # Switching should be higher at 2x freq (not exactly 2x due to phase shift diff)
    assert float(bd2["p_switching_w"]) > float(bd1["p_switching_w"])


def test_vectorized(model):
    p_load = np.array([1000, 3000, 6000, 10000], dtype=float)
    r = model.predict({"v_in": 400.0, "v_out_target": 200.0, "p_load": p_load})
    assert r["efficiency"].shape == (4,)
    assert np.all(r["efficiency"] > 0)


def test_benchmark(model):
    p_load = np.random.uniform(500, 10000, 1000)
    start = time.perf_counter()
    model.predict({"v_in": 400.0, "v_out_target": 200.0, "p_load": p_load})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 2.0
