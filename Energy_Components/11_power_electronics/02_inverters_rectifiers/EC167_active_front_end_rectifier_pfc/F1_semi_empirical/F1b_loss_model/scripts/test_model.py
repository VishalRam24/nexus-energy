"""EC167 -- AFE / PFC Rectifier -- F1b Loss Model -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"v_ac_ll_rms": 400.0, "v_dc": 700.0, "p_input": 25000.0})
    expected = ["efficiency", "p_loss_w", "modulation_index", "t_j_degc",
                "p_igbt_cond_w", "p_igbt_sw_w", "p_diode_cond_w", "p_diode_rr_w"]
    for k in expected:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC167"
    assert info["fidelity"] == "F1b"


def test_efficiency_in_bounds(model):
    r = model.predict({"v_ac_ll_rms": 400.0, "v_dc": 700.0, "p_input": 25000.0})
    eta = float(r["efficiency"])
    assert 0.85 < eta < 1.0, f"Efficiency {eta:.4f} outside expected range"


def test_loss_breakdown_sums_to_total(model):
    r = model.predict({"v_ac_ll_rms": 400.0, "v_dc": 700.0, "p_input": 25000.0})
    total = (float(r["p_igbt_cond_w"]) + float(r["p_igbt_sw_w"]) +
             float(r["p_diode_cond_w"]) + float(r["p_diode_rr_w"]))
    assert abs(total - float(r["p_loss_w"])) < 1e-6, \
        f"Component sum {total:.4f} != total {float(r['p_loss_w']):.4f}"


def test_all_losses_non_negative(model):
    r = model.predict({"v_ac_ll_rms": 400.0, "v_dc": 700.0, "p_input": 25000.0})
    for k in ["p_igbt_cond_w", "p_igbt_sw_w", "p_diode_cond_w", "p_diode_rr_w"]:
        assert float(r[k]) >= 0.0, f"{k} is negative"


def test_switching_loss_scales_with_frequency(model):
    """Switching losses scale linearly with f_sw."""
    import json
    p1 = json.loads(json.dumps(model.params))
    p2 = json.loads(json.dumps(model.params))
    p1["unit"]["f_sw"]["value"] = 4000.0
    p2["unit"]["f_sw"]["value"] = 8000.0
    from model import AFERectifierF1b
    m1 = AFERectifierF1b(p1)
    m2 = AFERectifierF1b(p2)
    sw1 = float(m1.igbt_switching_loss_per_device(100.0, 700.0))
    sw2 = float(m2.igbt_switching_loss_per_device(100.0, 700.0))
    assert abs(sw2 / sw1 - 2.0) < 1e-6


def test_modulation_index_range(model):
    """Modulation index must be clamped to [0, 1.15]."""
    r = model.predict({"v_ac_ll_rms": 400.0, "v_dc": 700.0, "p_input": 1000.0})
    m = float(r["modulation_index"])
    assert 0.0 <= m <= 1.15


def test_zero_input_zero_losses(model):
    r = model.predict({"v_ac_ll_rms": 400.0, "v_dc": 700.0, "p_input": 0.0})
    assert abs(float(r["p_loss_w"])) < 1e-9


def test_junction_temperature_above_ambient(model):
    T_a = model.params["unit"]["T_a"]["value"]
    r = model.predict({"v_ac_ll_rms": 400.0, "v_dc": 700.0, "p_input": 25000.0})
    assert float(r["t_j_degc"]) > T_a


def test_losses_increase_with_power(model):
    r1 = model.predict({"v_ac_ll_rms": 400.0, "v_dc": 700.0, "p_input": 10000.0})
    r2 = model.predict({"v_ac_ll_rms": 400.0, "v_dc": 700.0, "p_input": 25000.0})
    assert float(r2["p_loss_w"]) > float(r1["p_loss_w"])


def test_vectorized(model):
    p = np.linspace(1000, 30000, 10)
    r = model.predict({"v_ac_ll_rms": 400.0, "v_dc": 700.0, "p_input": p})
    assert len(r["efficiency"]) == 10


def test_benchmark(model):
    p = np.random.uniform(1000, 30000, 1000)
    start = time.perf_counter()
    model.predict({"v_ac_ll_rms": 400.0, "v_dc": 700.0, "p_input": p})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
