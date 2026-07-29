"""EC169 -- VFD -- F1b Loss Model -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"p_motor": 12000.0, "speed_pu": 0.8})
    expected = ["efficiency", "p_loss_w", "t_j_degc",
                "p_rectifier_w", "p_dc_link_w",
                "p_igbt_cond_w", "p_igbt_sw_w",
                "p_diode_cond_w", "p_diode_rr_w"]
    for k in expected:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC169"
    assert info["fidelity"] == "F1b"


def test_efficiency_physical_bounds(model):
    """VFD efficiency must be in (0.85, 1.0) at rated conditions (IEC IE2 class ≥ 96%)."""
    r = model.predict({"p_motor": 15000.0, "speed_pu": 1.0})
    eta = float(r["efficiency"])
    assert 0.85 < eta < 1.0, f"Efficiency {eta:.4f} outside physical range"


def test_loss_breakdown_sums_to_total(model):
    r = model.predict({"p_motor": 12000.0, "speed_pu": 0.8})
    total = (float(r["p_rectifier_w"]) + float(r["p_dc_link_w"]) +
             float(r["p_igbt_cond_w"]) + float(r["p_igbt_sw_w"]) +
             float(r["p_diode_cond_w"]) + float(r["p_diode_rr_w"]))
    assert abs(total - float(r["p_loss_w"])) < 1e-6


def test_all_losses_non_negative(model):
    r = model.predict({"p_motor": 12000.0, "speed_pu": 0.8})
    for k in ["p_rectifier_w", "p_dc_link_w", "p_igbt_cond_w",
              "p_igbt_sw_w", "p_diode_cond_w", "p_diode_rr_w"]:
        assert float(r[k]) >= 0.0, f"{k} negative"


def test_zero_power_zero_losses(model):
    r = model.predict({"p_motor": 0.0, "speed_pu": 0.0})
    assert abs(float(r["p_loss_w"])) < 1e-9


def test_losses_increase_with_power(model):
    """Total losses increase with motor power demand."""
    r1 = model.predict({"p_motor": 5000.0, "speed_pu": 0.5})
    r2 = model.predict({"p_motor": 12000.0, "speed_pu": 0.8})
    assert float(r2["p_loss_w"]) > float(r1["p_loss_w"])


def test_rectifier_loss_dominant_at_high_load(model):
    """Rectifier and inverter losses are both non-trivial -- together > dc link."""
    r = model.predict({"p_motor": 15000.0, "speed_pu": 1.0})
    assert float(r["p_rectifier_w"]) > float(r["p_dc_link_w"]), \
        "Rectifier loss should dominate DC link ESR loss"


def test_switching_loss_scales_with_frequency(model):
    """IGBT switching loss scales linearly with f_sw."""
    import json
    p1 = json.loads(json.dumps(model.params))
    p2 = json.loads(json.dumps(model.params))
    p1["unit"]["f_sw"]["value"] = 4000.0
    p2["unit"]["f_sw"]["value"] = 8000.0
    from model import VFDf1b
    m1 = VFDf1b(p1)
    m2 = VFDf1b(p2)
    sw1 = float(m1._igbt_sw_per_device(50.0))
    sw2 = float(m2._igbt_sw_per_device(50.0))
    assert abs(sw2 / sw1 - 2.0) < 1e-6


def test_junction_temperature_above_ambient(model):
    T_a = model.params["unit"]["T_a"]["value"]
    r = model.predict({"p_motor": 12000.0, "speed_pu": 0.8})
    assert float(r["t_j_degc"]) > T_a


def test_vectorized(model):
    p = np.linspace(1000, 15000, 15)
    r = model.predict({"p_motor": p, "speed_pu": 0.8})
    assert len(r["efficiency"]) == 15


def test_benchmark(model):
    p = np.random.uniform(1000, 15000, 1000)
    s = np.random.uniform(0.1, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"p_motor": p, "speed_pu": s})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
