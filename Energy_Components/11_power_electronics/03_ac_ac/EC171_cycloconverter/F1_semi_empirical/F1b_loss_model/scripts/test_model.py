"""EC171 -- Cycloconverter -- F1b Thyristor Loss Model -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"p_out": 400000.0, "v_out_ll_rms": 500.0})
    expected = ["efficiency", "p_loss_w", "firing_angle_deg", "t_j_degc",
                "p_conduction_w", "p_snubber_w"]
    for k in expected:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC171"
    assert info["fidelity"] == "F1b"


def test_efficiency_physical_range(model):
    """Cycloconverter efficiency typically 90-97% at rated load."""
    r = model.predict({"p_out": 400000.0, "v_out_ll_rms": 500.0})
    eta = float(r["efficiency"])
    assert 0.85 < eta < 1.0, f"Efficiency {eta:.4f} outside physical range"


def test_loss_breakdown_sums_to_total(model):
    r = model.predict({"p_out": 400000.0, "v_out_ll_rms": 500.0})
    total = float(r["p_conduction_w"]) + float(r["p_snubber_w"])
    assert abs(total - float(r["p_loss_w"])) < 1e-6


def test_firing_angle_increases_with_lower_output_voltage(model):
    """Lower output voltage → higher firing angle (cos(alpha) = V_out/V_in)."""
    r1 = model.predict({"p_out": 300000.0, "v_out_ll_rms": 600.0})
    r2 = model.predict({"p_out": 300000.0, "v_out_ll_rms": 300.0})
    assert float(r2["firing_angle_deg"]) > float(r1["firing_angle_deg"]), \
        "Lower output voltage must result in larger firing angle"


def test_conduction_loss_increases_with_current(model):
    """More current → higher conduction losses."""
    r1 = model.predict({"p_out": 200000.0, "v_out_ll_rms": 500.0})
    r2 = model.predict({"p_out": 400000.0, "v_out_ll_rms": 500.0})
    assert float(r2["p_conduction_w"]) > float(r1["p_conduction_w"])


def test_snubber_loss_constant_with_power(model):
    """Snubber loss depends only on V_in (constant for fixed supply), not on power."""
    r1 = model.predict({"p_out": 100000.0, "v_out_ll_rms": 500.0})
    r2 = model.predict({"p_out": 400000.0, "v_out_ll_rms": 500.0})
    assert abs(float(r1["p_snubber_w"]) - float(r2["p_snubber_w"])) < 1e-6, \
        "Snubber loss must be independent of output power"


def test_junction_temperature_above_ambient(model):
    T_a = model.params["unit"]["T_a"]["value"]
    r = model.predict({"p_out": 400000.0, "v_out_ll_rms": 500.0})
    assert float(r["t_j_degc"]) > T_a


def test_vectorized(model):
    p = np.linspace(50000, 500000, 10)
    r = model.predict({"p_out": p, "v_out_ll_rms": 500.0})
    assert len(r["efficiency"]) == 10


def test_zero_output_zero_conduction(model):
    """Zero output power → zero conduction loss."""
    r = model.predict({"p_out": 0.0, "v_out_ll_rms": 500.0})
    assert abs(float(r["p_conduction_w"])) < 1e-9


def test_benchmark(model):
    p = np.random.uniform(50000, 500000, 1000)
    start = time.perf_counter()
    model.predict({"p_out": p, "v_out_ll_rms": 500.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
