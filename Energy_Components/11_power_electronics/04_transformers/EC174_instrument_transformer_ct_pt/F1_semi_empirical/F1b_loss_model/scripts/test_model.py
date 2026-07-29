"""EC174 -- Instrument Transformer (CT/PT) -- F1b Loss + Accuracy Model -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"input_value": 200.0})
    expected = ["p_loss_w", "p_copper_w", "p_core_w",
                "ratio_error_pct", "within_accuracy_class", "t_winding_degc"]
    for k in expected:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC174"
    assert info["fidelity"] == "F1b"


def test_loss_breakdown_sums_to_total(model):
    r = model.predict({"input_value": 200.0})
    total = float(r["p_copper_w"]) + float(r["p_core_w"])
    assert abs(total - float(r["p_loss_w"])) < 1e-9


def test_losses_non_negative(model):
    r = model.predict({"input_value": 200.0})
    assert float(r["p_copper_w"]) >= 0.0
    assert float(r["p_core_w"]) >= 0.0


def test_zero_current_zero_copper_loss(model):
    """At zero primary current, CT copper loss = 0."""
    r = model.predict({"input_value": 0.0})
    assert abs(float(r["p_copper_w"])) < 1e-9


def test_copper_loss_scales_with_i_squared(model):
    """CT copper loss: P_cu = I2^2 * R2 → quadratic with I1."""
    r1 = model.predict({"input_value": 100.0})
    r2 = model.predict({"input_value": 200.0})
    ratio = float(r2["p_copper_w"]) / float(r1["p_copper_w"])
    assert abs(ratio - 4.0) < 0.01, f"Copper loss ratio {ratio:.4f}, expected 4.0"


def test_ratio_error_at_rated_within_class(model):
    """At rated primary current, CT ratio error must be within accuracy class."""
    I_rated = model.params["unit"]["I_rated"]["value"]
    r = model.predict({"input_value": I_rated})
    assert bool(r["within_accuracy_class"]), \
        f"Ratio error {float(r['ratio_error_pct']):.4f}% exceeds accuracy class " \
        f"{model.params['unit']['accuracy_class']['value']}%"


def test_ratio_error_small(model):
    """Ratio error at rated load must be less than 1% for a class 0.5 CT."""
    I_rated = model.params["unit"]["I_rated"]["value"]
    r = model.predict({"input_value": I_rated})
    assert abs(float(r["ratio_error_pct"])) < 1.0, \
        f"Ratio error {float(r['ratio_error_pct']):.4f}% should be < 1% for class 0.5"


def test_winding_temperature_above_ambient(model):
    T_a = model.params["unit"]["T_a"]["value"]
    r = model.predict({"input_value": 200.0})
    assert float(r["t_winding_degc"]) > T_a


def test_losses_increase_with_current(model):
    r1 = model.predict({"input_value": 50.0})
    r2 = model.predict({"input_value": 200.0})
    assert float(r2["p_loss_w"]) > float(r1["p_loss_w"])


def test_vectorized(model):
    i_primary = np.linspace(20, 400, 20)
    r = model.predict({"input_value": i_primary})
    assert len(r["p_loss_w"]) == 20


def test_benchmark(model):
    x = np.random.uniform(20, 400, 1000)
    start = time.perf_counter()
    model.predict({"input_value": x})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
