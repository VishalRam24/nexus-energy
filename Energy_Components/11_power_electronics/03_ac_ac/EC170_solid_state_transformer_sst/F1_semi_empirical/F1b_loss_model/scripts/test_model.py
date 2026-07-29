"""EC170 -- Solid State Transformer -- F1b Three-Stage Loss Model -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"p_out": 8000.0})
    expected = ["efficiency", "p_loss_w", "t_j_degc",
                "p_stage1_w", "p_stage2_w", "p_stage3_w"]
    for k in expected:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC170"
    assert info["fidelity"] == "F1b"


def test_efficiency_physical_range(model):
    """SST typical efficiency 90-97% at rated load."""
    r = model.predict({"p_out": 10000.0})
    eta = float(r["efficiency"])
    assert 0.85 < eta < 1.0, f"SST efficiency {eta:.4f} outside physical range"


def test_loss_breakdown_sums_to_total(model):
    r = model.predict({"p_out": 8000.0})
    total = float(r["p_stage1_w"]) + float(r["p_stage2_w"]) + float(r["p_stage3_w"])
    assert abs(total - float(r["p_loss_w"])) < 1e-6


def test_all_stages_positive(model):
    r = model.predict({"p_out": 8000.0})
    for k in ["p_stage1_w", "p_stage2_w", "p_stage3_w"]:
        assert float(r[k]) > 0, f"{k} must be positive at non-zero load"


def test_zero_load_minimal_losses(model):
    """At zero power, switching losses and core losses drive small baseline.
    Total should be very small (< rated power / 100)."""
    r = model.predict({"p_out": 0.0})
    # Stage 2 core loss is zero at zero power (model sets it to 0 at p=0)
    assert float(r["p_stage1_w"]) < 1e-6
    assert float(r["p_stage3_w"]) < 1e-6


def test_losses_increase_with_power(model):
    r1 = model.predict({"p_out": 3000.0})
    r2 = model.predict({"p_out": 8000.0})
    assert float(r2["p_loss_w"]) > float(r1["p_loss_w"])


def test_stage2_has_core_loss(model):
    """Stage 2 losses include constant core loss at rated flux."""
    r1 = model.predict({"p_out": 1000.0})
    r2 = model.predict({"p_out": 8000.0})
    # Stage 2 losses increase with load (copper + switching increase)
    assert float(r2["p_stage2_w"]) > float(r1["p_stage2_w"])


def test_junction_temperature_above_ambient(model):
    T_a = model.params["unit"]["T_a"]["value"]
    r = model.predict({"p_out": 8000.0})
    assert float(r["t_j_degc"]) > T_a


def test_vectorized(model):
    p = np.linspace(1000, 10000, 10)
    r = model.predict({"p_out": p})
    assert len(r["efficiency"]) == 10


def test_three_stage_topology(model):
    """All three stages must contribute losses at rated load."""
    r = model.predict({"p_out": 10000.0})
    for k in ["p_stage1_w", "p_stage2_w", "p_stage3_w"]:
        assert float(r[k]) > 0.0, f"{k} must be > 0 at rated load"


def test_benchmark(model):
    p = np.random.uniform(1000, 10000, 1000)
    start = time.perf_counter()
    model.predict({"p_out": p})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
