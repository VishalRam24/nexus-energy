"""EC185 — SVC — F1a Reactive Compensation — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"Q_demand_MVAR": 50.0})
    for k in ["Q_out_MVAR", "Q_limited", "P_loss_MW", "operating_mode", "utilization"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC185"
    assert info["fidelity"] == "F1a"


def test_Q_out_within_range(model):
    """Q_out must be within [Q_min, Q_max]."""
    Q_dem = np.linspace(-100, 200, 200)
    r = model.predict({"Q_demand_MVAR": Q_dem})
    Q_min = model._model.Q_min
    Q_max = model._model.Q_max
    assert np.all(r["Q_out_MVAR"] >= Q_min - 1e-9)
    assert np.all(r["Q_out_MVAR"] <= Q_max + 1e-9)


def test_positive_demand_gives_positive_Q(model):
    """Capacitive demand → positive Q_out."""
    r = model.predict({"Q_demand_MVAR": 60.0})
    assert float(r["Q_out_MVAR"]) > 0.0


def test_negative_demand_gives_negative_Q(model):
    """Inductive demand → negative Q_out (within inductive range)."""
    r = model.predict({"Q_demand_MVAR": -30.0})
    assert float(r["Q_out_MVAR"]) < 0.0


def test_capping_above_Q_max(model):
    """Demand above Q_max → Q_out = Q_max."""
    Q_max = model._model.Q_max
    r = model.predict({"Q_demand_MVAR": Q_max + 50.0})
    assert abs(float(r["Q_out_MVAR"]) - Q_max) < 1e-9
    assert bool(r["Q_limited"]) is True


def test_capping_below_Q_min(model):
    """Demand below Q_min → Q_out = Q_min."""
    Q_min = model._model.Q_min
    r = model.predict({"Q_demand_MVAR": Q_min - 30.0})
    assert abs(float(r["Q_out_MVAR"]) - Q_min) < 1e-9
    assert bool(r["Q_limited"]) is True


def test_no_limiting_in_range(model):
    """Demand within range → Q_limited = False, Q_out = Q_demand."""
    r = model.predict({"Q_demand_MVAR": 40.0})
    assert bool(r["Q_limited"]) is False
    assert abs(float(r["Q_out_MVAR"]) - 40.0) < 1e-9


def test_P_loss_positive(model):
    """Losses must be positive for nonzero Q_out."""
    r = model.predict({"Q_demand_MVAR": 50.0})
    assert float(r["P_loss_MW"]) > 0.0


def test_P_loss_zero_at_standby(model):
    """Zero Q_out → zero losses."""
    r = model.predict({"Q_demand_MVAR": 0.0})
    assert float(r["P_loss_MW"]) == 0.0


def test_loss_scales_with_abs_Q(model):
    """P_loss proportional to |Q_out|."""
    r1 = model.predict({"Q_demand_MVAR": 30.0})
    r2 = model.predict({"Q_demand_MVAR": 60.0})
    ratio = float(r2["P_loss_MW"]) / float(r1["P_loss_MW"])
    assert abs(ratio - 2.0) < 1e-9, f"Loss ratio={ratio:.6f}, expected 2.0"


def test_capacitive_mode(model):
    r = model.predict({"Q_demand_MVAR": 50.0})
    assert r["operating_mode"] == "capacitive"


def test_inductive_mode(model):
    r = model.predict({"Q_demand_MVAR": -30.0})
    assert r["operating_mode"] == "inductive"


def test_vectorized_input(model):
    Q = np.linspace(-50, 100, 100)
    r = model.predict({"Q_demand_MVAR": Q})
    assert r["Q_out_MVAR"].shape == (100,)
    assert np.all(np.abs(r["Q_out_MVAR"]) <= max(abs(model._model.Q_min), model._model.Q_max) + 1e-9)


def test_benchmark(model):
    Q = np.random.uniform(-50, 100, 1000)
    start = time.perf_counter()
    model.predict({"Q_demand_MVAR": Q})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
