"""EC186 — STATCOM — F1a Reactive Compensation — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"Q_demand_MVAR": 60.0})
    for k in ["Q_out_MVAR", "Q_limited", "P_loss_MW", "P_standby_MW",
              "P_total_loss_MW", "operating_mode", "utilization"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC186"
    assert info["fidelity"] == "F1a"


def test_symmetric_range(model):
    """STATCOM has symmetric ±Q_max range (unlike asymmetric SVC)."""
    assert model._model.Q_min == -model._model.Q_max, \
        "STATCOM Q range must be symmetric"


def test_Q_out_within_range(model):
    Q_dem = np.linspace(-200, 200, 300)
    r = model.predict({"Q_demand_MVAR": Q_dem})
    assert np.all(r["Q_out_MVAR"] >= model._model.Q_min - 1e-9)
    assert np.all(r["Q_out_MVAR"] <= model._model.Q_max + 1e-9)


def test_capacitive_output_positive(model):
    r = model.predict({"Q_demand_MVAR": 70.0})
    assert float(r["Q_out_MVAR"]) > 0.0


def test_inductive_output_negative(model):
    r = model.predict({"Q_demand_MVAR": -70.0})
    assert float(r["Q_out_MVAR"]) < 0.0


def test_standby_loss_always_present(model):
    """Standby losses exist even at Q=0 (VSC cooling/auxiliary)."""
    r = model.predict({"Q_demand_MVAR": 0.0})
    assert float(r["P_standby_MW"]) == model._model.P_standby
    assert float(r["P_total_loss_MW"]) == model._model.P_standby


def test_total_loss_exceeds_standby(model):
    """For Q>0, total loss > standby loss."""
    r = model.predict({"Q_demand_MVAR": 50.0})
    assert float(r["P_total_loss_MW"]) > float(r["P_standby_MW"])


def test_cap_above_Qmax(model):
    Q_max = model._model.Q_max
    r = model.predict({"Q_demand_MVAR": Q_max + 40.0})
    assert abs(float(r["Q_out_MVAR"]) - Q_max) < 1e-9
    assert bool(r["Q_limited"]) is True


def test_cap_below_Qmin(model):
    Q_min = model._model.Q_min
    r = model.predict({"Q_demand_MVAR": Q_min - 40.0})
    assert abs(float(r["Q_out_MVAR"]) - Q_min) < 1e-9
    assert bool(r["Q_limited"]) is True


def test_no_limiting_in_range(model):
    r = model.predict({"Q_demand_MVAR": -50.0})
    assert bool(r["Q_limited"]) is False


def test_utilization_between_0_and_1(model):
    Q_range = np.linspace(-100, 100, 50)
    r = model.predict({"Q_demand_MVAR": Q_range})
    assert np.all(r["utilization"] >= 0.0)
    assert np.all(r["utilization"] <= 1.0 + 1e-9)


def test_variable_loss_proportional_to_abs_Q(model):
    r1 = model.predict({"Q_demand_MVAR": 30.0})
    r2 = model.predict({"Q_demand_MVAR": 60.0})
    ratio = float(r2["P_loss_MW"]) / float(r1["P_loss_MW"])
    assert abs(ratio - 2.0) < 1e-9


def test_benchmark(model):
    Q = np.random.uniform(-100, 100, 1000)
    start = time.perf_counter()
    model.predict({"Q_demand_MVAR": Q})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
