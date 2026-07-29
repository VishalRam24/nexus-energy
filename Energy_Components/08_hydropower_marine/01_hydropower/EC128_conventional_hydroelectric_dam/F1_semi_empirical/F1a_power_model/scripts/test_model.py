"""EC128 — Hydroelectric Dam — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"flow_rate_m3s": 30.0, "head_m": 100.0})
    for k in ["power_kw", "turbine_efficiency", "overall_efficiency", "capacity_factor"]:
        assert k in r


def test_get_info(model):
    assert model.get_info()["ec_id"] == "EC128"


def test_power_proportional_to_flow(model):
    """Power should increase with flow (at design head)."""
    flows = np.linspace(10.0, 30.0, 20)
    r = model.predict({"flow_rate_m3s": flows, "head_m": 100.0})
    # Power should generally increase — check overall trend
    assert r["power_kw"][-1] > r["power_kw"][0], "Power must increase with flow in valid range"


def test_power_proportional_to_head(model):
    """Power should increase with head when below rated limit."""
    # Use partial flow (Q=20 m3/s) so P stays well below P_rated across all heads
    heads = np.array([60.0, 80.0, 100.0, 120.0, 140.0])
    r1 = model.predict({"flow_rate_m3s": 20.0, "head_m": heads})
    assert np.all(np.diff(r1["power_kw"]) > 0), "Power must increase with head (below rated limit)"


def test_peak_efficiency_at_design_flow(model):
    """Turbine efficiency should peak at Q_design."""
    m = model._model
    flows = np.linspace(m.Q_design * 0.4, m.Q_design * 1.05, 200)
    r = model.predict({"flow_rate_m3s": flows, "head_m": 100.0})
    peak_idx = np.argmax(r["turbine_efficiency"])
    peak_flow = flows[peak_idx]
    assert abs(peak_flow - m.Q_design) / m.Q_design < 0.05, \
        f"Peak efficiency should be near Q_design={m.Q_design}, got {peak_flow:.2f}"


def test_power_rated_at_design(model):
    """Power at design point should be close to rated capacity."""
    r = model.predict({"flow_rate_m3s": 30.0, "head_m": 100.0})
    P = float(r["power_kw"])
    P_rated = model._model.P_rated
    assert abs(P - P_rated) / P_rated < 0.05, \
        f"P at design = {P:.1f} kW, expected ~{P_rated:.1f} kW"


def test_power_below_rated(model):
    """Power must never exceed P_rated."""
    flows = np.linspace(0.0, 33.0, 100)
    heads = np.linspace(50.0, 150.0, 100)
    r = model.predict({"flow_rate_m3s": flows, "head_m": heads})
    assert np.all(r["power_kw"] <= model._model.P_rated + 1.0), "Power must not exceed P_rated"


def test_zero_power_below_cutoff(model):
    """Below minimum flow (q < 0.3), power should be zero."""
    Q_min = 0.3 * model._model.Q_design
    r = model.predict({"flow_rate_m3s": Q_min * 0.5, "head_m": 100.0})
    assert float(r["power_kw"]) == 0.0, "Power must be zero below cut-in flow"


def test_efficiency_physical_range(model):
    """Efficiency values must be between 0 and 1."""
    flows = np.linspace(0.0, 33.0, 50)
    r = model.predict({"flow_rate_m3s": flows, "head_m": 100.0})
    assert np.all(r["turbine_efficiency"] >= 0.0)
    assert np.all(r["turbine_efficiency"] <= 1.0)
    assert np.all(r["overall_efficiency"] >= 0.0)
    assert np.all(r["overall_efficiency"] <= 1.0)


def test_capacity_factor_range(model):
    """Capacity factor must be in [0, 1]."""
    r = model.predict({"flow_rate_m3s": 30.0, "head_m": 100.0})
    cf = float(r["capacity_factor"])
    assert 0.0 <= cf <= 1.0, f"Capacity factor = {cf}"


def test_benchmark(model):
    flows = np.random.uniform(5.0, 33.0, 1000)
    heads = np.random.uniform(50.0, 150.0, 1000)
    start = time.perf_counter()
    model.predict({"flow_rate_m3s": flows, "head_m": heads})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
