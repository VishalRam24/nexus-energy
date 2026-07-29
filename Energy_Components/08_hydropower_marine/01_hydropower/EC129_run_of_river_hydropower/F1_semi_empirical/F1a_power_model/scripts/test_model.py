"""EC129 — Run-of-River Hydropower — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"flow_rate_m3s": 50.0, "gross_head_m": 8.0})
    for k in ["power_kw", "net_head_m", "turbine_efficiency", "overall_efficiency", "capacity_factor"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC129"
    assert "F1a" in info["fidelity"]


def test_zero_flow_gives_zero_power(model):
    """Physics: zero flow → zero power."""
    r = model.predict({"flow_rate_m3s": 0.0, "gross_head_m": 8.0})
    assert float(r["power_kw"]) == 0.0, "Zero flow must yield zero power"


def test_zero_head_gives_zero_power(model):
    """Physics: zero gross head → zero net head → zero power."""
    r = model.predict({"flow_rate_m3s": 50.0, "gross_head_m": 0.0})
    assert float(r["power_kw"]) == 0.0, "Zero head must yield zero power"


def test_power_positive_at_design_point(model):
    """Design-point operation must produce positive power."""
    r = model.predict({"flow_rate_m3s": 50.0, "gross_head_m": 8.0})
    assert float(r["power_kw"]) > 0.0, "Design point must produce power"


def test_below_cutoff_gives_zero_power(model):
    """Below q_min flow, power must be zero."""
    m = model._model
    Q_below = m.Q_design * m.q_min * 0.5   # well below cut-in
    r = model.predict({"flow_rate_m3s": Q_below, "gross_head_m": 8.0})
    assert float(r["power_kw"]) == 0.0, f"Power must be zero at Q={Q_below:.2f} m3/s (below q_min)"


def test_power_increases_with_flow(model):
    """In valid flow range, power should increase with flow."""
    m = model._model
    # Use flows strictly in valid range [q_min, 1.0] * Q_design
    flows = np.linspace(m.Q_design * (m.q_min + 0.05), m.Q_design * 0.95, 20)
    r = model.predict({"flow_rate_m3s": flows, "gross_head_m": 8.0})
    assert r["power_kw"][-1] > r["power_kw"][0], "Power must increase with flow in valid range"


def test_power_increases_with_head(model):
    """Power must increase with gross head (partial flow, below rated limit)."""
    m = model._model
    Q_partial = m.Q_design * 0.6   # use 60% flow to stay well below rated
    heads = np.array([3.0, 5.0, 8.0, 12.0, 18.0])
    r = model.predict({"flow_rate_m3s": Q_partial, "gross_head_m": heads})
    assert np.all(np.diff(r["power_kw"]) > 0), "Power must increase monotonically with head"


def test_power_not_exceed_rated(model):
    """Power must never exceed P_rated."""
    m = model._model
    flows = np.linspace(0.0, 57.5, 100)
    heads = np.linspace(2.0, 20.0, 100)
    r = model.predict({"flow_rate_m3s": flows, "gross_head_m": heads})
    assert np.all(r["power_kw"] <= m.P_rated + 1.0), "Power must not exceed P_rated"


def test_efficiency_physical_range(model):
    """Efficiency values must be in [0, 1]."""
    flows = np.linspace(0.0, 57.5, 50)
    r = model.predict({"flow_rate_m3s": flows, "gross_head_m": 8.0})
    assert np.all(r["turbine_efficiency"] >= 0.0)
    assert np.all(r["turbine_efficiency"] <= 1.0)
    assert np.all(r["overall_efficiency"] >= 0.0)
    assert np.all(r["overall_efficiency"] <= 1.0)


def test_peak_efficiency_at_design_flow(model):
    """Turbine efficiency must peak near Q_design."""
    m = model._model
    flows = np.linspace(m.Q_design * 0.4, m.Q_design * 1.05, 200)
    r = model.predict({"flow_rate_m3s": flows, "gross_head_m": 8.0})
    peak_idx = np.argmax(r["turbine_efficiency"])
    peak_flow = flows[peak_idx]
    assert abs(peak_flow - m.Q_design) / m.Q_design < 0.05, \
        f"Peak eta should be near Q_design={m.Q_design}, got {peak_flow:.2f}"


def test_net_head_less_than_gross(model):
    """Net head must be strictly less than gross head (penstock losses)."""
    r = model.predict({"flow_rate_m3s": 50.0, "gross_head_m": 8.0})
    assert float(r["net_head_m"]) < 8.0, "Net head must be less than gross head"


def test_capacity_factor_range(model):
    """Capacity factor must be in [0, 1]."""
    r = model.predict({"flow_rate_m3s": 50.0, "gross_head_m": 8.0})
    cf = float(r["capacity_factor"])
    assert 0.0 <= cf <= 1.0, f"Capacity factor = {cf}"


def test_water_density_scaling(model):
    """Power must scale proportionally with water density (rho)."""
    import json
    from pathlib import Path
    base = Path(__file__).parent.parent
    with open(base / "data" / "parameters.json") as f:
        params = json.load(f)
    params_heavy = json.loads(json.dumps(params))
    params_heavy["unit"]["rho"]["value"] = 1025.0   # seawater density
    from model import RunOfRiverF1a
    m1 = RunOfRiverF1a(params)
    m2 = RunOfRiverF1a(params_heavy)
    P1 = m1.power_kw(40.0, 8.0)
    P2 = m2.power_kw(40.0, 8.0)
    ratio = float(P2) / float(P1)
    assert abs(ratio - 1025.0 / 1000.0) < 0.01, \
        f"Power must scale with density: expected {1025/1000:.4f}, got {ratio:.4f}"


def test_benchmark(model):
    flows = np.random.uniform(5.0, 57.5, 1000)
    heads = np.random.uniform(2.0, 20.0, 1000)
    start = time.perf_counter()
    model.predict({"flow_rate_m3s": flows, "gross_head_m": heads})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
