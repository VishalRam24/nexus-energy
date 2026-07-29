"""EC130 — Small/Micro Hydropower — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import TURBINE_TYPES, select_turbine


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"flow_rate_m3s": 1.5, "net_head_m": 40.0})
    for k in ["power_kw", "turbine_efficiency", "overall_efficiency", "capacity_factor", "turbine_type"]:
        assert k in r


def test_get_info(model):
    assert model.get_info()["ec_id"] == "EC130"


def test_zero_flow_gives_zero_power(model):
    r = model.predict({"flow_rate_m3s": 0.0, "net_head_m": 40.0})
    assert float(r["power_kw"]) == 0.0, "Zero flow must yield zero power"


def test_zero_head_gives_zero_power(model):
    r = model.predict({"flow_rate_m3s": 1.5, "net_head_m": 0.0})
    assert float(r["power_kw"]) == 0.0, "Zero net head must yield zero power"


def test_positive_power_at_design(model):
    r = model.predict({"flow_rate_m3s": 1.5, "net_head_m": 40.0, "turbine_type": "francis"})
    assert float(r["power_kw"]) > 0.0


def test_below_cutoff_zero_power(model):
    m = model._model
    Q_below = m.Q_design * m.q_min * 0.5
    r = model.predict({"flow_rate_m3s": Q_below, "net_head_m": 40.0})
    assert float(r["power_kw"]) == 0.0, "Power must be zero below q_min"


def test_turbine_selection_pelton(model):
    """H > 100 m should select Pelton."""
    r = model.predict({"flow_rate_m3s": 1.5, "net_head_m": 200.0, "turbine_type": "auto"})
    assert r["turbine_type"] == "pelton"


def test_turbine_selection_kaplan(model):
    """H < 20 m should select Kaplan."""
    r = model.predict({"flow_rate_m3s": 1.5, "net_head_m": 10.0, "turbine_type": "auto"})
    assert r["turbine_type"] == "kaplan"


def test_turbine_selection_francis(model):
    """20 m <= H <= 100 m should select Francis."""
    r = model.predict({"flow_rate_m3s": 1.5, "net_head_m": 50.0, "turbine_type": "auto"})
    assert r["turbine_type"] == "francis"


def test_pelton_eta_higher_than_kaplan(model):
    """At design flow: Pelton eta_peak (0.91) > Kaplan (0.87)."""
    m = model._model
    eta_p = m.turbine_efficiency(m.Q_design, "pelton")
    eta_k = m.turbine_efficiency(m.Q_design, "kaplan")
    assert float(eta_p) > float(eta_k), "Pelton peak efficiency must exceed Kaplan"


def test_power_increases_with_head(model):
    """Power must increase with net head at partial flow (below P_rated across all heads)."""
    m = model._model
    # Use low flow (30% of design) so P_rated cap is not hit even at H=120 m
    Q_partial = m.Q_design * 0.30
    heads = np.array([10.0, 30.0, 60.0, 120.0])
    powers = [float(model.predict({"flow_rate_m3s": Q_partial, "net_head_m": H, "turbine_type": "auto"})["power_kw"])
              for H in heads]
    assert all(powers[i+1] > powers[i] for i in range(len(powers)-1)), \
        f"Power must increase monotonically with head (at Q=30% design): {powers}"


def test_power_not_exceed_rated(model):
    m = model._model
    flows = np.linspace(0.0, 1.65, 50)
    heads = np.linspace(2.0, 300.0, 50)
    for H in heads[:10]:  # test across heads
        r = model.predict({"flow_rate_m3s": flows, "net_head_m": H})
        assert np.all(r["power_kw"] <= m.P_rated + 1.0), f"Power exceeded P_rated at H={H}"


def test_efficiency_physical_range(model):
    flows = np.linspace(0.0, 1.65, 50)
    for t in ["pelton", "francis", "kaplan"]:
        r = model.predict({"flow_rate_m3s": flows, "net_head_m": 40.0, "turbine_type": t})
        assert np.all(r["turbine_efficiency"] >= 0.0)
        assert np.all(r["turbine_efficiency"] <= 1.0)


def test_density_scaling(model):
    """Power scales proportionally with water density."""
    import json
    from pathlib import Path
    base = Path(__file__).parent.parent
    with open(base / "data" / "parameters.json") as f:
        params = json.load(f)
    params2 = json.loads(json.dumps(params))
    params2["unit"]["rho"]["value"] = 1025.0
    from model import SmallMicroHydroF1a
    m1 = SmallMicroHydroF1a(params)
    m2 = SmallMicroHydroF1a(params2)
    P1 = float(m1.power_kw(1.0, 40.0, "francis"))
    P2 = float(m2.power_kw(1.0, 40.0, "francis"))
    assert abs(P2 / P1 - 1025.0 / 1000.0) < 0.01


def test_benchmark(model):
    flows = np.random.uniform(0.1, 1.65, 1000)
    heads = np.random.uniform(2.0, 300.0, 1000)
    start = time.perf_counter()
    model.predict({"flow_rate_m3s": flows, "net_head_m": heads})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
