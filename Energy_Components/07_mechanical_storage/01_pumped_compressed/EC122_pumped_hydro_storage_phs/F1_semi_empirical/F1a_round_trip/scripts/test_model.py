"""EC122 — Pumped Hydro Storage — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys_generate(model):
    r = model.predict({"mode": "generate", "flow_rate": 50.0, "head": 300.0})
    for k in ["power_kw", "efficiency", "energy_capacity_gwh", "round_trip_eta"]:
        assert k in r


def test_predict_keys_pump(model):
    r = model.predict({"mode": "pump", "flow_rate": 50.0, "head": 300.0})
    for k in ["power_kw", "efficiency", "energy_capacity_gwh", "round_trip_eta"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC122"
    assert info["fidelity"] == "F1a"


def test_p_pump_greater_than_p_gen(model):
    """For same Q and H, pumping power > generation power (losses)."""
    Q, H = 50.0, 300.0
    p_gen = float(model.predict({"mode": "generate", "flow_rate": Q, "head": H})["power_kw"])
    p_pump = float(model.predict({"mode": "pump", "flow_rate": Q, "head": H})["power_kw"])
    assert p_pump > p_gen, f"Pump power ({p_pump:.1f}) should exceed gen power ({p_gen:.1f})"


def test_round_trip_eta_range(model):
    """Round-trip efficiency should be ~0.75-0.80 for typical PHS."""
    rte = model.predict({"mode": "generate", "flow_rate": 50.0, "head": 300.0})["round_trip_eta"]
    assert 0.70 <= rte <= 0.85, f"Round-trip eta = {rte:.3f} outside expected 0.70-0.85"


def test_power_proportional_to_flow(model):
    """Power scales linearly with flow rate."""
    Q1, Q2, H = 25.0, 50.0, 300.0
    p1 = float(model.predict({"mode": "generate", "flow_rate": Q1, "head": H})["power_kw"])
    p2 = float(model.predict({"mode": "generate", "flow_rate": Q2, "head": H})["power_kw"])
    ratio = p2 / p1
    assert abs(ratio - 2.0) < 0.01, f"Power ratio {ratio:.3f} should be 2.0 (linear in Q)"


def test_power_proportional_to_head(model):
    """Power scales linearly with head."""
    Q, H1, H2 = 50.0, 150.0, 300.0
    p1 = float(model.predict({"mode": "generate", "flow_rate": Q, "head": H1})["power_kw"])
    p2 = float(model.predict({"mode": "generate", "flow_rate": Q, "head": H2})["power_kw"])
    ratio = p2 / p1
    assert abs(ratio - 2.0) < 0.01, f"Power ratio {ratio:.3f} should be 2.0 (linear in H)"


def test_energy_capacity_positive(model):
    """Energy capacity must be positive."""
    r = model.predict({"mode": "generate", "flow_rate": 50.0, "head": 300.0})
    assert float(r["energy_capacity_gwh"]) > 0.0


def test_energy_capacity_scales_with_head(model):
    """Energy capacity scales linearly with head."""
    r1 = model.predict({"mode": "generate", "flow_rate": 50.0, "head": 150.0})
    r2 = model.predict({"mode": "generate", "flow_rate": 50.0, "head": 300.0})
    e1 = float(r1["energy_capacity_gwh"])
    e2 = float(r2["energy_capacity_gwh"])
    assert abs(e2 / e1 - 2.0) < 0.01


def test_invalid_mode(model):
    """Invalid mode should raise ValueError."""
    with pytest.raises(ValueError):
        model.predict({"mode": "charge", "flow_rate": 50.0, "head": 300.0})


def test_vectorized_input(model):
    """Model must accept array inputs."""
    Q = np.linspace(10, 100, 20)
    H = np.linspace(100, 400, 20)
    r = model.predict({"mode": "generate", "flow_rate": Q, "head": H})
    assert len(r["power_kw"]) == 20


def test_benchmark(model):
    Q = np.random.uniform(10, 200, 1000)
    H = np.random.uniform(50, 800, 1000)
    start = time.perf_counter()
    model.predict({"mode": "generate", "flow_rate": Q, "head": H})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
