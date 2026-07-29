"""EC189 — Natural Gas Pipeline — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"length_km": 100.0, "diameter_m": 0.6096,
                       "P_in_bar": 75.0, "P_out_bar": 50.0})
    for k in ["Q_std_m3_per_day", "Q_std_m3_per_s", "Q_kg_per_s",
               "pressure_drop_bar", "weymouth_f"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC189"
    assert info["fidelity"] == "F1a"


def test_flow_positive(model):
    r = model.predict({"length_km": 100.0, "diameter_m": 0.6096,
                       "P_in_bar": 75.0, "P_out_bar": 50.0})
    assert float(r["Q_std_m3_per_day"]) > 0
    assert float(r["Q_kg_per_s"]) > 0


def test_zero_flow_when_pressures_equal(model):
    """No pressure differential → no flow."""
    r = model.predict({"length_km": 100.0, "diameter_m": 0.6096,
                       "P_in_bar": 70.0, "P_out_bar": 70.0})
    assert float(r["Q_std_m3_per_day"]) == pytest.approx(0.0, abs=1e-6)


def test_q_scales_with_sqrt_dp2(model):
    """Core Weymouth physics: Q ∝ √(P_in²-P_out²). Doubling dP² → flow x√2."""
    # dP² case 1: P1=75, P2=50 → dP² = 75²-50²=3125
    # dP² case 2: P1=75, P2=0  (scale factor)
    # Instead: vary P_out, verify linear relationship with sqrt(P1²-P2²)
    P_out = np.array([20.0, 30.0, 40.0, 50.0, 60.0])
    P_in = 75.0
    sqrt_dp2 = np.sqrt(P_in ** 2 - P_out ** 2)
    r = model.predict({"length_km": 100.0, "diameter_m": 0.6096,
                       "P_in_bar": P_in, "P_out_bar": P_out})
    Q = r["Q_std_m3_per_day"]
    # ratio Q[i]/Q[0] must equal sqrt_dp2[i]/sqrt_dp2[0]
    ratios_model = Q / Q[0]
    ratios_theory = sqrt_dp2 / sqrt_dp2[0]
    np.testing.assert_allclose(ratios_model, ratios_theory, rtol=1e-5)


def test_flow_increases_with_pressure_differential(model):
    """Higher dP → higher flow."""
    P_out = np.array([70.0, 60.0, 50.0, 40.0, 30.0])  # decreasing P_out = increasing dP
    r = model.predict({"length_km": 100.0, "diameter_m": 0.6096,
                       "P_in_bar": 75.0, "P_out_bar": P_out})
    assert np.all(np.diff(r["Q_std_m3_per_day"]) > 0)


def test_flow_increases_with_diameter(model):
    """Larger pipe → more flow (D^8/3 dependence)."""
    D = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    r = model.predict({"length_km": 100.0, "diameter_m": D,
                       "P_in_bar": 75.0, "P_out_bar": 50.0})
    assert np.all(np.diff(r["Q_std_m3_per_day"]) > 0)


def test_flow_decreases_with_length(model):
    """Longer pipeline → less flow for same pressures."""
    L = np.array([50.0, 100.0, 200.0, 400.0])
    r = model.predict({"length_km": L, "diameter_m": 0.6096,
                       "P_in_bar": 75.0, "P_out_bar": 50.0})
    assert np.all(np.diff(r["Q_std_m3_per_day"]) < 0)


def test_diameter_exponent(model):
    """Q ∝ D^(8/3): doubling D should multiply Q by 2^(8/3)."""
    r1 = model.predict({"length_km": 100.0, "diameter_m": 0.3,
                         "P_in_bar": 75.0, "P_out_bar": 50.0})
    r2 = model.predict({"length_km": 100.0, "diameter_m": 0.6,
                         "P_in_bar": 75.0, "P_out_bar": 50.0})
    ratio = float(r2["Q_std_m3_per_day"]) / float(r1["Q_std_m3_per_day"])
    expected = 2.0 ** (8.0 / 3.0)
    assert ratio == pytest.approx(expected, rel=1e-5)


def test_pressure_drop_consistent(model):
    """pressure_drop_bar == P_in - P_out."""
    r = model.predict({"length_km": 100.0, "diameter_m": 0.6096,
                       "P_in_bar": 75.0, "P_out_bar": 50.0})
    assert float(r["pressure_drop_bar"]) == pytest.approx(25.0, rel=1e-9)


def test_realistic_flow_range(model):
    """24-inch, 100 km, 75→50 bar: expect ~3-30 Mm³/day (industry typical)."""
    r = model.predict({"length_km": 100.0, "diameter_m": 0.6096,
                       "P_in_bar": 75.0, "P_out_bar": 50.0})
    Q_Mm3 = float(r["Q_std_m3_per_day"]) / 1e6
    assert 1.0 < Q_Mm3 < 50.0, f"Q={Q_Mm3:.2f} Mm³/day outside realistic range"


def test_weymouth_friction_decreases_with_diameter(model):
    """Weymouth f = 0.032/D^(1/3) — larger pipe → lower friction factor."""
    D = np.array([0.2, 0.4, 0.6, 0.9])
    r = model.predict({"length_km": 100.0, "diameter_m": D,
                       "P_in_bar": 75.0, "P_out_bar": 50.0})
    assert np.all(np.diff(r["weymouth_f"]) < 0)


def test_benchmark(model):
    rng = np.random.default_rng(42)
    L = rng.uniform(10, 500, 1000)
    D = rng.uniform(0.2, 1.0, 1000)
    P1 = rng.uniform(30, 150, 1000)
    P2 = P1 * rng.uniform(0.5, 0.9, 1000)
    start = time.perf_counter()
    model.predict({"length_km": L, "diameter_m": D, "P_in_bar": P1, "P_out_bar": P2})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
