"""EC133 — Tidal Lagoon — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"tidal_range_m": 9.0})
    for k in ["avg_power_kw", "avg_power_mw", "theoretical_power_kw", "energy_per_cycle_mwh"]:
        assert k in r


def test_get_info(model):
    assert model.get_info()["ec_id"] == "EC133"


def test_zero_range_gives_zero_power(model):
    """Zero tidal range → zero potential energy → zero power."""
    r = model.predict({"tidal_range_m": 0.0})
    assert float(r["avg_power_mw"]) == 0.0, "Zero tidal range must yield zero power"


def test_below_minimum_head_gives_zero(model):
    """Below h_min threshold, no generation."""
    m = model._model
    R_below = m.h_min * 2.0 * 0.5   # amplitude well below h_min
    r = model.predict({"tidal_range_m": R_below})
    assert float(r["avg_power_mw"]) == 0.0, "Power must be zero below minimum head threshold"


def test_positive_power_at_design(model):
    m = model._model
    r = model.predict({"tidal_range_m": m.h_design * 2.0})
    assert float(r["avg_power_mw"]) > 0.0


def test_power_scales_as_range_squared(model):
    """P ∝ h² ∝ (R/2)² — must hold for bidirectional too."""
    ranges = np.array([3.0, 6.0, 9.0])
    r = model.predict({"tidal_range_m": ranges})
    P = r["avg_power_mw"]
    ratio_12 = float(P[1]) / float(P[0])
    expected_12 = (ranges[1] / ranges[0]) ** 2
    assert abs(ratio_12 - expected_12) < 0.01, \
        f"P must scale as R^2: expected {expected_12:.4f}, got {ratio_12:.4f}"


def test_bidirectional_doubles_ebb_only():
    """
    Bidirectional (n_cycles=2) should produce 2x the theoretical power of
    an equivalent ebb-only (n_cycles=1) system with same geometry.
    """
    import json
    from pathlib import Path
    base = Path(__file__).parent.parent
    with open(base / "data" / "parameters.json") as f:
        params = json.load(f)
    params_ebb = json.loads(json.dumps(params))
    params_ebb["unit"]["n_cycles_per_period"]["value"] = 1.0
    from model import TidalLagoonF1a
    m_bi = TidalLagoonF1a(params)
    m_ebb = TidalLagoonF1a(params_ebb)
    P_bi = float(m_bi.theoretical_avg_power_w(9.0))
    P_ebb = float(m_ebb.theoretical_avg_power_w(9.0))
    ratio = P_bi / P_ebb
    assert abs(ratio - 2.0) < 0.01, \
        f"Bidirectional must give 2x ebb-only theoretical power: got ratio {ratio:.4f}"


def test_power_scales_with_area(model):
    """Power scales linearly with lagoon area."""
    r1 = model.predict({"tidal_range_m": 9.0, "lagoon_area_m2": 10e6})
    r2 = model.predict({"tidal_range_m": 9.0, "lagoon_area_m2": 20e6})
    ratio = float(r2["avg_power_mw"]) / float(r1["avg_power_mw"])
    assert abs(ratio - 2.0) < 0.01, f"Power must be linear in area: got {ratio:.4f}"


def test_eta_applied_correctly(model):
    """P_avg = eta * P_theoretical."""
    m = model._model
    r = model.predict({"tidal_range_m": 9.0})
    expected = float(r["theoretical_power_kw"]) * m.eta
    actual = float(r["avg_power_kw"])
    assert abs(actual - expected) / expected < 1e-6


def test_seawater_density(model):
    """Model must use seawater density (1025), not freshwater (1000)."""
    assert model._model.rho == 1025.0, f"Expected rho=1025, got {model._model.rho}"


def test_power_monotonically_increases_with_range(model):
    """Power must increase monotonically with tidal range above h_min."""
    m = model._model
    ranges = np.linspace(m.h_min * 2 + 0.1, 12.0, 50)
    r = model.predict({"tidal_range_m": ranges})
    assert np.all(np.diff(r["avg_power_mw"]) >= 0), "Power must increase with tidal range"


def test_lagoon_vs_barrage_lower_eta():
    """
    EC133 lagoon eta (0.25) should be lower than typical EC131 barrage eta (0.28).
    This reflects bidirectional turbine efficiency penalty.
    """
    import json
    from pathlib import Path
    base = Path(__file__).parent.parent
    with open(base / "data" / "parameters.json") as f:
        params = json.load(f)
    lagoon_eta = params["unit"]["eta_plant"]["value"]
    assert lagoon_eta <= 0.28, \
        f"Lagoon eta should be <= 0.28 (barrage reference); got {lagoon_eta}"


def test_energy_cycle_consistency(model):
    """E_cycle = P_avg * T_tide / 3600."""
    m = model._model
    r = model.predict({"tidal_range_m": 9.0})
    expected_E = float(r["avg_power_mw"]) * m.T_tide / 3600.0
    assert abs(float(r["energy_per_cycle_mwh"]) - expected_E) < 1e-6


def test_power_nonnegative(model):
    ranges = np.linspace(0.0, 12.0, 100)
    r = model.predict({"tidal_range_m": ranges})
    assert np.all(r["avg_power_mw"] >= 0.0)


def test_benchmark(model):
    ranges = np.random.uniform(0.0, 12.0, 1000)
    start = time.perf_counter()
    model.predict({"tidal_range_m": ranges})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
