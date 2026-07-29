"""EC131 — Tidal Barrage — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"tidal_range_m": 8.0})
    for k in ["avg_power_kw", "avg_power_mw", "theoretical_power_kw", "energy_per_cycle_mwh"]:
        assert k in r


def test_get_info(model):
    assert model.get_info()["ec_id"] == "EC131"


def test_zero_range_gives_zero_power(model):
    """Zero tidal range → zero power (physics: no potential energy)."""
    r = model.predict({"tidal_range_m": 0.0})
    assert float(r["avg_power_mw"]) == 0.0, "Zero tidal range must yield zero power"


def test_below_minimum_head_gives_zero(model):
    """Below h_min (1 m amplitude = 2 m range), no generation."""
    m = model._model
    R_below = m.h_min * 2.0 * 0.5   # amplitude < h_min
    r = model.predict({"tidal_range_m": R_below})
    assert float(r["avg_power_mw"]) == 0.0, "Power must be zero below minimum head threshold"


def test_power_positive_at_design(model):
    """Design tidal range must produce positive power."""
    m = model._model
    r = model.predict({"tidal_range_m": m.h_design * 2.0})
    assert float(r["avg_power_mw"]) > 0.0


def test_power_scales_as_range_squared(model):
    """P ∝ h² ∝ (R/2)² — power must scale quadratically with tidal range."""
    ranges = np.array([4.0, 8.0, 12.0])
    r = model.predict({"tidal_range_m": ranges})
    P = r["avg_power_mw"]
    # Ratio of powers should equal ratio of ranges squared
    ratio_12 = float(P[1]) / float(P[0])
    expected_12 = (ranges[1] / ranges[0]) ** 2
    assert abs(ratio_12 - expected_12) < 0.01, \
        f"P should scale as R^2: expected ratio {expected_12:.4f}, got {ratio_12:.4f}"


def test_power_scales_with_basin_area(model):
    """Power scales linearly with basin area."""
    r1 = model.predict({"tidal_range_m": 8.0, "basin_area_m2": 10e6})
    r2 = model.predict({"tidal_range_m": 8.0, "basin_area_m2": 20e6})
    ratio = float(r2["avg_power_mw"]) / float(r1["avg_power_mw"])
    assert abs(ratio - 2.0) < 0.01, f"Power must be linear in basin area: got {ratio:.4f}"


def test_eta_applied_correctly(model):
    """P_avg must equal eta * P_theoretical."""
    m = model._model
    r = model.predict({"tidal_range_m": 10.0})
    expected = float(r["theoretical_power_kw"]) * m.eta
    actual = float(r["avg_power_kw"])
    assert abs(actual - expected) / expected < 1e-6, \
        f"eta not applied correctly: expected {expected:.1f} kW, got {actual:.1f} kW"


def test_seawater_density_applied(model):
    """Model must use seawater density (rho=1025), not freshwater (1000)."""
    m = model._model
    assert m.rho == 1025.0, f"Expected rho=1025 kg/m3 (seawater), got {m.rho}"


def test_power_increases_monotonically_with_range(model):
    """Power must increase monotonically with tidal range above h_min."""
    ranges = np.linspace(3.0, 16.0, 50)  # all above 2*h_min
    r = model.predict({"tidal_range_m": ranges})
    assert np.all(np.diff(r["avg_power_mw"]) >= 0), "Power must increase with tidal range"


def test_energy_per_cycle_consistency(model):
    """Energy per cycle = P_avg * T_tide / 3600."""
    m = model._model
    r = model.predict({"tidal_range_m": 8.0})
    expected_E = float(r["avg_power_mw"]) * m.T_tide / 3600.0
    assert abs(float(r["energy_per_cycle_mwh"]) - expected_E) < 1e-6


def test_power_nonnegative(model):
    ranges = np.linspace(0.0, 16.0, 100)
    r = model.predict({"tidal_range_m": ranges})
    assert np.all(r["avg_power_mw"] >= 0.0), "Power must never be negative"


def test_benchmark(model):
    ranges = np.random.uniform(0.0, 16.0, 1000)
    start = time.perf_counter()
    model.predict({"tidal_range_m": ranges})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
