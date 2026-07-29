"""EC132 — Tidal Stream Turbine — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"current_speed_ms": 2.5})
    for k in ["power_kw", "capacity_factor", "power_coefficient"]:
        assert k in r


def test_get_info(model):
    assert model.get_info()["ec_id"] == "EC132"


def test_zero_speed_gives_zero_power(model):
    """v=0 → zero kinetic energy → zero power."""
    r = model.predict({"current_speed_ms": 0.0})
    assert float(r["power_kw"]) == 0.0, "Zero current speed must yield zero power"


def test_below_cut_in_gives_zero_power(model):
    """Below cut-in velocity, no power extracted."""
    m = model._model
    v_below = m.v_cut_in * 0.5
    r = model.predict({"current_speed_ms": v_below})
    assert float(r["power_kw"]) == 0.0, f"Power must be zero below cut-in ({m.v_cut_in} m/s)"


def test_above_cut_out_gives_zero_power(model):
    """Above cut-out velocity, turbine feathered → zero power."""
    m = model._model
    v_above = m.v_cut_out + 0.5
    r = model.predict({"current_speed_ms": v_above})
    assert float(r["power_kw"]) == 0.0, f"Power must be zero above cut-out ({m.v_cut_out} m/s)"


def test_rated_power_at_rated_speed(model):
    """At rated speed, power should equal P_rated."""
    m = model._model
    r = model.predict({"current_speed_ms": m.v_rated})
    P = float(r["power_kw"])
    assert abs(P - m.P_rated) / m.P_rated < 0.05, \
        f"P at v_rated = {P:.1f} kW, expected ~{m.P_rated:.1f} kW"


def test_power_not_exceed_rated(model):
    """Power must never exceed rated capacity."""
    m = model._model
    speeds = np.linspace(0.0, 4.5, 200)
    r = model.predict({"current_speed_ms": speeds})
    assert np.all(r["power_kw"] <= m.P_rated + 1.0), "Power must not exceed P_rated"


def test_power_rises_below_rated(model):
    """Power must increase with speed between cut-in and rated."""
    m = model._model
    speeds = np.linspace(m.v_cut_in + 0.1, m.v_rated - 0.1, 20)
    r = model.predict({"current_speed_ms": speeds})
    assert np.all(np.diff(r["power_kw"]) > 0), "Power must rise monotonically below rated speed"


def test_power_scales_as_v_cubed(model):
    """In sub-rated region, P ∝ v^3."""
    m = model._model
    v1, v2 = 1.2, 2.0   # both in sub-rated range
    r1 = model.predict({"current_speed_ms": v1})
    r2 = model.predict({"current_speed_ms": v2})
    ratio_P = float(r2["power_kw"]) / float(r1["power_kw"])
    ratio_v3 = (v2 / v1) ** 3
    assert abs(ratio_P - ratio_v3) / ratio_v3 < 0.02, \
        f"P must scale as v^3: expected {ratio_v3:.4f}, got {ratio_P:.4f}"


def test_seawater_density_used(model):
    """Model must use rho=1025 (seawater), not 1.225 (air)."""
    m = model._model
    assert m.rho_ref == 1025.0, f"Expected rho=1025 kg/m3, got {m.rho_ref}"


def test_density_scaling(model):
    """Power scales linearly with water density (below rated)."""
    v_sub = 1.5  # sub-rated
    r1 = model.predict({"current_speed_ms": v_sub, "water_density": 1000.0})
    r2 = model.predict({"current_speed_ms": v_sub, "water_density": 1025.0})
    ratio = float(r2["power_kw"]) / float(r1["power_kw"])
    expected = 1025.0 / 1000.0
    assert abs(ratio - expected) < 0.01, \
        f"Power must scale with rho: expected {expected:.4f}, got {ratio:.4f}"


def test_cp_below_betz_limit(model):
    """Cp must always be <= Betz limit (0.593)."""
    speeds = np.linspace(0.0, 4.5, 100)
    r = model.predict({"current_speed_ms": speeds})
    assert np.all(r["power_coefficient"] <= 0.593 + 1e-6), "Cp must not exceed Betz limit"


def test_cp_nonnegative(model):
    speeds = np.linspace(0.0, 4.5, 100)
    r = model.predict({"current_speed_ms": speeds})
    assert np.all(r["power_coefficient"] >= 0.0)


def test_capacity_factor_range(model):
    speeds = np.linspace(0.0, 4.5, 100)
    r = model.predict({"current_speed_ms": speeds})
    assert np.all(r["capacity_factor"] >= 0.0)
    assert np.all(r["capacity_factor"] <= 1.0)


def test_benchmark(model):
    speeds = np.random.uniform(0.0, 4.5, 1000)
    start = time.perf_counter()
    model.predict({"current_speed_ms": speeds})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
