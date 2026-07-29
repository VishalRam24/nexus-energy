"""EC065 — Offshore Fixed-Bottom Wind Turbine — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

BETZ_LIMIT = 16.0 / 27.0  # 0.5926


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"wind_speed": 10.0})
    for k in ["power_kw", "capacity_factor", "power_coefficient"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC065"
    assert "power_kw" in info["outputs"]


def test_zero_power_below_cut_in(model):
    """No power below cut-in speed (3.5 m/s)."""
    for v in [0.0, 1.0, 2.0, 3.0, 3.4]:
        r = model.predict({"wind_speed": v})
        assert float(r["power_kw"]) == 0.0, f"Power={float(r['power_kw']):.1f}kW at v={v}m/s"


def test_zero_power_above_cut_out(model):
    """No power above cut-out speed (25 m/s)."""
    for v in [25.1, 27.0, 30.0]:
        r = model.predict({"wind_speed": v})
        assert float(r["power_kw"]) == 0.0, f"Power={float(r['power_kw']):.1f}kW at v={v}m/s"


def test_rated_power_at_rated_speed(model):
    """At and above rated power wind speed (14+ m/s), output must be at rated (3600 kW)."""
    for v in [14.0, 15.0, 20.0, 24.9]:
        r = model.predict({"wind_speed": v})
        p = float(r["power_kw"])
        assert abs(p - 3600.0) < 5.0, f"P={p:.1f}kW at v={v}m/s, expected ~3600kW"


def test_power_does_not_exceed_rated(model):
    """Power must never exceed rated capacity."""
    v = np.linspace(0, 30, 300)
    r = model.predict({"wind_speed": v})
    assert np.all(r["power_kw"] <= 3600.0 + 1e-6), "Power exceeded rated capacity"


def test_power_non_negative(model):
    """Power must always be non-negative."""
    v = np.linspace(0, 30, 300)
    r = model.predict({"wind_speed": v})
    assert np.all(r["power_kw"] >= 0.0), "Negative power detected"


def test_cp_below_betz_limit(model):
    """Power coefficient must never exceed Betz limit (0.5926)."""
    v = np.linspace(0.5, 30, 200)
    r = model.predict({"wind_speed": v})
    cp = r["power_coefficient"]
    assert np.all(cp <= BETZ_LIMIT + 1e-6), f"Cp={np.max(cp):.4f} > Betz limit {BETZ_LIMIT:.4f}"


def test_capacity_factor_range(model):
    """Capacity factor must be in [0, 1]."""
    v = np.linspace(0, 30, 200)
    r = model.predict({"wind_speed": v})
    assert np.all(r["capacity_factor"] >= 0.0)
    assert np.all(r["capacity_factor"] <= 1.0 + 1e-6)


def test_density_correction_reduces_power(model):
    """Lower air density (warm/humid offshore air) reduces power output."""
    v = 10.0
    r_ref = model.predict({"wind_speed": v, "air_density": 1.225})
    r_low = model.predict({"wind_speed": v, "air_density": 1.05})
    assert float(r_low["power_kw"]) < float(r_ref["power_kw"]), \
        "Lower density should reduce power"


def test_density_correction_increases_power(model):
    """Higher air density (cold offshore air) increases power output."""
    v = 10.0
    r_ref = model.predict({"wind_speed": v, "air_density": 1.225})
    r_high = model.predict({"wind_speed": v, "air_density": 1.35})
    assert float(r_high["power_kw"]) > float(r_ref["power_kw"]), \
        "Higher density should increase power"


def test_power_increases_in_operational_range(model):
    """Power must be monotonically increasing from cut-in to rated speed."""
    v = np.linspace(3.5, 12.5, 50)
    r = model.predict({"wind_speed": v})
    diffs = np.diff(r["power_kw"])
    assert np.all(diffs >= -1e-3), "Power not monotonically increasing in operational range"


def test_benchmark(model):
    v = np.random.uniform(0, 30, 1000)
    start = time.perf_counter()
    model.predict({"wind_speed": v})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
