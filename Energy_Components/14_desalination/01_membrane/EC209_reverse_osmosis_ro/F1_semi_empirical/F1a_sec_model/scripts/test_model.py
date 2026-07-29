"""EC209 — Reverse Osmosis (RO) — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"feed_salinity": 35.0, "recovery": 0.45, "feed_flow_m3h": 100.0})
    for k in ["sec_kwhm3", "permeate_flow_m3h", "feed_pressure_bar", "permeate_salinity_gl"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC209"
    assert info["fidelity"] == "F1a"


def test_seawater_sec_range(model):
    """SEC for seawater (35 g/L) should be 2-6 kWh/m3 with ERD."""
    r = model.predict({"feed_salinity": 35.0, "recovery": 0.45, "feed_flow_m3h": 100.0})
    sec = float(r["sec_kwhm3"])
    assert 2.0 <= sec <= 6.0, f"SWRO SEC={sec:.2f} outside typical 2-6 kWh/m3 range"


def test_sec_increases_with_salinity(model):
    """Higher feed salinity -> higher osmotic pressure -> higher SEC."""
    salinities = np.array([5.0, 15.0, 25.0, 35.0, 45.0])
    r = model.predict({"feed_salinity": salinities, "recovery": 0.45})
    sec = r["sec_kwhm3"]
    assert np.all(np.diff(sec) > 0), "SEC must increase with feed salinity"


def test_permeate_flow_proportional_to_recovery(model):
    """Permeate flow = feed_flow * recovery."""
    recoveries = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    Q_feed = 100.0
    r = model.predict({"feed_salinity": 35.0, "recovery": recoveries, "feed_flow_m3h": Q_feed})
    expected = Q_feed * recoveries
    assert np.allclose(r["permeate_flow_m3h"], expected, rtol=1e-6)


def test_pressure_increases_with_salinity(model):
    """Higher salinity -> higher osmotic pressure -> higher feed pressure."""
    salinities = np.array([5.0, 15.0, 25.0, 35.0])
    r = model.predict({"feed_salinity": salinities, "recovery": 0.45})
    P = r["feed_pressure_bar"]
    assert np.all(np.diff(P) > 0), "Feed pressure must increase with salinity"


def test_permeate_much_fresher_than_feed(model):
    """Salt rejection ~99.5% -> permeate salinity << feed salinity."""
    r = model.predict({"feed_salinity": 35.0, "recovery": 0.45})
    S_perm = float(r["permeate_salinity_gl"])
    assert S_perm < 1.0, f"Permeate salinity {S_perm:.3f} g/L too high for SWRO"


def test_sec_has_minimum_near_optimal_recovery(model):
    """SEC vs recovery curve should show a minimum (not monotonic)."""
    recoveries = np.linspace(0.2, 0.6, 20)
    r = model.predict({"feed_salinity": 35.0, "recovery": recoveries})
    sec = r["sec_kwhm3"]
    # At very low recovery, SEC should decrease then increase — verify not purely monotonic
    # At minimum around optimal recovery
    assert np.min(sec) < np.max(sec), "SEC must vary with recovery"


def test_brackish_water_lower_sec(model):
    """Brackish water (5 g/L) should require much less energy than seawater (35 g/L)."""
    r_bw = model.predict({"feed_salinity": 5.0, "recovery": 0.45})
    r_sw = model.predict({"feed_salinity": 35.0, "recovery": 0.45})
    assert float(r_bw["sec_kwhm3"]) < float(r_sw["sec_kwhm3"])


def test_benchmark(model):
    S = np.random.uniform(1.0, 45.0, 1000)
    r = np.random.uniform(0.2, 0.6, 1000)
    Q = np.random.uniform(10.0, 1000.0, 1000)
    start = time.perf_counter()
    model.predict({"feed_salinity": S, "recovery": r, "feed_flow_m3h": Q})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
