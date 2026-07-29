"""EC101 — CCGT — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    for k in ["power_mw", "efficiency", "fuel_rate_kgs", "exhaust_temp_c"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC101"
    assert info["fidelity"] == "F1a"


def test_efficiency_below_07(model):
    """CCGT net efficiency must not exceed 70% (physical upper bound)."""
    plr   = np.linspace(0.3, 1.0, 30)
    T_amb = np.linspace(-20, 50, 30)
    r = model.predict({"part_load_ratio": plr, "ambient_temp": T_amb})
    assert np.all(r["efficiency"] < 0.70), "Efficiency exceeded 70% — check parameters"


def test_fuel_greater_than_power(model):
    """Fuel energy input must exceed electrical output (eta < 1)."""
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    p    = float(r["power_mw"])                         # MW_e
    fuel_mw = float(r["fuel_rate_kgs"]) * 50.0         # MW_th
    assert fuel_mw > p, f"Fuel {fuel_mw:.1f} MW_th <= Power {p:.1f} MW_e"


def test_efficiency_drops_low_plr(model):
    """Efficiency at PLR=0.3 must be lower than at PLR=1.0 (ISO conditions)."""
    r_full = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    r_part = model.predict({"part_load_ratio": 0.3, "ambient_temp": 15.0})
    assert float(r_part["efficiency"]) < float(r_full["efficiency"])


def test_efficiency_drops_high_tamb(model):
    """Efficiency at T_amb=50C must be lower than at ISO (15C)."""
    r_iso  = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    r_hot  = model.predict({"part_load_ratio": 1.0, "ambient_temp": 50.0})
    assert float(r_hot["efficiency"]) < float(r_iso["efficiency"])


def test_rated_iso_efficiency(model):
    """At PLR=1, T_amb=15C, efficiency should be ~0.64."""
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    eta = float(r["efficiency"])
    assert 0.60 < eta < 0.68, f"Expected ~0.64, got {eta:.4f}"


def test_power_scales_with_plr(model):
    """Electrical output must scale linearly with PLR."""
    plr = np.array([0.3, 0.5, 0.7, 1.0])
    r = model.predict({"part_load_ratio": plr, "ambient_temp": 15.0})
    P = r["power_mw"]
    ratios = P / P[-1]
    np.testing.assert_allclose(ratios, plr / 1.0, rtol=1e-6)


def test_exhaust_temp_increases_with_plr(model):
    """Exhaust temperature rises with increasing PLR."""
    plr = np.linspace(0.3, 1.0, 20)
    r = model.predict({"part_load_ratio": plr, "ambient_temp": 15.0})
    assert np.all(np.diff(r["exhaust_temp_c"]) > 0)


def test_fuel_rate_positive(model):
    """Fuel consumption must always be positive."""
    plr   = np.linspace(0.3, 1.0, 20)
    T_amb = np.linspace(-20, 50, 20)
    r = model.predict({"part_load_ratio": plr, "ambient_temp": T_amb})
    assert np.all(r["fuel_rate_kgs"] > 0)


def test_benchmark(model):
    plr   = np.random.uniform(0.3, 1.0, 1000)
    T_amb = np.random.uniform(-20, 50, 1000)
    start = time.perf_counter()
    model.predict({"part_load_ratio": plr, "ambient_temp": T_amb})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
