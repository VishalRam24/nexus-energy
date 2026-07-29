"""EC206 — CO2 Mineralization — F1b Part-Load Conversion Degradation — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"PLR": 1.0, "operating_hours": 0})
    for k in ["co2_stored_kg_h", "sec_kwh_tco2", "conversion_efficiency",
              "conversion_relative_pct", "carbonate_product_kg_h",
              "mineral_feed_t_per_tco2"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC206"
    assert info["fidelity"] == "F1b"


def test_conversion_100pct_at_start(model):
    """Fresh reactor: relative conversion = 100%."""
    r = model.predict({"PLR": 1.0, "operating_hours": 0})
    c = float(np.atleast_1d(r["conversion_relative_pct"])[0])
    assert abs(c - 100.0) < 1e-6, f"Conversion at t=0 = {c:.2f}%"


def test_conversion_design_value(model):
    """Design conversion efficiency ~0.80."""
    r = model.predict({"PLR": 1.0, "operating_hours": 0})
    eta = float(np.atleast_1d(r["conversion_efficiency"])[0])
    assert abs(eta - 0.80) < 0.01, f"Conversion at design = {eta:.3f}"


def test_conversion_degrades_with_time(model):
    """Conversion efficiency decreases with operating hours."""
    hours_list = [0, 5000, 15000, 30000]
    etas = [float(np.atleast_1d(model.predict({"PLR": 1.0, "operating_hours": h})["conversion_efficiency"])[0])
            for h in hours_list]
    assert all(etas[i] >= etas[i + 1] - 1e-6 for i in range(len(etas) - 1)), \
        f"Conversion not declining: {etas}"


def test_conversion_floor(model):
    """Conversion has a minimum floor (surface passivation limit)."""
    r = model.predict({"PLR": 1.0, "operating_hours": 1e6})
    eta = float(np.atleast_1d(r["conversion_efficiency"])[0])
    assert eta >= 0.35, f"Conversion floor too low: {eta:.3f}"


def test_sec_increases_at_part_load(model):
    """SEC increases at lower PLR."""
    sec_full = float(np.atleast_1d(model.predict({"PLR": 1.0, "operating_hours": 0})["sec_kwh_tco2"])[0])
    sec_part = float(np.atleast_1d(model.predict({"PLR": 0.3, "operating_hours": 0})["sec_kwh_tco2"])[0])
    assert sec_part > sec_full, f"SEC: full={sec_full:.1f}, part={sec_part:.1f}"


def test_sec_increases_with_degradation(model):
    """SEC increases as conversion degrades."""
    sec_fresh = float(np.atleast_1d(model.predict({"PLR": 1.0, "operating_hours": 0})["sec_kwh_tco2"])[0])
    sec_aged = float(np.atleast_1d(model.predict({"PLR": 1.0, "operating_hours": 20000})["sec_kwh_tco2"])[0])
    assert sec_aged >= sec_fresh - 1e-6


def test_co2_stored_positive(model):
    r = model.predict({"co2_flow_kg_h": 1000.0, "PLR": 0.5, "operating_hours": 0})
    co2 = float(np.atleast_1d(r["co2_stored_kg_h"])[0])
    assert co2 > 0


def test_co2_scales_with_plr(model):
    co2_half = float(np.atleast_1d(model.predict({"co2_flow_kg_h": 1000.0, "PLR": 0.5, "operating_hours": 0})["co2_stored_kg_h"])[0])
    co2_full = float(np.atleast_1d(model.predict({"co2_flow_kg_h": 1000.0, "PLR": 1.0, "operating_hours": 0})["co2_stored_kg_h"])[0])
    assert co2_full > co2_half


def test_carbonate_product_positive(model):
    r = model.predict({"co2_flow_kg_h": 1000.0, "PLR": 1.0, "operating_hours": 0})
    carb = float(np.atleast_1d(r["carbonate_product_kg_h"])[0])
    assert carb > 0


def test_mineral_feed_increases_with_degradation(model):
    """More mineral needed when conversion drops."""
    mf_fresh = float(np.atleast_1d(model.predict({"PLR": 1.0, "operating_hours": 0})["mineral_feed_t_per_tco2"])[0])
    mf_aged = float(np.atleast_1d(model.predict({"PLR": 1.0, "operating_hours": 20000})["mineral_feed_t_per_tco2"])[0])
    assert mf_aged >= mf_fresh - 1e-6


def test_array_input(model):
    PLRs = np.linspace(0.3, 1.0, 10)
    r = model.predict({"PLR": PLRs, "operating_hours": 0})
    assert len(np.atleast_1d(r["sec_kwh_tco2"])) == 10


def test_benchmark(model):
    PLRs = np.random.uniform(0.3, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"PLR": PLRs, "operating_hours": 5000})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
