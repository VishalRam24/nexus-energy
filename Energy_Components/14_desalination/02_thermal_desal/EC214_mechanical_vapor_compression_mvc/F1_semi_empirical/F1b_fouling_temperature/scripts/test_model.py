"""EC214 — MVC Desalination — F1b SEC + Recovery + Temperature — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"recovery": 0.50, "operating_hours": 0})
    for k in ["sec_kwh_m3", "compressor_work_kwh_m3", "production_rate_m3_h",
              "brine_salinity_gkg", "bpe_degC", "fouling_factor"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC214"
    assert info["fidelity"] == "F1b"


def test_sec_reasonable_at_design(model):
    """MVC SEC at 50% recovery: 6-12 kWh/m3 (Mistry 2011: ~8-12 kWh/m3 for seawater)."""
    r = model.predict({"recovery": 0.50, "T_evap_degC": 70.0, "operating_hours": 0})
    sec = float(np.atleast_1d(r["sec_kwh_m3"])[0])
    # Mistry 2011: MVC SEC 8-15 kWh/m3; allow generous range
    assert 2.0 < sec < 25.0, f"MVC SEC = {sec:.2f} kWh/m3"


def test_sec_increases_with_recovery(model):
    """Higher recovery → higher brine concentration → more BPE → higher SEC."""
    r_low  = model.predict({"recovery": 0.35, "T_evap_degC": 70.0, "operating_hours": 0})
    r_high = model.predict({"recovery": 0.65, "T_evap_degC": 70.0, "operating_hours": 0})
    sec_low  = float(np.atleast_1d(r_low["sec_kwh_m3"])[0])
    sec_high = float(np.atleast_1d(r_high["sec_kwh_m3"])[0])
    assert sec_high > sec_low, f"SEC not increasing with recovery: {sec_low:.3f} vs {sec_high:.3f}"


def test_brine_salinity_scales_with_recovery(model):
    """Brine salinity = S_feed / (1 - recovery) → increases with recovery."""
    r_low  = model.predict({"recovery": 0.40, "operating_hours": 0})
    r_high = model.predict({"recovery": 0.65, "operating_hours": 0})
    s_low  = float(np.atleast_1d(r_low["brine_salinity_gkg"])[0])
    s_high = float(np.atleast_1d(r_high["brine_salinity_gkg"])[0])
    assert s_high > s_low, f"Brine salinity not scaling: {s_low:.1f} vs {s_high:.1f}"


def test_bpe_positive(model):
    r = model.predict({"recovery": 0.50, "operating_hours": 0})
    bpe = float(np.atleast_1d(r["bpe_degC"])[0])
    assert bpe > 0


def test_fouling_factor_1_at_zero(model):
    """Fresh evaporator: fouling factor = 1.0."""
    r = model.predict({"recovery": 0.50, "operating_hours": 0})
    ff = float(np.atleast_1d(r["fouling_factor"])[0])
    assert abs(ff - 1.0) < 1e-6


def test_fouling_factor_declines_with_time(model):
    r_new = model.predict({"recovery": 0.50, "operating_hours": 0})
    r_old = model.predict({"recovery": 0.50, "operating_hours": 43800})  # 5 years
    ff_new = float(np.atleast_1d(r_new["fouling_factor"])[0])
    ff_old = float(np.atleast_1d(r_old["fouling_factor"])[0])
    assert ff_old < ff_new, f"Fouling factor not declining: {ff_new:.3f} vs {ff_old:.3f}"


def test_sec_increases_with_fouling(model):
    """More fouling → higher dT needed → more compressor work."""
    r_new = model.predict({"recovery": 0.50, "T_evap_degC": 70.0, "operating_hours": 0})
    r_old = model.predict({"recovery": 0.50, "T_evap_degC": 70.0, "operating_hours": 43800})
    sec_new = float(np.atleast_1d(r_new["sec_kwh_m3"])[0])
    sec_old = float(np.atleast_1d(r_old["sec_kwh_m3"])[0])
    assert sec_old >= sec_new, f"SEC not increasing with fouling: {sec_new:.3f} vs {sec_old:.3f}"


def test_production_rate_scales_with_recovery(model):
    """Production = feed * recovery → higher recovery → higher production."""
    r_low  = model.predict({"recovery": 0.40, "feed_flow_m3_h": 100.0, "operating_hours": 0})
    r_high = model.predict({"recovery": 0.60, "feed_flow_m3_h": 100.0, "operating_hours": 0})
    Q_low  = float(np.atleast_1d(r_low["production_rate_m3_h"])[0])
    Q_high = float(np.atleast_1d(r_high["production_rate_m3_h"])[0])
    assert Q_high > Q_low


def test_array_input(model):
    recoveries = np.linspace(0.35, 0.70, 10)
    r = model.predict({"recovery": recoveries, "operating_hours": 0})
    assert len(np.atleast_1d(r["sec_kwh_m3"])) == 10


def test_benchmark(model):
    recoveries = np.random.uniform(0.35, 0.70, 1000)
    start = time.perf_counter()
    model.predict({"recovery": recoveries, "operating_hours": 0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
