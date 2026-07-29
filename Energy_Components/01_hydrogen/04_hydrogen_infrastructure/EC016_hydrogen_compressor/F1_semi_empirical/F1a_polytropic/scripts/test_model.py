"""EC016 — Hydrogen Compressor — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"P_inlet": 20.0, "P_outlet": 700.0})
    for k in ["specific_work_kJ_per_kg", "sec_kwh_per_kg", "shaft_power_kw",
              "stage_pressure_ratio", "stage_discharge_T_K", "compression_efficiency"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC016"
    assert info["fidelity"] == "F1a"


def test_work_positive(model):
    r = model.predict({"P_inlet": 20.0, "P_outlet": 700.0})
    assert float(r["specific_work_kJ_per_kg"]) > 0
    assert float(r["sec_kwh_per_kg"]) > 0
    assert float(r["shaft_power_kw"]) > 0


def test_work_increases_with_outlet_pressure(model):
    P_out = np.array([100, 200, 350, 500, 700, 900])
    r = model.predict({"P_inlet": 20.0, "P_outlet": P_out})
    assert np.all(np.diff(r["specific_work_kJ_per_kg"]) > 0)
    assert np.all(np.diff(r["sec_kwh_per_kg"]) > 0)


def test_work_decreases_with_inlet_pressure(model):
    """Higher suction pressure => less work for same discharge."""
    P_in = np.array([5, 10, 20, 30, 50])
    r = model.predict({"P_inlet": P_in, "P_outlet": 700.0})
    assert np.all(np.diff(r["specific_work_kJ_per_kg"]) < 0)


def test_stage_ratio_consistent(model):
    """N=4 stages: PR_stage^4 = PR_total."""
    r = model.predict({"P_inlet": 20.0, "P_outlet": 700.0})
    PR_stage = float(r["stage_pressure_ratio"])
    PR_total = 700.0 / 20.0
    assert PR_stage ** 4 == pytest.approx(PR_total, rel=1e-6)


def test_stage_discharge_above_inlet(model):
    """Discharge T must exceed inlet T (compression heats gas)."""
    r = model.predict({"P_inlet": 20.0, "P_outlet": 700.0, "T_inlet": 298.15})
    assert float(r["stage_discharge_T_K"]) > 298.15


def test_shaft_power_scales_with_mass_flow(model):
    r1 = model.predict({"P_inlet": 20.0, "P_outlet": 700.0, "m_dot": 0.01})
    r2 = model.predict({"P_inlet": 20.0, "P_outlet": 700.0, "m_dot": 0.02})
    ratio = float(r2["shaft_power_kw"]) / float(r1["shaft_power_kw"])
    assert ratio == pytest.approx(2.0, rel=1e-6)


def test_sec_in_realistic_range(model):
    """20 -> 700 bar SEC should be ~1-3 kWh/kg for H2 compression (Sdanghi 2019)."""
    r = model.predict({"P_inlet": 20.0, "P_outlet": 700.0})
    sec = float(r["sec_kwh_per_kg"])
    assert 0.5 < sec < 5.0, f"SEC={sec:.3f} kWh/kg outside realistic range"


def test_no_compression_when_pin_equals_pout(model):
    """w should be ~0 when PR=1."""
    r = model.predict({"P_inlet": 20.0, "P_outlet": 20.0})
    assert float(r["specific_work_kJ_per_kg"]) == pytest.approx(0.0, abs=1e-6)


def test_efficiency_below_unity(model):
    r = model.predict({"P_inlet": 20.0, "P_outlet": 700.0})
    eta = float(r["compression_efficiency"])
    assert 0.0 < eta < 1.0


def test_benchmark(model):
    P_in = np.full(1000, 20.0)
    P_out = np.random.uniform(50, 900, 1000)
    start = time.perf_counter()
    model.predict({"P_inlet": P_in, "P_outlet": P_out})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
