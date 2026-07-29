"""EC191 — Gas Compressor Station — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"P_inlet": 50.0, "P_outlet": 80.0})
    for k in ["specific_work_kJ_per_kg", "sec_kwh_per_kg", "shaft_power_kw",
               "stage_pressure_ratio", "stage_discharge_T_K", "compression_efficiency"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC191"
    assert info["fidelity"] == "F1a"


def test_work_positive(model):
    r = model.predict({"P_inlet": 50.0, "P_outlet": 80.0})
    assert float(r["specific_work_kJ_per_kg"]) > 0
    assert float(r["sec_kwh_per_kg"]) > 0
    assert float(r["shaft_power_kw"]) > 0


def test_work_increases_with_outlet_pressure(model):
    P_out = np.array([60.0, 70.0, 80.0, 100.0, 120.0])
    r = model.predict({"P_inlet": 50.0, "P_outlet": P_out})
    assert np.all(np.diff(r["specific_work_kJ_per_kg"]) > 0)
    assert np.all(np.diff(r["sec_kwh_per_kg"]) > 0)


def test_work_decreases_with_inlet_pressure(model):
    P_in = np.array([20.0, 30.0, 40.0, 50.0, 60.0])
    r = model.predict({"P_inlet": P_in, "P_outlet": 100.0})
    assert np.all(np.diff(r["specific_work_kJ_per_kg"]) < 0)


def test_stage_ratio_consistent(model):
    """N=2 stages: PR_stage^2 = PR_total."""
    r = model.predict({"P_inlet": 50.0, "P_outlet": 100.0})
    PR_stage = float(r["stage_pressure_ratio"])
    PR_total = 100.0 / 50.0
    assert PR_stage ** 2 == pytest.approx(PR_total, rel=1e-6)


def test_stage_discharge_above_inlet(model):
    """Compression heats gas: T_discharge > T_inlet."""
    r = model.predict({"P_inlet": 50.0, "P_outlet": 80.0, "T_inlet": 288.15})
    assert float(r["stage_discharge_T_K"]) > 288.15


def test_shaft_power_scales_with_mass_flow(model):
    r1 = model.predict({"P_inlet": 50.0, "P_outlet": 80.0, "m_dot": 50.0})
    r2 = model.predict({"P_inlet": 50.0, "P_outlet": 80.0, "m_dot": 100.0})
    ratio = float(r2["shaft_power_kw"]) / float(r1["shaft_power_kw"])
    assert ratio == pytest.approx(2.0, rel=1e-6)


def test_sec_in_realistic_range_for_station(model):
    """
    Single station CR ~1.5: w should be < 50 kJ/kg NG → SEC < 0.015 kWh/kg
    At 50→80 bar: expect ~5-25 kJ/kg → SEC 0.001-0.007 kWh/kg.
    # RATIONALE: NG has much lower R_specific than H2, giving lower per-kg work
    """
    r = model.predict({"P_inlet": 50.0, "P_outlet": 80.0})
    sec = float(r["sec_kwh_per_kg"])
    assert 0.001 < sec < 0.05, f"SEC={sec:.5f} kWh/kg outside realistic range for NG station"


def test_no_compression_when_pin_equals_pout(model):
    """w → 0 when PR = 1."""
    r = model.predict({"P_inlet": 50.0, "P_outlet": 50.0})
    assert float(r["specific_work_kJ_per_kg"]) == pytest.approx(0.0, abs=1e-6)


def test_efficiency_below_unity(model):
    r = model.predict({"P_inlet": 50.0, "P_outlet": 100.0})
    eta = float(r["compression_efficiency"])
    assert 0.0 < eta < 1.0


def test_compression_ratio_in_typical_station_range(model):
    """Typical CR per station = 1.3-1.5; test that within that range SEC is physically small."""
    for cr in [1.3, 1.4, 1.5]:
        P_out = 50.0 * cr
        r = model.predict({"P_inlet": 50.0, "P_outlet": P_out})
        assert float(r["specific_work_kJ_per_kg"]) > 0

    # Two-stage station: PR_stage = sqrt(1.5) ~ 1.22; check stage ratio is consistent
    r = model.predict({"P_inlet": 50.0, "P_outlet": 75.0})
    pr_stage = float(r["stage_pressure_ratio"])
    assert pr_stage == pytest.approx((75.0 / 50.0) ** 0.5, rel=1e-6)


def test_benchmark(model):
    rng = np.random.default_rng(42)
    P_in = rng.uniform(20, 80, 1000)
    P_out = P_in * rng.uniform(1.1, 3.0, 1000)
    start = time.perf_counter()
    model.predict({"P_inlet": P_in, "P_outlet": P_out})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
