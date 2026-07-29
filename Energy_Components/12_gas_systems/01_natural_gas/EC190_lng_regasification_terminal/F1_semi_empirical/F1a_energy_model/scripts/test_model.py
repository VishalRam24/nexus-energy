"""EC190 — LNG Regasification Terminal — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"sendout_rate_ton_per_h": 500.0})
    for k in ["power_demand_kw", "cold_recovery_kw", "net_power_kw",
               "gas_sendout_kg_per_s", "gas_sendout_m3_per_day", "net_sec_kwh_per_ton"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC190"
    assert info["fidelity"] == "F1a"


def test_power_positive(model):
    r = model.predict({"sendout_rate_ton_per_h": 500.0})
    assert float(r["power_demand_kw"]) > 0
    assert float(r["gas_sendout_kg_per_s"]) > 0


def test_zero_sendout_zero_power(model):
    r = model.predict({"sendout_rate_ton_per_h": 0.0})
    assert float(r["power_demand_kw"]) == pytest.approx(0.0, abs=1e-6)
    assert float(r["gas_sendout_kg_per_s"]) == pytest.approx(0.0, abs=1e-9)


def test_power_scales_linearly_with_sendout(model):
    """P_demand = SEC × m: power must be linear in sendout rate."""
    m_arr = np.array([100.0, 200.0, 400.0, 800.0])
    r = model.predict({"sendout_rate_ton_per_h": m_arr, "sec_kwh_per_ton": 50.0})
    ratios = r["power_demand_kw"] / r["power_demand_kw"][0]
    expected = m_arr / m_arr[0]
    np.testing.assert_allclose(ratios, expected, rtol=1e-9)


def test_sec_in_realistic_range(model):
    """SEC must be 20-100 kWh/ton for credible heat sources."""
    for sec in [25.0, 50.0, 80.0, 100.0]:
        r = model.predict({"sendout_rate_ton_per_h": 500.0, "sec_kwh_per_ton": sec})
        assert float(r["net_sec_kwh_per_ton"]) <= sec + 1e-9  # net <= gross
        assert float(r["net_sec_kwh_per_ton"]) >= 0.0


def test_cold_recovery_reduces_net_power(model):
    """Non-zero cold recovery must reduce net power vs gross."""
    r0 = model.predict({"sendout_rate_ton_per_h": 500.0, "f_cold": 0.0})
    r1 = model.predict({"sendout_rate_ton_per_h": 500.0, "f_cold": 0.3})
    assert float(r1["net_power_kw"]) < float(r0["net_power_kw"])
    assert float(r1["cold_recovery_kw"]) > 0.0


def test_cold_recovery_bounded(model):
    """Cold recovery cannot exceed gross power demand."""
    r = model.predict({"sendout_rate_ton_per_h": 500.0, "f_cold": 0.5, "sec_kwh_per_ton": 50.0})
    assert float(r["net_power_kw"]) >= 0.0


def test_higher_ambient_increases_cold_recovery(model):
    """Higher ambient T → larger Carnot factor → more cold exergy available."""
    T_arr = np.array([270.0, 280.0, 290.0, 300.0])
    r = model.predict({"sendout_rate_ton_per_h": 500.0, "T_ambient_K": T_arr, "f_cold": 0.3})
    assert np.all(np.diff(r["cold_recovery_kw"]) > 0)


def test_gas_sendout_unit_consistency(model):
    """Mass flow and volumetric flow must be consistent."""
    r = model.predict({"sendout_rate_ton_per_h": 500.0})
    kg_s = float(r["gas_sendout_kg_per_s"])
    m3_day = float(r["gas_sendout_m3_per_day"])
    # At rho_std ≈ 0.717 kg/m³: m³/day = (kg/s) / 0.717 * 86400
    expected_m3_day = kg_s / 0.717 * 86400.0
    assert m3_day == pytest.approx(expected_m3_day, rel=1e-6)


def test_power_demand_formula(model):
    """P [kW] = SEC [kWh/ton] × sendout [ton/h]."""
    r = model.predict({"sendout_rate_ton_per_h": 300.0, "sec_kwh_per_ton": 60.0})
    assert float(r["power_demand_kw"]) == pytest.approx(300.0 * 60.0, rel=1e-9)


def test_benchmark(model):
    rng = np.random.default_rng(42)
    m = rng.uniform(50, 2000, 1000)
    sec = rng.uniform(20, 100, 1000)
    start = time.perf_counter()
    model.predict({"sendout_rate_ton_per_h": m, "sec_kwh_per_ton": sec})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
