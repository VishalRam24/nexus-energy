"""EC190 — LNG Regasification Terminal — F1b Ambient+Part-Load — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"sendout_rate_ton_per_h": 100.0, "PLR": 1.0})
    for k in ["gross_sec_kwh_per_ton", "net_sec_kwh_per_ton",
              "net_power_kw", "cold_recovery_kw", "gas_sendout_kg_per_s"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC190"
    assert info["fidelity"] == "F1b"


def test_sec_at_design(model):
    """At PLR=1.0, design T_amb: gross SEC should match base ~50 kWh/ton."""
    r = model.predict({"sendout_rate_ton_per_h": 100.0, "PLR": 1.0,
                        "T_ambient_K": 283.15})
    sec = float(np.atleast_1d(r["gross_sec_kwh_per_ton"])[0])
    assert 40.0 <= sec <= 70.0, f"Design SEC = {sec:.2f} kWh/ton"


def test_sec_increases_at_partload(model):
    """SEC should increase at lower PLR (off-design penalty)."""
    r_full = model.predict({"sendout_rate_ton_per_h": 100.0, "PLR": 1.0})
    r_half = model.predict({"sendout_rate_ton_per_h": 100.0, "PLR": 0.5})
    sec_full = float(np.atleast_1d(r_full["gross_sec_kwh_per_ton"])[0])
    sec_half = float(np.atleast_1d(r_half["gross_sec_kwh_per_ton"])[0])
    assert sec_half > sec_full, f"SEC not higher at part-load: {sec_full:.2f} vs {sec_half:.2f}"


def test_sec_lower_at_warm_ambient(model):
    """Warmer ambient → lower SEC (less heating energy needed)."""
    r_cold = model.predict({"sendout_rate_ton_per_h": 100.0, "PLR": 1.0, "T_ambient_K": 268.0})
    r_warm = model.predict({"sendout_rate_ton_per_h": 100.0, "PLR": 1.0, "T_ambient_K": 303.0})
    sec_cold = float(np.atleast_1d(r_cold["gross_sec_kwh_per_ton"])[0])
    sec_warm = float(np.atleast_1d(r_warm["gross_sec_kwh_per_ton"])[0])
    assert sec_warm < sec_cold, f"Warmer ambient not reducing SEC: {sec_cold:.2f} vs {sec_warm:.2f}"


def test_net_sec_less_than_gross(model):
    """Net SEC (after cold recovery) should be less than gross SEC."""
    r = model.predict({"sendout_rate_ton_per_h": 500.0, "PLR": 1.0, "T_ambient_K": 283.15})
    gross = float(np.atleast_1d(r["gross_sec_kwh_per_ton"])[0])
    net = float(np.atleast_1d(r["net_sec_kwh_per_ton"])[0])
    assert net <= gross + 1e-6, f"Net SEC {net:.2f} > Gross SEC {gross:.2f}"


def test_cold_recovery_positive(model):
    """Cold energy recovery must be positive."""
    r = model.predict({"sendout_rate_ton_per_h": 100.0, "PLR": 1.0})
    cold = float(np.atleast_1d(r["cold_recovery_kw"])[0])
    assert cold > 0, f"Cold recovery = {cold:.2f} kW"


def test_gas_sendout_scales_linearly(model):
    """Gas sendout scales linearly with sendout rate."""
    r1 = model.predict({"sendout_rate_ton_per_h": 100.0, "PLR": 1.0})
    r2 = model.predict({"sendout_rate_ton_per_h": 200.0, "PLR": 1.0})
    g1 = float(np.atleast_1d(r1["gas_sendout_kg_per_s"])[0])
    g2 = float(np.atleast_1d(r2["gas_sendout_kg_per_s"])[0])
    assert abs(g2 / g1 - 2.0) < 0.01, f"Gas sendout not scaling 2x: {g1:.4f} vs {g2:.4f}"


def test_net_power_positive(model):
    """Net power must be non-negative."""
    for plr in [0.3, 0.5, 1.0]:
        r = model.predict({"sendout_rate_ton_per_h": 100.0, "PLR": plr})
        P = float(np.atleast_1d(r["net_power_kw"])[0])
        assert P >= 0, f"Net power negative at PLR={plr}: {P:.2f}"


def test_array_input(model):
    """Handle array PLR inputs."""
    PLRs = np.linspace(0.3, 1.0, 10)
    r = model.predict({"sendout_rate_ton_per_h": 100.0, "PLR": PLRs})
    assert len(np.atleast_1d(r["gross_sec_kwh_per_ton"])) == 10


def test_benchmark(model):
    PLRs = np.random.uniform(0.3, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"sendout_rate_ton_per_h": 100.0, "PLR": PLRs})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
