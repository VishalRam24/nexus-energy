"""EC203 — Membrane CO2 Separation — F1b Part-Load + Aging — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"capture_rate_target": 0.60, "operating_hours": 0})
    for k in ["co2_captured_kg_h", "sec_kwh_ton", "compression_kwh_ton",
              "total_energy_kwh_ton", "permeance_pct"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC203"
    assert info["fidelity"] == "F1b"


def test_permeance_100pct_at_start(model):
    """Fresh membrane at reference T: permeance = 100%."""
    r = model.predict({"operating_hours": 0, "T_feed_degC": 25.0,
                       "capture_rate_target": 0.60})
    perm = float(np.atleast_1d(r["permeance_pct"])[0])
    assert abs(perm - 100.0) < 1.0, f"Permeance at t=0, T=25C = {perm:.2f}%"


def test_permeance_declines_with_time(model):
    """Membrane permeance should decline due to physical aging."""
    hours_list = [0, 8760, 26280, 52560]
    perms = []
    for h in hours_list:
        r = model.predict({"operating_hours": h, "T_feed_degC": 25.0,
                           "capture_rate_target": 0.60})
        perms.append(float(np.atleast_1d(r["permeance_pct"])[0]))
    assert all(perms[i] >= perms[i + 1] for i in range(len(perms) - 1)), \
        f"Permeance not declining: {perms}"


def test_aging_rate_6pct_per_year(model):
    """After 1 year: permeance = exp(-0.06)*100 ~ 94.2%."""
    r = model.predict({"operating_hours": 8760, "T_feed_degC": 25.0,
                       "capture_rate_target": 0.60})
    perm = float(np.atleast_1d(r["permeance_pct"])[0])
    expected = np.exp(-0.06) * 100.0
    assert abs(perm - expected) < 2.0, f"1-yr permeance = {perm:.2f}%, expected ~{expected:.2f}%"


def test_temperature_affects_permeance(model):
    """Temperature changes permeance per Arrhenius relationship.
    For glassy polymers (E_a = -10 kJ/mol): permeance decreases at higher T
    (lower T → higher permeability due to CO2 condensability in glassy polymers).
    Baker & Low (2014) Macromolecules: CO2 permeability in glassy polymers often
    decreases with T (E_a < 0 for condensable gases in glassy polymers).
    RATIONALE: E_a < 0 means permeance is higher at LOWER temperatures; the
    test verifies directional consistency, not a specific sign.
    """
    r_cold = model.predict({"T_feed_degC": 20.0, "operating_hours": 0,
                            "capture_rate_target": 0.60})
    r_warm = model.predict({"T_feed_degC": 60.0, "operating_hours": 0,
                            "capture_rate_target": 0.60})
    perm_cold = float(np.atleast_1d(r_cold["permeance_pct"])[0])
    perm_warm = float(np.atleast_1d(r_warm["permeance_pct"])[0])
    # E_a = -10 kJ/mol: lower T → higher permeance (glassy polymer behavior)
    # Verify temperature has a measurable effect on permeance
    assert abs(perm_warm - perm_cold) > 1.0, \
        f"Temperature has no effect on permeance: cold={perm_cold:.1f}%, warm={perm_warm:.1f}%"
    # With E_a=-10 kJ/mol and T_ref=25°C: cold (20°C) > warm (60°C)
    assert perm_cold > perm_warm, \
        f"For E_a<0 (glassy polymer), cold permeance should > warm: {perm_cold:.1f}% vs {perm_warm:.1f}%"


def test_sec_increases_with_capture_rate(model):
    """SEC should increase exponentially with capture rate."""
    crs = [0.40, 0.60, 0.80, 0.90]
    secs = []
    for cr in crs:
        r = model.predict({"capture_rate_target": cr, "operating_hours": 0})
        secs.append(float(np.atleast_1d(r["sec_kwh_ton"])[0]))
    assert all(secs[i] < secs[i + 1] for i in range(len(secs) - 1)), \
        f"SEC not increasing with CR: {secs}"


def test_sec_reasonable_range(model):
    """SEC at CR=0.60: ~300-600 kWh/tCO2 (Merkel 2010 range)."""
    r = model.predict({"capture_rate_target": 0.60, "operating_hours": 0})
    sec = float(np.atleast_1d(r["sec_kwh_ton"])[0])
    # Merkel 2010: membrane SEC ~100-500 kWh/tCO2; allow generous range
    assert 100.0 < sec < 1500.0, f"SEC at CR=0.60 = {sec:.1f} kWh/tCO2"


def test_compression_positive(model):
    """Compression energy should be positive."""
    r = model.predict({"capture_rate_target": 0.60, "operating_hours": 0,
                       "injection_pressure_bar": 100.0})
    e = float(np.atleast_1d(r["compression_kwh_ton"])[0])
    assert e > 0, f"Compression energy = {e:.4f}"


def test_compression_increases_with_pressure(model):
    """Higher injection pressure → more compression work."""
    r_low  = model.predict({"capture_rate_target": 0.60, "operating_hours": 0,
                            "injection_pressure_bar": 10.0})
    r_high = model.predict({"capture_rate_target": 0.60, "operating_hours": 0,
                            "injection_pressure_bar": 150.0})
    e_low  = float(np.atleast_1d(r_low["compression_kwh_ton"])[0])
    e_high = float(np.atleast_1d(r_high["compression_kwh_ton"])[0])
    assert e_high > e_low, f"Compression not increasing with pressure: {e_low:.2f} vs {e_high:.2f}"


def test_co2_captured_positive(model):
    """CO2 captured must be positive."""
    r = model.predict({"co2_feed_fraction": 0.12, "feed_flow_mol_s": 100.0,
                       "capture_rate_target": 0.60, "operating_hours": 0})
    co2 = float(np.atleast_1d(r["co2_captured_kg_h"])[0])
    assert co2 > 0, f"CO2 captured = {co2:.4f}"


def test_co2_scales_with_capture_rate(model):
    """CO2 captured should scale proportionally with target capture rate."""
    r1 = model.predict({"feed_flow_mol_s": 100.0, "co2_feed_fraction": 0.12,
                        "capture_rate_target": 0.40, "operating_hours": 0})
    r2 = model.predict({"feed_flow_mol_s": 100.0, "co2_feed_fraction": 0.12,
                        "capture_rate_target": 0.80, "operating_hours": 0})
    co2_1 = float(np.atleast_1d(r1["co2_captured_kg_h"])[0])
    co2_2 = float(np.atleast_1d(r2["co2_captured_kg_h"])[0])
    assert co2_2 > co2_1, f"CO2 not scaling with CR: {co2_1:.1f} vs {co2_2:.1f}"


def test_total_energy_positive(model):
    r = model.predict({"capture_rate_target": 0.60, "operating_hours": 0})
    e = float(np.atleast_1d(r["total_energy_kwh_ton"])[0])
    assert e > 0


def test_array_input(model):
    """Model should handle array capture rate inputs."""
    crs = np.linspace(0.40, 0.85, 10)
    r = model.predict({"capture_rate_target": crs, "operating_hours": 0})
    assert len(np.atleast_1d(r["sec_kwh_ton"])) == 10


def test_benchmark(model):
    crs = np.random.uniform(0.40, 0.85, 1000)
    start = time.perf_counter()
    model.predict({"capture_rate_target": crs, "operating_hours": 5000})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
