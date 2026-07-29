"""EC198 — Post-Combustion Capture — F1b Part-Load Degradation — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"PLR": 1.0, "operating_hours": 0})
    for k in ["co2_captured_kg_h", "reboiler_duty_gj_ton", "electrical_kwh_ton",
              "solvent_degradation_pct", "total_energy_penalty_pct"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC198"
    assert info["fidelity"] == "F1b"


def test_reboiler_duty_at_design(model):
    """At PLR=1.0, fresh solvent, CR=0.90: q should be ~3.5 GJ/tCO2."""
    r = model.predict({"PLR": 1.0, "operating_hours": 0, "capture_rate": 0.90})
    q = float(np.atleast_1d(r["reboiler_duty_gj_ton"])[0])
    assert 3.0 < q < 4.5, f"Reboiler duty at design = {q:.2f}"


def test_reboiler_increases_at_part_load(model):
    """Reboiler duty should increase at lower PLR."""
    PLRs = [0.3, 0.5, 0.7, 1.0]
    qs = []
    for plr in PLRs:
        r = model.predict({"PLR": plr, "operating_hours": 0})
        qs.append(float(np.atleast_1d(r["reboiler_duty_gj_ton"])[0]))
    assert qs[0] > qs[-1], f"Reboiler duty not higher at part-load: {qs}"


def test_reboiler_increases_with_degradation(model):
    """Reboiler duty should increase with operating hours (degraded solvent)."""
    hours_list = [0, 10000, 30000, 50000]
    qs = []
    for h in hours_list:
        r = model.predict({"PLR": 1.0, "operating_hours": h})
        qs.append(float(np.atleast_1d(r["reboiler_duty_gj_ton"])[0]))
    assert all(qs[i] <= qs[i + 1] + 1e-6 for i in range(len(qs) - 1)), \
        f"Reboiler not increasing with degradation: {qs}"


def test_degradation_zero_at_start(model):
    """No degradation at t=0."""
    r = model.predict({"PLR": 1.0, "operating_hours": 0})
    deg = float(np.atleast_1d(r["solvent_degradation_pct"])[0])
    assert abs(deg) < 1e-6


def test_degradation_2pct_per_1000h(model):
    """At 1000h: degradation should be 2%."""
    r = model.predict({"PLR": 1.0, "operating_hours": 1000})
    deg = float(np.atleast_1d(r["solvent_degradation_pct"])[0])
    assert abs(deg - 2.0) < 0.1, f"Degradation at 1000h = {deg:.2f}%, expected 2%"


def test_co2_captured_positive(model):
    """CO2 captured must be positive."""
    r = model.predict({"flue_gas_flow_mol_s": 100.0, "co2_concentration": 0.12,
                        "capture_rate": 0.90, "PLR": 0.5, "operating_hours": 0})
    co2 = float(np.atleast_1d(r["co2_captured_kg_h"])[0])
    assert co2 > 0


def test_co2_scales_with_plr(model):
    """CO2 captured should scale with PLR."""
    r1 = model.predict({"PLR": 0.5, "operating_hours": 0})
    r2 = model.predict({"PLR": 1.0, "operating_hours": 0})
    co2_1 = float(np.atleast_1d(r1["co2_captured_kg_h"])[0])
    co2_2 = float(np.atleast_1d(r2["co2_captured_kg_h"])[0])
    assert co2_2 > co2_1


def test_electrical_positive(model):
    """Electrical consumption must be positive."""
    r = model.predict({"PLR": 0.5, "operating_hours": 0})
    e = float(np.atleast_1d(r["electrical_kwh_ton"])[0])
    assert e > 0


def test_energy_penalty_bounded(model):
    """Energy penalty should be between 10% and 60%."""
    r = model.predict({"PLR": 0.5, "operating_hours": 20000})
    pen = float(np.atleast_1d(r["total_energy_penalty_pct"])[0])
    assert 10.0 <= pen <= 60.0, f"Energy penalty = {pen:.1f}%"


def test_array_input(model):
    """Model should handle array PLR inputs."""
    PLRs = np.linspace(0.3, 1.0, 10)
    r = model.predict({"PLR": PLRs, "operating_hours": 0})
    assert len(np.atleast_1d(r["reboiler_duty_gj_ton"])) == 10


def test_benchmark(model):
    PLRs = np.random.uniform(0.3, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"PLR": PLRs, "operating_hours": 5000})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
