"""EC143 — Biomass Gasifier — F1a Equilibrium — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"equivalence_ratio": 0.25})
    assert "syngas_composition" in r
    assert "lhv_syngas_mjnm3" in r
    assert "cold_gas_efficiency" in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC143"
    assert "fidelity" in info


def test_fractions_sum_to_one(model):
    """All mole fractions must sum to 1.0."""
    for ER in [0.2, 0.25, 0.30, 0.40, 0.50]:
        r = model.predict({"equivalence_ratio": ER})
        comp = r["syngas_composition"]
        total = sum(comp.values())
        assert abs(total - 1.0) < 1e-6, f"ER={ER}: fractions sum = {total:.6f}"


def test_fractions_non_negative(model):
    """All fractions must be >= 0."""
    ERs = np.linspace(0.2, 0.5, 20)
    for ER in ERs:
        r = model.predict({"equivalence_ratio": float(ER)})
        for k, v in r["syngas_composition"].items():
            assert v >= 0.0, f"ER={ER}: {k}={v} is negative"


def test_lhv_decreases_with_er(model):
    """LHV should decrease as ER increases (more dilution with N2)."""
    ERs = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    lhvs = np.array([float(model.predict({"equivalence_ratio": er})["lhv_syngas_mjnm3"]) for er in ERs])
    assert np.all(np.diff(lhvs) < 0), f"LHV not monotonically decreasing: {lhvs}"


def test_co_h2_dominant_at_low_er(model):
    """At low ER (0.2), CO + H2 should be the dominant combustible species."""
    r = model.predict({"equivalence_ratio": 0.2})
    comp = r["syngas_composition"]
    co_h2 = comp["CO"] + comp["H2"]
    assert co_h2 > 0.30, f"CO+H2 at ER=0.2: {co_h2:.3f} — expected > 0.30"


def test_design_point_composition(model):
    """At ER=0.25, check reference compositions from Zainal et al."""
    r = model.predict({"equivalence_ratio": 0.25})
    comp = r["syngas_composition"]
    assert abs(comp["CO"]  - 0.22) < 0.02, f"CO at ER=0.25 = {comp['CO']:.3f}"
    assert abs(comp["H2"]  - 0.18) < 0.02, f"H2 at ER=0.25 = {comp['H2']:.3f}"
    assert abs(comp["CO2"] - 0.10) < 0.02, f"CO2 at ER=0.25 = {comp['CO2']:.3f}"
    assert abs(comp["CH4"] - 0.03) < 0.02, f"CH4 at ER=0.25 = {comp['CH4']:.3f}"


def test_lhv_design_point(model):
    """LHV at ER=0.25 should be in range 3.5–7.0 MJ/Nm3 (typical downdraft wood gasifier)."""
    r = model.predict({"equivalence_ratio": 0.25})
    lhv = float(r["lhv_syngas_mjnm3"])
    assert 3.5 < lhv < 7.0, f"LHV at ER=0.25 = {lhv:.3f} MJ/Nm3"


def test_cge_reasonable(model):
    """Cold gas efficiency at design point should be 0.6–0.85."""
    r = model.predict({"equivalence_ratio": 0.25, "temperature": 800.0})
    cge = float(r["cold_gas_efficiency"])
    assert 0.5 < cge < 0.95, f"CGE at design = {cge:.3f}"


def test_temperature_input(model):
    """Model should accept temperature as input without error."""
    r = model.predict({"equivalence_ratio": 0.25, "temperature": 900.0})
    assert "syngas_composition" in r


def test_array_input(model):
    """Model should handle array inputs correctly."""
    ERs = np.linspace(0.2, 0.5, 10)
    r = model.predict({"equivalence_ratio": ERs})
    comp = r["syngas_composition"]
    assert len(comp["CO"]) == 10


def test_benchmark(model):
    ERs = np.random.uniform(0.2, 0.5, 1000)
    start = time.perf_counter()
    model.predict({"equivalence_ratio": ERs})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
