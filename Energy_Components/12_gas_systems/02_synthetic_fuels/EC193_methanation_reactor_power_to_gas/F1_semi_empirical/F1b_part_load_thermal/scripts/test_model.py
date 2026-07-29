"""EC193 — Methanation Reactor — F1b Part-Load Thermal — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": 4.0, "PLR": 1.0})
    for k in ["ch4_production_mol_s", "conversion", "heat_recovery_kw",
              "overall_efficiency", "selectivity"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC193"
    assert info["fidelity"] == "F1b"


def test_conversion_decreases_with_part_load(model):
    """Conversion must decrease as PLR drops."""
    PLRs = [0.3, 0.5, 0.7, 1.0]
    Xs = []
    for plr in PLRs:
        r = model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": 4.0, "PLR": plr})
        Xs.append(float(np.atleast_1d(r["conversion"])[0]))
    assert all(Xs[i] <= Xs[i + 1] + 1e-6 for i in range(len(Xs) - 1)), \
        f"Conversion not non-decreasing with PLR: {Xs}"


def test_conversion_at_full_load(model):
    """At PLR=1.0, conversion should match F1a design (~0.98 at optimal T)."""
    r = model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": 4.0, "PLR": 1.0})
    X = float(np.atleast_1d(r["conversion"])[0])
    assert 0.8 < X <= 1.0, f"Full-load conversion = {X:.4f}, expected ~0.98"


def test_ch4_production_proportional_to_flow(model):
    """CH4 production should scale with CO2 feed flow."""
    r1 = model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": 4.0, "PLR": 1.0})
    r2 = model.predict({"co2_flow_mol_s": 2.0, "h2_co2_ratio": 4.0, "PLR": 1.0})
    ch4_1 = float(np.atleast_1d(r1["ch4_production_mol_s"])[0])
    ch4_2 = float(np.atleast_1d(r2["ch4_production_mol_s"])[0])
    assert abs(ch4_2 / ch4_1 - 2.0) < 0.01, \
        f"CH4 not scaling linearly: {ch4_1:.4f} vs {ch4_2:.4f}"


def test_heat_recovery_positive(model):
    """Heat recovery must be positive (exothermic reaction)."""
    r = model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": 4.0, "PLR": 0.5})
    Q = float(np.atleast_1d(r["heat_recovery_kw"])[0])
    assert Q > 0, f"Heat recovery = {Q:.4f} kW, expected positive"


def test_heat_recovery_increases_with_plr(model):
    """Heat recovery should increase with PLR."""
    Qs = []
    for plr in [0.3, 0.5, 0.7, 1.0]:
        r = model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": 4.0, "PLR": plr})
        Qs.append(float(np.atleast_1d(r["heat_recovery_kw"])[0]))
    assert all(Qs[i] <= Qs[i + 1] + 1e-6 for i in range(len(Qs) - 1)), \
        f"Heat recovery not non-decreasing: {Qs}"


def test_selectivity_range(model):
    """Selectivity must be in [0, 1]."""
    for plr in [0.3, 0.5, 1.0]:
        r = model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": 4.0, "PLR": plr})
        S = float(np.atleast_1d(r["selectivity"])[0])
        assert 0.0 <= S <= 1.0, f"Selectivity = {S:.4f} at PLR={plr}"


def test_efficiency_bounded(model):
    """Overall efficiency must be in (0, 1)."""
    for plr in [0.3, 0.5, 0.7, 1.0]:
        r = model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": 4.0, "PLR": plr})
        eta = float(np.atleast_1d(r["overall_efficiency"])[0])
        assert 0.0 < eta < 1.0, f"Efficiency = {eta:.4f} at PLR={plr}"


def test_substoichiometric_h2_limits_conversion(model):
    """Low H2/CO2 ratio should limit conversion."""
    r_low = model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": 3.5, "PLR": 1.0})
    r_stoich = model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": 4.0, "PLR": 1.0})
    X_low = float(np.atleast_1d(r_low["conversion"])[0])
    X_stoich = float(np.atleast_1d(r_stoich["conversion"])[0])
    assert X_low <= X_stoich + 1e-6


def test_dh_heat_magnitude(model):
    """At full load, Q should be ~165 * X * n_CO2 * f_recovery kW."""
    r = model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": 4.0, "PLR": 1.0})
    X = float(np.atleast_1d(r["conversion"])[0])
    Q = float(np.atleast_1d(r["heat_recovery_kw"])[0])
    Q_expected = X * 165.0 * 1.0 * 0.85  # DH * n * f_recovery
    assert abs(Q - Q_expected) / Q_expected < 0.05, \
        f"Q={Q:.2f} vs expected={Q_expected:.2f}"


def test_array_input(model):
    """Model should handle array PLR inputs."""
    PLRs = np.linspace(0.3, 1.0, 10)
    r = model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": 4.0, "PLR": PLRs})
    assert len(np.atleast_1d(r["conversion"])) == 10


def test_benchmark(model):
    PLRs = np.random.uniform(0.3, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": 4.0, "PLR": PLRs})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
