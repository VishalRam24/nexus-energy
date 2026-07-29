"""EC193 — Methanation Reactor — F1a Sabatier Equilibrium — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"temperature": 300.0, "pressure": 10.0})
    for k in ["conversion", "ch4_rate_mols", "efficiency", "heat_released_kw"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC193"
    assert "fidelity" in info


def test_conversion_less_than_one(model):
    """Conversion must never exceed 1.0."""
    Ts = np.linspace(200, 500, 20)
    Ps = np.linspace(1, 30, 20)
    for T, P in zip(Ts, Ps):
        r = model.predict({"temperature": float(T), "pressure": float(P)})
        assert float(r["conversion"]) <= 1.0 + 1e-9


def test_conversion_decreases_at_high_T(model):
    """Sabatier is exothermic: conversion decreases at high temperatures (>300°C)."""
    Ts = np.array([300.0, 350.0, 400.0, 450.0, 500.0])
    Xs = np.array([float(model.predict({"temperature": float(T), "pressure": 10.0})["conversion"])
                   for T in Ts])
    assert np.all(np.diff(Xs) < 0), f"Conversion not decreasing above 300°C: {Xs}"


def test_conversion_increases_with_pressure(model):
    """Higher pressure should increase (or maintain) conversion (Le Chatelier's principle)."""
    Ps = np.array([1.0, 5.0, 10.0, 20.0, 30.0])
    Xs = np.array([float(model.predict({"temperature": 300.0, "pressure": float(P)})["conversion"])
                   for P in Ps])
    # Monotonically non-decreasing (may saturate at 1.0 at very high P)
    assert np.all(np.diff(Xs) >= 0), f"Conversion not non-decreasing with pressure: {Xs}"
    # Must strictly increase from low to moderate pressures (before saturation)
    assert Xs[2] > Xs[0], f"X at P=10 not > X at P=1: {Xs[2]:.4f} vs {Xs[0]:.4f}"


def test_design_point_conversion(model):
    """At T=300°C, P=10 bar, X should be ~0.98 (maximum by design)."""
    r = model.predict({"temperature": 300.0, "pressure": 10.0, "h2_co2_ratio": 4.0})
    X = float(r["conversion"])
    assert abs(X - 0.98) < 0.02, f"X at design = {X:.4f}"


def test_stoichiometry(model):
    """CH4 rate = X * n_CO2_in (from parameters.json, n_CO2 = 1 mol/s)."""
    r = model.predict({"temperature": 300.0, "pressure": 10.0})
    X = float(r["conversion"])
    ch4 = float(r["ch4_rate_mols"])
    assert abs(ch4 - X) < 1e-9, f"Stoichiometry error: ch4={ch4:.4f}, X={X:.4f}"


def test_h2_limitation(model):
    """Sub-stoichiometric H2 (ratio < 4) should limit conversion."""
    r_stoich = model.predict({"temperature": 300.0, "pressure": 10.0, "h2_co2_ratio": 4.0})
    r_lean   = model.predict({"temperature": 300.0, "pressure": 10.0, "h2_co2_ratio": 3.5})
    assert float(r_lean["conversion"]) <= float(r_stoich["conversion"])


def test_heat_released_positive(model):
    """Sabatier is exothermic: heat released must be positive."""
    r = model.predict({"temperature": 300.0, "pressure": 10.0})
    assert float(r["heat_released_kw"]) > 0


def test_efficiency_range(model):
    """Energy efficiency should be < 1.0 (LHV_CH4 < 4*LHV_H2 for X<1)."""
    Ts = np.linspace(200, 500, 10)
    for T in Ts:
        r = model.predict({"temperature": float(T), "pressure": 10.0})
        eta = float(r["efficiency"])
        assert 0.0 <= eta <= 1.0 + 1e-9, f"T={T}: efficiency={eta:.4f}"


def test_array_input(model):
    """Model should handle array inputs."""
    Ts = np.linspace(200, 500, 15)
    r = model.predict({"temperature": Ts, "pressure": 10.0})
    assert len(r["conversion"]) == 15


def test_benchmark(model):
    Ts = np.random.uniform(200, 500, 1000)
    Ps = np.random.uniform(1, 30, 1000)
    start = time.perf_counter()
    model.predict({"temperature": Ts, "pressure": Ps})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
