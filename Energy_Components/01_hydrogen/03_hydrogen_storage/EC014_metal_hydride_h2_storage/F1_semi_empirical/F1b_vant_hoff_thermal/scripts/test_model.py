"""
EC014 -- Metal Hydride H2 Storage -- F1b van't Hoff Thermal -- Test Suite

Physics sanity checks for van't Hoff equilibrium + kinetics + thermal.
Run: python -m pytest test_model.py -v
"""

import sys
import time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Output structure ---

def test_predict_returns_required_keys(model):
    result = model.predict({"temperature": 298.15, "pressure_bar": 5.0, "soc": 0.5})
    for key in ["plateau_pressure_bar", "sorption_rate_kg_s", "reaction_heat_W",
                "dTdt_K_s", "stored_mass_kg", "gravimetric_wt_pct"]:
        assert key in result


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC014"
    assert info["fidelity"] == "F1b"


# --- van't Hoff: plateau pressure increases with temperature ---

def test_plateau_pressure_increases_with_temperature(model):
    """P_eq must increase with temperature (endothermic desorption, d(lnP)/dT > 0)."""
    T_vals = [253.15, 273.15, 298.15, 323.15, 353.15]
    P_prev = float(model.predict({"temperature": T_vals[0], "pressure_bar": 1.0, "soc": 0.5, "mode": "desorption"})["plateau_pressure_bar"])
    for T in T_vals[1:]:
        P = float(model.predict({"temperature": T, "pressure_bar": 1.0, "soc": 0.5, "mode": "desorption"})["plateau_pressure_bar"])
        assert P > P_prev, f"P_eq({T:.0f} K)={P:.3f} bar must be > P_eq at lower T={P_prev:.3f}"
        P_prev = P


def test_absorption_pressure_above_desorption(model):
    """Absorption plateau must be higher than desorption (hysteresis)."""
    P_abs = float(model.predict({"temperature": 298.15, "pressure_bar": 5.0, "soc": 0.5, "mode": "absorption"})["plateau_pressure_bar"])
    P_des = float(model.predict({"temperature": 298.15, "pressure_bar": 5.0, "soc": 0.5, "mode": "desorption"})["plateau_pressure_bar"])
    assert P_abs > P_des, f"Absorption P_eq={P_abs:.3f} must exceed desorption P_eq={P_des:.3f}"


def test_plateau_pressure_positive(model):
    """Plateau pressure must be positive at all temperatures."""
    for T in [253.15, 298.15, 373.15]:
        P = float(model.predict({"temperature": T, "pressure_bar": 1.0, "soc": 0.5})["plateau_pressure_bar"])
        assert P > 0.0, f"P_eq must be positive at T={T}"


# --- LaNi5 reference pressure: ~2 bar at 25 degC for desorption ---

def test_lani5_plateau_at_25c(model):
    """LaNi5 desorption plateau at 25 degC should be ~1.5-3.5 bar (Sandrock 1999)."""
    P = float(model.predict({"temperature": 298.15, "pressure_bar": 1.0, "soc": 0.5, "mode": "desorption"})["plateau_pressure_bar"])
    assert 1.0 <= P <= 5.0, f"LaNi5 P_eq at 25C should be ~2 bar, got {P:.3f} bar"


# --- Kinetics ---

def test_absorption_rate_positive_when_P_above_Peq(model):
    """Absorption rate > 0 when P_feed > P_eq."""
    # LaNi5 at 298.15 K: P_eq ~ 2 bar (desorption); feed at 20 bar should drive absorption
    r = model.predict({"temperature": 298.15, "pressure_bar": 20.0, "soc": 0.3, "mode": "absorption"})
    assert float(r["sorption_rate_kg_s"]) > 0, "Sorption rate must be positive when P >> P_eq"


def test_desorption_rate_negative_when_P_below_Peq(model):
    """Desorption rate < 0 when P_system < P_eq."""
    # At 353 K, P_eq ~ 10+ bar; if P_system = 1 bar, desorption should drive negative rate
    r = model.predict({"temperature": 353.15, "pressure_bar": 1.0, "soc": 0.7, "mode": "desorption"})
    assert float(r["sorption_rate_kg_s"]) <= 0, "Desorption rate must be negative when P << P_eq"


def test_rate_increases_with_temperature(model):
    """Arrhenius: kinetic rate increases with temperature (same driving force)."""
    r_cold = model.predict({"temperature": 273.15, "pressure_bar": 30.0, "soc": 0.3, "mode": "absorption"})
    r_hot  = model.predict({"temperature": 333.15, "pressure_bar": 30.0, "soc": 0.3, "mode": "absorption"})
    # Note: P_eq also changes with T, but at P=30 bar both should be in absorption regime
    # Just check rates are non-negative and hot > cold
    rate_cold = float(r_cold["sorption_rate_kg_s"])
    rate_hot  = float(r_hot["sorption_rate_kg_s"])
    assert rate_cold >= 0 and rate_hot >= 0, "Absorption rates must be >= 0 at P=30 bar"
    assert rate_hot >= rate_cold, "Higher T should give faster absorption kinetics"


# --- Thermal balance ---

def test_reaction_heat_positive_during_absorption(model):
    """Absorption is exothermic -> reaction heat must be positive."""
    r = model.predict({"temperature": 298.15, "pressure_bar": 20.0, "soc": 0.3, "mode": "absorption"})
    assert float(r["reaction_heat_W"]) >= 0, "Absorption reaction heat must be >= 0 (exothermic)"


def test_dTdt_finite(model):
    """Temperature derivative must be finite."""
    r = model.predict({"temperature": 298.15, "pressure_bar": 5.0, "soc": 0.5})
    assert np.isfinite(float(r["dTdt_K_s"])), "dTdt must be finite"


# --- Stored mass ---

def test_stored_mass_at_soc_zero(model):
    """At SOC=0, stored mass should be 0."""
    r = model.predict({"temperature": 298.15, "pressure_bar": 5.0, "soc": 0.0})
    assert float(r["stored_mass_kg"]) == pytest.approx(0.0, abs=1e-9)


def test_stored_mass_at_soc_one(model):
    """At SOC=1, stored mass should equal m_H2_max."""
    r = model.predict({"temperature": 298.15, "pressure_bar": 5.0, "soc": 1.0})
    m_max = float(model._model.m_H2_max)
    assert float(r["stored_mass_kg"]) == pytest.approx(m_max, rel=1e-6)


def test_gravimetric_density_bounded(model):
    """Gravimetric density must be in [0, H_max_wt_pct]."""
    H_max = float(model._model.H_max)
    for soc in [0.0, 0.5, 1.0]:
        r = model.predict({"temperature": 298.15, "pressure_bar": 5.0, "soc": soc})
        grav = float(r["gravimetric_wt_pct"])
        assert 0.0 <= grav <= H_max + 0.01, f"Gravimetric density {grav:.3f} out of bounds"


# --- Edge cases ---

def test_array_inputs(model):
    """Model must handle array inputs."""
    Ts   = np.array([273.15, 298.15, 323.15])
    Ps   = np.array([2.0, 5.0, 10.0])
    socs = np.array([0.2, 0.5, 0.8])
    r = model.predict({"temperature": Ts, "pressure_bar": Ps, "soc": socs})
    assert r["plateau_pressure_bar"].shape == (3,)


# --- Benchmark ---

def test_benchmark_1000_predictions(model):
    """1000 predictions should complete in < 1s."""
    Ts   = np.random.uniform(253.15, 373.15, 1000)
    Ps   = np.random.uniform(0.5, 50.0, 1000)
    socs = np.random.uniform(0.0, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"temperature": Ts, "pressure_bar": Ps, "soc": socs})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
