"""EC195 — Ammonia Synthesis (Haber-Bosch) — F1a Conversion — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"temperature": 450.0, "pressure": 200.0})
    for k in ["conversion_per_pass", "nh3_rate_kgs", "energy_gj_per_ton", "efficiency"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC195"
    assert "fidelity" in info


def test_conversion_less_than_eq_limit(model):
    """Per-pass conversion must be strictly below equilibrium limit at all conditions."""
    Ts = np.linspace(350, 550, 10)
    for T in Ts:
        r    = model.predict({"temperature": float(T), "pressure": 200.0})
        X    = float(r["conversion_per_pass"])
        X_eq = float(model._model.equilibrium_conversion(float(T), 200.0))
        assert X <= X_eq + 1e-6, (
            f"T={T}: X={X:.4f} > X_eq={X_eq:.4f} — violates thermodynamic limit"
        )
        # X must be physically meaningful (positive, less than 1)
        assert 0.0 <= X <= 1.0, f"T={T}: X={X:.4f} out of bounds"


def test_conversion_increases_with_pressure(model):
    """Higher pressure increases per-pass conversion (Le Chatelier)."""
    Ps = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    Xs = np.array([float(model.predict({"temperature": 450.0, "pressure": float(P)})["conversion_per_pass"])
                   for P in Ps])
    assert np.all(np.diff(Xs) >= 0), f"Conversion not non-decreasing with P: {Xs}"
    assert Xs[-1] > Xs[0], "Conversion must increase from P=100 to P=300"


def test_conversion_decreases_at_very_high_T(model):
    """Haber-Bosch is exothermic: equilibrium conversion decreases at high T."""
    Ts = np.array([400.0, 450.0, 500.0, 550.0])
    Xs_eq = np.array([float(model._model.equilibrium_conversion(float(T), 200.0)) for T in Ts])
    assert np.all(np.diff(Xs_eq) < 0), f"Equilibrium conversion not decreasing: {Xs_eq}"


def test_design_point_conversion(model):
    """At T=450°C, P=200 bar, X should be ~0.15 (per design parameters)."""
    r = model.predict({"temperature": 450.0, "pressure": 200.0})
    X = float(r["conversion_per_pass"])
    assert abs(X - 0.15) < 0.05, f"X at design = {X:.4f}, expected ~0.15"


def test_energy_around_28_gjton(model):
    """Specific energy at design point should be ~28 GJ/tNH3."""
    r = model.predict({"temperature": 450.0, "pressure": 200.0})
    E = float(r["energy_gj_per_ton"])
    assert 20.0 < E < 40.0, f"Specific energy = {E:.2f} GJ/tNH3"


def test_nh3_rate_positive(model):
    """NH3 production rate must be positive."""
    r = model.predict({"temperature": 450.0, "pressure": 200.0})
    assert float(r["nh3_rate_kgs"]) > 0


def test_array_input(model):
    """Model should handle array inputs."""
    Ts = np.linspace(350, 550, 20)
    r = model.predict({"temperature": Ts, "pressure": 200.0})
    assert len(r["conversion_per_pass"]) == 20


def test_efficiency_positive(model):
    """Energy efficiency must be positive."""
    r = model.predict({"temperature": 450.0, "pressure": 200.0})
    eta = float(r["efficiency"])
    assert eta > 0.0


def test_benchmark(model):
    Ts = np.random.uniform(350, 550, 1000)
    Ps = np.random.uniform(100, 300, 1000)
    start = time.perf_counter()
    model.predict({"temperature": Ts, "pressure": Ps})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
