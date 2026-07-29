"""EC093 — Adsorption Chiller — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0})
    for k in ["cop", "cooling_kw", "driving_heat_kw", "heat_rejection_kw", "electrical_kw"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC093"
    assert info["fidelity"] == "F1a"


def test_cop_positive(model):
    r = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0})
    assert float(r["cop"]) > 0


def test_cop_in_expected_range(model):
    """Single-stage silica-gel chiller COP_c is typically 0.4–0.7."""
    r = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0})
    cop = float(r["cop"])
    assert 0.3 < cop < 0.8, f"COP_c = {cop:.2f} outside expected band"


def test_cop_lower_than_vapour_compression(model):
    """Adsorption COP_c must be < ~1 (much lower than VC chillers)."""
    r = model.predict({"T_hot": 95.0, "T_cool": 22.0, "T_chilled": 18.0})
    assert float(r["cop"]) < 1.0


def test_cop_increases_with_T_hot(model):
    Th = np.array([60.0, 70.0, 80.0, 90.0])
    r = model.predict({"T_hot": Th, "T_cool": 30.0, "T_chilled": 14.0})
    assert np.all(np.diff(r["cop"]) > 0)


def test_cop_decreases_with_T_cool(model):
    Tc = np.array([22.0, 28.0, 34.0, 40.0])
    r = model.predict({"T_hot": 85.0, "T_cool": Tc, "T_chilled": 14.0})
    assert np.all(np.diff(r["cop"]) < 0)


def test_cop_increases_with_T_chilled(model):
    """Higher chilled-water temperature → less lift → higher COP."""
    Tx = np.array([6.0, 10.0, 14.0, 18.0])
    r = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": Tx})
    assert np.all(np.diff(r["cop"]) > 0)


def test_energy_balance(model):
    """Q_reject = Q_cool + Q_drive."""
    r = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0})
    qc = float(r["cooling_kw"])
    qd = float(r["driving_heat_kw"])
    qr = float(r["heat_rejection_kw"])
    assert abs(qr - (qc + qd)) < 0.05


def test_capacity_scales_with_plr(model):
    r1 = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0, "part_load_ratio": 1.0})
    r2 = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0, "part_load_ratio": 0.4})
    assert abs(float(r2["cooling_kw"]) - 0.4 * float(r1["cooling_kw"])) < 1e-6


def test_benchmark(model):
    n = 1000
    Th = np.random.uniform(60, 95, n)
    Tc = np.random.uniform(22, 38, n)
    Tx = np.random.uniform(6, 18, n)
    start = time.perf_counter()
    model.predict({"T_hot": Th, "T_cool": Tc, "T_chilled": Tx})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
