"""EC091 — Vapor Compression Chiller — F2a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_evap_degC": 5.0, "T_cond_degC": 35.0})
    for k in ["cop_cooling", "cooling_capacity_kw", "compressor_kw", "heat_rejection_kw"]:
        assert k in r


def test_get_info(model):
    assert model.get_info()["ec_id"] == "EC091"


def test_cop_positive(model):
    r = model.predict({"T_evap_degC": 5.0, "T_cond_degC": 35.0})
    assert r["cop_cooling"] > 0


def test_cop_typical_range(model):
    """COP_cooling at typical conditions should be 3-8."""
    r = model.predict({"T_evap_degC": 5.0, "T_cond_degC": 35.0})
    assert 2.0 < r["cop_cooling"] < 10.0, f"COP={r['cop_cooling']:.2f}"


def test_cop_increases_with_tevap(model):
    """Higher evaporator temp -> higher COP."""
    cops = []
    for Te in [0, 3, 5, 7, 10]:
        r = model.predict({"T_evap_degC": Te, "T_cond_degC": 35.0})
        cops.append(r["cop_cooling"])
    for i in range(len(cops) - 1):
        assert cops[i] < cops[i+1]


def test_cop_decreases_with_tcond(model):
    """Higher condenser temp -> lower COP."""
    cops = []
    for Tc in [30, 35, 40, 45, 50]:
        r = model.predict({"T_evap_degC": 5.0, "T_cond_degC": Tc})
        cops.append(r["cop_cooling"])
    for i in range(len(cops) - 1):
        assert cops[i] > cops[i+1]


def test_heat_rejection_balance(model):
    """Q_rejection = Q_cooling + W_comp (approximately)."""
    r = model.predict({"T_evap_degC": 5.0, "T_cond_degC": 35.0})
    expected = r["cooling_capacity_kw"] + r["compressor_kw"]
    assert abs(r["heat_rejection_kw"] - expected) / expected < 0.1


def test_benchmark(model):
    start = time.perf_counter()
    for _ in range(100):
        model.predict({"T_evap_degC": 5.0, "T_cond_degC": 35.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 100 cycles in {elapsed*1000:.1f} ms")
    assert elapsed < 10.0
