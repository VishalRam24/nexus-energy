"""EC069 — GSHP — F2a Vapor Cycle SS — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_cond_degC": 45.0})
    for k in ["cop", "heating_capacity_kw", "compressor_power_kw", "T_evap_degC",
              "T_source_degC", "state_points"]:
        assert k in r


def test_get_info(model):
    assert model.get_info()["ec_id"] == "EC069"


def test_cop_greater_than_one(model):
    for Tc in [30, 35, 45, 55]:
        r = model.predict({"T_cond_degC": Tc})
        assert r["cop"] > 1.0, f"COP={r['cop']} at T_cond={Tc}"


def test_cop_higher_than_ashp(model):
    """GSHP should have COP > 3 at T_cond=35 due to stable ground source."""
    r = model.predict({"T_cond_degC": 35.0})
    assert r["cop"] > 3.0, f"GSHP COP at 35C should be > 3, got {r['cop']:.2f}"


def test_cop_decreases_with_tcond(model):
    """COP should decrease as condensing temp increases."""
    cops = []
    for Tc in [30, 35, 40, 45, 50]:
        r = model.predict({"T_cond_degC": Tc})
        cops.append(r["cop"])
    for i in range(len(cops) - 1):
        assert cops[i] > cops[i + 1]


def test_ground_source_temp_reasonable(model):
    """Source temp should be near T_ground (10C) with some depression."""
    r = model.predict({"T_cond_degC": 45.0})
    assert 0.0 < r["T_source_degC"] < 12.0


def test_evap_temp_below_source(model):
    """Evaporator temp must be below source temp."""
    r = model.predict({"T_cond_degC": 45.0})
    assert r["T_evap_degC"] < r["T_source_degC"]


def test_energy_balance(model):
    """Q_heating ~ Q_evap + W_comp."""
    r = model.predict({"T_cond_degC": 45.0})
    balance = abs(r["heating_capacity_kw"] - (r["Q_evap_kw"] + r["compressor_power_kw"]))
    # Allow some tolerance for motor efficiency
    assert balance < 2.0, f"Energy balance error: {balance:.2f} kW"


def test_benchmark(model):
    start = time.perf_counter()
    for _ in range(50):
        model.predict({"T_cond_degC": 45.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 50 cycles in {elapsed*1000:.1f} ms")
    assert elapsed < 10.0
