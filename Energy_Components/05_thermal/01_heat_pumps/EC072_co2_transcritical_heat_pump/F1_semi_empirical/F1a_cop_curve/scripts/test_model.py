"""EC072 — CO2 Transcritical Heat Pump — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_evap": 0.0, "T_water_in": 15.0, "T_water_out": 65.0})
    for k in ["cop", "heating_capacity_kw", "electrical_input_kw", "p_high_opt_bar"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC072"
    assert info["fidelity"] == "F1a"


def test_cop_greater_than_one(model):
    r = model.predict({"T_evap": 0.0, "T_water_in": 15.0, "T_water_out": 65.0})
    assert float(r["cop"]) > 1.0


def test_cop_at_design(model):
    """Design COP at E0/W15->65 should be ~3-4 for transcritical CO2."""
    r = model.predict({"T_evap": 0.0, "T_water_in": 15.0, "T_water_out": 65.0})
    cop = float(r["cop"])
    assert 2.5 < cop < 5.0, f"Design COP={cop:.2f} outside expected band"


def test_cop_increases_with_T_evap(model):
    """Higher evaporator temperature → smaller lift → higher COP."""
    Te = np.array([-15.0, -5.0, 0.0, 5.0, 15.0])
    r = model.predict({"T_evap": Te, "T_water_in": 15.0, "T_water_out": 65.0})
    assert np.all(np.diff(r["cop"]) > 0)


def test_cop_drops_at_high_water_outlet(model):
    """COP at very high T_water_out should be lower than at design."""
    r_des = model.predict({"T_evap": 0.0, "T_water_in": 15.0, "T_water_out": 65.0})
    r_hi  = model.predict({"T_evap": 0.0, "T_water_in": 15.0, "T_water_out": 85.0})
    assert float(r_hi["cop"]) < float(r_des["cop"])


def test_cop_benefits_from_low_T_water_in(model):
    """Lower water inlet (bigger glide) is favourable for transcritical CO2."""
    r_low = model.predict({"T_evap": 0.0, "T_water_in": 10.0, "T_water_out": 65.0})
    r_hi  = model.predict({"T_evap": 0.0, "T_water_in": 35.0, "T_water_out": 65.0})
    assert float(r_low["cop"]) > float(r_hi["cop"])


def test_energy_balance_with_aux(model):
    r = model.predict({"T_evap": 0.0, "T_water_in": 15.0, "T_water_out": 65.0})
    q   = float(r["heating_capacity_kw"])
    w   = float(r["electrical_input_kw"])
    cop = float(r["cop"])
    assert q / w <= cop + 0.05


def test_optimum_pressure_positive(model):
    r = model.predict({"T_evap": 0.0, "T_water_in": 15.0, "T_water_out": 65.0})
    assert float(r["p_high_opt_bar"]) > 60.0  # transcritical: well above CO2 critical (~74 bar typical)


def test_benchmark(model):
    Te = np.random.uniform(-15, 15, 1000)
    Tw = np.random.uniform(10, 30,  1000)
    To = np.random.uniform(50, 80,  1000)
    start = time.perf_counter()
    model.predict({"T_evap": Te, "T_water_in": Tw, "T_water_out": To})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
