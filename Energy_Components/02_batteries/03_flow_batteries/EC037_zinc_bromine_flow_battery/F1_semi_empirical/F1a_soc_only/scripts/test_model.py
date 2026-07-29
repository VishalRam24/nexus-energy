"""EC037 — Zinc-Bromine Flow Battery — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"soc": 0.5, "current": 50.0})
    for k in ["cell_voltage", "stack_voltage", "power", "efficiency"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC037"
    assert info["fidelity"] == "F1a"


def test_nernst_increases_with_soc(model):
    soc = np.linspace(0.1, 0.9, 50)
    r = model.predict({"soc": soc, "current": 0.0})
    assert np.all(np.diff(r["cell_voltage"]) > 0)


def test_voltage_drops_with_discharge(model):
    currents = np.array([0.0, 20.0, 50.0, 100.0, 150.0])
    r = model.predict({"soc": 0.5, "current": currents})
    assert np.all(np.diff(r["cell_voltage"]) < 0)


def test_voltage_rises_during_charge(model):
    r_ocv = model.predict({"soc": 0.5, "current":   0.0})
    r_chg = model.predict({"soc": 0.5, "current": -50.0})
    assert float(r_chg["stack_voltage"]) > float(r_ocv["stack_voltage"])


def test_efficiency_below_one(model):
    soc = np.linspace(0.1, 0.9, 20)
    r = model.predict({"soc": soc, "current": 50.0})
    assert np.all(r["efficiency"] < 1.0)


def test_efficiency_positive(model):
    soc = np.linspace(0.1, 0.9, 20)
    r = model.predict({"soc": soc, "current": 50.0})
    assert np.all(r["efficiency"] >= 0.0)


def test_stack_voltage_is_ncells_times_cell(model):
    r = model.predict({"soc": 0.5, "current": 30.0})
    N = model._model.N_cells
    np.testing.assert_allclose(r["stack_voltage"], N * r["cell_voltage"], rtol=1e-10)


def test_power_sign(model):
    r_dis = model.predict({"soc": 0.5, "current":  50.0})
    r_chg = model.predict({"soc": 0.5, "current": -50.0})
    assert float(r_dis["power"]) > 0
    assert float(r_chg["power"]) < 0


def test_nernst_at_50pct_equals_E0(model):
    """At SOC=0.5 and I=0, E_Nernst == E0 (log term is zero)."""
    r  = model.predict({"soc": 0.5, "current": 0.0})
    E0 = model._model.E0
    np.testing.assert_allclose(float(r["cell_voltage"]), E0, atol=1e-6)


def test_E0_around_185V(model):
    """ZBFB cell standard potential ~1.85 V."""
    E0 = model._model.E0
    assert 1.7 <= E0 <= 1.95


def test_soc_boundary_clamping(model):
    """SOC at 0 or 1 should be clamped (no inf/nan)."""
    for soc_val in [0.0, 1.0, 0.001, 0.999]:
        r = model.predict({"soc": soc_val, "current": 0.0})
        v = float(r["cell_voltage"])
        assert np.isfinite(v)


def test_soc_out_of_range_raises(model):
    for bad_soc in [-0.1, 1.1, -1.0, 2.0]:
        with pytest.raises(ValueError, match="SOC must be in"):
            model.predict({"soc": bad_soc, "current": 0.0})


def test_benchmark(model):
    soc     = np.random.uniform(0.1, 0.9, 1000)
    current = np.random.uniform(-100, 100, 1000)
    start = time.perf_counter()
    model.predict({"soc": soc, "current": current})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
