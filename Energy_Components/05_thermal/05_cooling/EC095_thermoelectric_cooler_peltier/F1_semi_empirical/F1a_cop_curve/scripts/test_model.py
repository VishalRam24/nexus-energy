"""EC095 — Thermoelectric Cooler (Peltier) — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"current": 3.0, "T_cold": 5.0, "T_hot": 35.0})
    for k in ["cooling_power_w", "electrical_input_w", "heat_rejection_w", "cop", "i_optimum_a"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC095"
    assert info["fidelity"] == "F1a"


def test_cooling_positive_at_design(model):
    r = model.predict({"current": 3.0, "T_cold": 5.0, "T_hot": 35.0})
    assert float(r["cooling_power_w"]) > 0


def test_cop_positive(model):
    r = model.predict({"current": 3.0, "T_cold": 5.0, "T_hot": 35.0})
    assert float(r["cop"]) > 0


def test_cop_below_two(model):
    """TECs typically deliver COP_c well below 2 even at small ΔT."""
    r = model.predict({"current": 2.0, "T_cold": 20.0, "T_hot": 25.0})
    assert float(r["cop"]) < 2.0


def test_cop_decreases_with_dT(model):
    """At fixed current, larger ΔT (T_h - T_c) lowers COP."""
    Th = np.array([20.0, 30.0, 40.0, 50.0])
    r = model.predict({"current": 3.0, "T_cold": 10.0, "T_hot": Th})
    assert np.all(np.diff(r["cop"]) < 0)


def test_cooling_decreases_with_dT(model):
    """Heat conduction back from hot side erodes Q_c as ΔT grows."""
    Th = np.array([20.0, 30.0, 40.0, 50.0])
    r = model.predict({"current": 3.0, "T_cold": 10.0, "T_hot": Th})
    assert np.all(np.diff(r["cooling_power_w"]) < 0)


def test_energy_balance(model):
    """Q_h = Q_c + W_in (first law)."""
    r = model.predict({"current": 3.0, "T_cold": 5.0, "T_hot": 35.0})
    qc = float(r["cooling_power_w"])
    w  = float(r["electrical_input_w"])
    qh = float(r["heat_rejection_w"])
    assert abs(qh - (qc + w)) < 1e-6


def test_zero_current_no_cooling(model):
    """At I=0, no Peltier effect — Q_c is clipped to 0 (back-conduction otherwise)."""
    r = model.predict({"current": 0.0, "T_cold": 5.0, "T_hot": 35.0})
    assert float(r["cooling_power_w"]) == 0.0
    assert float(r["electrical_input_w"]) == 0.0


def test_cop_at_zero_dT(model):
    """At ΔT=0 the Peltier-only mode reaches its highest COP (limited by Joule)."""
    r0  = model.predict({"current": 2.0, "T_cold": 25.0, "T_hot": 25.0})
    r10 = model.predict({"current": 2.0, "T_cold": 15.0, "T_hot": 25.0})
    assert float(r0["cop"]) > float(r10["cop"])


def test_optimum_current_in_range(model):
    r = model.predict({"current": 3.0, "T_cold": 5.0, "T_hot": 35.0})
    Iopt = float(r["i_optimum_a"])
    assert 0.0 < Iopt <= model._model.I_max + 1e-9


def test_benchmark(model):
    n = 1000
    I  = np.random.uniform(0.5, 6.0, n)
    Tc = np.random.uniform(-10, 25, n)
    Th = np.random.uniform(20, 60, n)
    start = time.perf_counter()
    model.predict({"current": I, "T_cold": Tc, "T_hot": Th})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
