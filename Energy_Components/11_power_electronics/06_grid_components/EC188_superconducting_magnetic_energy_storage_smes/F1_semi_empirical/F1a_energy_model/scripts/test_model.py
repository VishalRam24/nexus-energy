"""EC188 — SMES — F1a Energy Model — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"SOC": 0.8, "P_request_MW": 5.0, "mode": "discharge"})
    for k in ["P_delivered_MW", "P_grid_MW", "P_cryo_MW", "P_total_parasitic_MW",
              "SOC_new", "E_stored_MJ", "eta_instantaneous", "dE_MJ"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC188"
    assert info["fidelity"] == "F1a"


def test_energy_formula(model):
    """E = 0.5 * L * I^2."""
    L = model._model.L
    I = 1000.0
    r = model.energy_from_current({"I_A": I})
    expected = 0.5 * L * I**2 / 1e6  # MJ
    assert abs(float(r["E_MJ"]) - expected) < 1e-9


def test_SOC_at_full_charge(model):
    """SOC=1 at I=I_max."""
    I_max = model._model.I_max
    r = model.energy_from_current({"I_A": I_max})
    assert abs(float(r["SOC"]) - 1.0) < 1e-6


def test_SOC_at_zero_current(model):
    """SOC=0 at I=0."""
    r = model.energy_from_current({"I_A": 0.0})
    assert abs(float(r["SOC"])) < 1e-12


def test_cryo_always_drawn(model):
    """Cryogenic power drawn at all SOC levels (even standby)."""
    for soc in [0.0, 0.5, 1.0]:
        r = model.predict({"SOC": soc, "P_request_MW": 0.0, "mode": "discharge"})
        assert float(r["P_cryo_MW"]) == model._model.P_cryo


def test_discharge_reduces_SOC(model):
    """Discharging reduces SOC."""
    r = model.predict({"SOC": 0.8, "P_request_MW": 5.0, "mode": "discharge", "dt_s": 60.0})
    assert float(r["SOC_new"]) < 0.8, \
        f"SOC should decrease after discharge: {float(r['SOC_new']):.4f}"


def test_charge_increases_SOC(model):
    """Charging increases SOC."""
    r = model.predict({"SOC": 0.3, "P_request_MW": 5.0, "mode": "charge", "dt_s": 60.0})
    assert float(r["SOC_new"]) > 0.3, \
        f"SOC should increase after charging: {float(r['SOC_new']):.4f}"


def test_SOC_bounded(model):
    """SOC must stay in [0, 1]."""
    # Over-discharge from empty
    r1 = model.predict({"SOC": 0.0, "P_request_MW": 10.0, "mode": "discharge", "dt_s": 1000.0})
    assert 0.0 <= float(r1["SOC_new"]) <= 1.0
    # Over-charge from full
    r2 = model.predict({"SOC": 1.0, "P_request_MW": 10.0, "mode": "charge", "dt_s": 1000.0})
    assert 0.0 <= float(r2["SOC_new"]) <= 1.0


def test_eta_below_one(model):
    """Instantaneous efficiency < 1 (converter + cryo losses)."""
    r = model.predict({"SOC": 0.8, "P_request_MW": 8.0, "mode": "discharge"})
    assert float(r["eta_instantaneous"]) < 1.0


def test_eta_positive(model):
    """Efficiency > 0 for high-power discharge."""
    r = model.predict({"SOC": 0.8, "P_request_MW": 8.0, "mode": "discharge"})
    assert float(r["eta_instantaneous"]) > 0.0


def test_E_stored_consistent_with_SOC(model):
    """E_stored = SOC_new * E_max."""
    r = model.predict({"SOC": 0.5, "P_request_MW": 3.0, "mode": "discharge", "dt_s": 1.0})
    E_max = model._model.E_max_MJ
    expected_E = float(r["SOC_new"]) * E_max
    assert abs(float(r["E_stored_MJ"]) - expected_E) < 1e-9


def test_vectorized_SOC(model):
    SOC_arr = np.linspace(0.1, 1.0, 30)
    r = model.predict({"SOC": SOC_arr, "P_request_MW": 5.0, "mode": "discharge"})
    assert r["SOC_new"].shape == (30,)
    assert np.all(r["SOC_new"] >= 0.0)
    assert np.all(r["SOC_new"] <= 1.0)


def test_benchmark(model):
    SOC = np.random.uniform(0.1, 1.0, 1000)
    P = np.random.uniform(1.0, 10.0, 1000)
    start = time.perf_counter()
    model.predict({"SOC": SOC, "P_request_MW": P, "mode": "discharge"})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
