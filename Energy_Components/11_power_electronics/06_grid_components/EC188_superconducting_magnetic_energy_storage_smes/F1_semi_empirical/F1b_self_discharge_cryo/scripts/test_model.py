"""EC188 — SMES — F1b Self-Discharge + Cryogenic — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"SOC": 0.8, "P_request_MW": 1.0})
    for k in ["P_delivered_MW", "P_grid_MW", "P_cryo_load_MW", "P_ac_loss_MW",
              "SOC_new", "E_stored_MJ", "I_coil_A",
              "eta_instantaneous", "eta_rt_estimate", "self_discharge_tau_h"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC188"
    assert info["fidelity"] == "F1b"


def test_cryo_power_positive(model):
    P = model.model.cryo_power_MW(20.0, 0.0)
    assert float(P) > 0.0, "Cryogenic cooling must always consume power"


def test_cryo_power_increases_with_temperature(model):
    """Warmer operating temperature → worse COP → more grid power needed."""
    P_cold = model.model.cryo_power_MW(10.0, 0.0)
    P_warm = model.model.cryo_power_MW(40.0, 0.0)
    assert float(P_warm) > float(P_cold), \
        f"P_cryo(40K)={float(P_warm):.4f} MW should > P_cryo(10K)={float(P_cold):.4f} MW"


def test_ac_loss_increases_with_current(model):
    """AC losses scale with I^n_ac."""
    P_low  = model.model.ac_loss_power_W(100.0)
    P_high = model.model.ac_loss_power_W(300.0)
    assert float(P_high) > float(P_low)


def test_self_discharge_tau_positive(model):
    tau = model.model.self_discharge_tau_h(0.8)
    assert float(tau) > 0.0


def test_self_discharge_tau_longer_at_low_soc(model):
    """Lower SOC → less stored energy → shorter time to min, but P_cryo is constant.
    Actually at low I, ac losses are smaller → tau could go either way.
    Physical: tau = (E - E_min) / P_total, decreases as SOC→0."""
    tau_high = model.model.self_discharge_tau_h(0.9)
    tau_low  = model.model.self_discharge_tau_h(0.2)
    assert float(tau_high) >= float(tau_low), \
        "Higher SOC → more energy stored → longer time to discharge to minimum"


def test_discharge_soc_decreases(model):
    r = model.predict({"SOC": 0.8, "P_request_MW": 1.0, "mode": "discharge", "dt_s": 10.0})
    assert float(r["SOC_new"]) < 0.8, "Discharge must decrease SOC"


def test_charge_soc_increases(model):
    r = model.predict({"SOC": 0.3, "P_request_MW": 1.0, "mode": "charge", "dt_s": 10.0})
    assert float(r["SOC_new"]) > 0.3, "Charge must increase SOC"


def test_efficiency_less_than_one_discharge(model):
    r = model.predict({"SOC": 0.8, "P_request_MW": 1.5, "mode": "discharge"})
    assert float(r["eta_instantaneous"]) < 1.0


def test_efficiency_less_than_one_charge(model):
    r = model.predict({"SOC": 0.2, "P_request_MW": 1.5, "mode": "charge"})
    assert float(r["eta_instantaneous"]) < 1.0


def test_rt_efficiency_positive(model):
    r = model.predict({"SOC": 0.8, "P_request_MW": 1.5})
    assert float(r["eta_rt_estimate"]) > 0.0


def test_cryo_load_reported_in_output(model):
    r = model.predict({"SOC": 0.8, "P_request_MW": 1.0})
    assert float(r["P_cryo_load_MW"]) > 0.0


def test_soc_bounded(model):
    r = model.predict({"SOC": 0.99, "P_request_MW": 2.0, "mode": "charge", "dt_s": 3600.0})
    assert 0.0 <= float(r["SOC_new"]) <= 1.0


def test_vectorized(model):
    SOC = np.linspace(0.1, 0.9, 20)
    r = model.predict({"SOC": SOC, "P_request_MW": 1.0})
    assert r["SOC_new"].shape == (20,)


def test_benchmark(model):
    SOC = np.random.uniform(0.1, 0.9, 1000)
    start = time.perf_counter()
    model.predict({"SOC": SOC, "P_request_MW": 1.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 5.0
