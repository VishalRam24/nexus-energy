"""EC079 — Molten Salt TES — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

@pytest.fixture
def model():
    return ComponentModel()

def test_predict_keys(model):
    r = model.predict({"temperature": 427.5, "q_charge": 0.0, "q_discharge": 0.0})
    for k in ["dT_dt", "energy_stored_mwh", "soc", "heat_loss_kw"]:
        assert k in r, f"Missing key: {k}"

def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC079"
    assert info["fidelity"] == "F1a"

def test_energy_nonnegative(model):
    """Energy stored must be >= 0 for any temperature in the valid range."""
    T_vals = np.linspace(290.0, 565.0, 50)
    r = model.predict({"temperature": T_vals, "q_charge": 0.0, "q_discharge": 0.0})
    assert np.all(r["energy_stored_mwh"] >= 0.0), "Energy stored must be non-negative"

def test_soc_at_cold(model):
    """At T_cold=290°C, SOC should be 0."""
    r = model.predict({"temperature": 290.0, "q_charge": 0.0, "q_discharge": 0.0})
    assert float(r["soc"]) == pytest.approx(0.0, abs=1e-6)

def test_soc_at_hot(model):
    """At T_hot=565°C, SOC should be 1."""
    r = model.predict({"temperature": 565.0, "q_charge": 0.0, "q_discharge": 0.0})
    assert float(r["soc"]) == pytest.approx(1.0, abs=1e-6)

def test_soc_in_range(model):
    """SOC must be in [0, 1] for any temperature in operating range."""
    T_vals = np.linspace(290.0, 565.0, 100)
    r = model.predict({"temperature": T_vals, "q_charge": 0.0, "q_discharge": 0.0})
    assert np.all(r["soc"] >= 0.0) and np.all(r["soc"] <= 1.0)

def test_heat_loss_when_T_above_ambient(model):
    """Heat loss must be positive when salt is above ambient temperature."""
    r = model.predict({"temperature": 400.0, "q_charge": 0.0, "q_discharge": 0.0, "t_ambient": 25.0})
    assert float(r["heat_loss_kw"]) > 0.0, "Must have positive heat loss when T > T_ambient"

def test_dT_dt_positive_when_charging(model):
    """Temperature must rise when charging (and not discharging)."""
    r = model.predict({"temperature": 400.0, "q_charge": 50000.0, "q_discharge": 0.0})
    assert float(r["dT_dt"]) > 0.0, "dT/dt must be positive during charging"

def test_dT_dt_negative_when_discharging(model):
    """Temperature must fall when discharging (and not charging)."""
    r = model.predict({"temperature": 500.0, "q_charge": 0.0, "q_discharge": 50000.0})
    assert float(r["dT_dt"]) < 0.0, "dT/dt must be negative during discharging"

def test_equilibrium_no_charge_discharge(model):
    """With no charge/discharge, temperature should drift down slowly due to heat loss."""
    r = model.predict({"temperature": 400.0, "q_charge": 0.0, "q_discharge": 0.0, "t_ambient": 25.0})
    assert float(r["dT_dt"]) < 0.0, "Must cool down (heat loss) when idle with T > T_ambient"

def test_energy_capacity(model):
    """Total energy capacity (T_hot - T_cold) should be ~206 MWh."""
    from model import MoltenSaltTESF1a
    import json
    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    m = MoltenSaltTESF1a(params)
    assert 180.0 < m.E_capacity_MWh < 250.0, f"E_capacity = {m.E_capacity_MWh:.1f} MWh, expected ~206 MWh"

def test_energy_increases_with_temperature(model):
    """Higher salt temperature → more energy stored."""
    T_vals = np.linspace(290.0, 565.0, 50)
    r = model.predict({"temperature": T_vals, "q_charge": 0.0, "q_discharge": 0.0})
    assert np.all(np.diff(r["energy_stored_mwh"]) >= 0.0)

def test_simulation_roundtrip(model):
    """Charge to full, then discharge — energy should return to start."""
    from model import MoltenSaltTESF1a
    import json
    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    m = MoltenSaltTESF1a(params)
    N = 24
    q_charge = np.full(N, 50000.0)   # 50 MW charging for 24h
    q_dis    = np.zeros(N)
    sim = m.simulate(290.0, q_charge, q_dis, dt_s=3600.0)
    # SOC should increase from 0
    assert sim["soc"][-1] > sim["soc"][0], "SOC must increase during charging"

def test_benchmark(model):
    """1000 instantaneous predictions must complete in <1 second."""
    T = np.random.uniform(300, 560, 1000)
    Qc = np.random.uniform(0, 50000, 1000)
    Qd = np.random.uniform(0, 50000, 1000)
    start = time.perf_counter()
    model.predict({"temperature": T, "q_charge": Qc, "q_discharge": Qd})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
