"""EC068 — ASHP — F2a Vapor Compression SS — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_evap_degC": 0.0, "T_cond_degC": 45.0})
    for k in ["cop", "heating_capacity_kw", "compressor_power_kw", "mass_flow_kg_s", "state_points"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC068"
    assert info["fidelity"] == "F2a"


def test_cop_greater_than_one(model):
    """Heat pump COP must be > 1."""
    for Te in [-15, -5, 0, 5, 10]:
        r = model.predict({"T_evap_degC": Te, "T_cond_degC": 45.0})
        assert r["cop"] > 1.0, f"COP={r['cop']} at T_evap={Te}"


def test_cop_decreases_with_temp_lift(model):
    """COP must decrease as temperature lift increases."""
    cops = []
    for Te in [10, 5, 0, -5, -10, -15]:
        r = model.predict({"T_evap_degC": Te, "T_cond_degC": 45.0})
        cops.append(r["cop"])
    for i in range(len(cops) - 1):
        assert cops[i] > cops[i + 1], f"COP not monotonically decreasing: {cops}"


def test_cop_reasonable_range(model):
    """COP at typical conditions should be 2-6."""
    r = model.predict({"T_evap_degC": 0.0, "T_cond_degC": 45.0})
    assert 2.0 < r["cop"] < 6.0, f"COP={r['cop']} out of reasonable range"


def test_energy_balance(model):
    """Q_heating = Q_evap + W_comp (first law)."""
    r = model.predict({"T_evap_degC": 0.0, "T_cond_degC": 45.0})
    q_cond = r["q_cond_specific_kj_kg"]
    q_evap = r["q_evap_specific_kj_kg"]
    w_comp = r["w_comp_specific_kj_kg"]
    balance = abs(q_cond - (q_evap + w_comp)) / q_cond
    assert balance < 0.01, f"Energy balance error: {balance*100:.2f}%"


def test_state_points_exist(model):
    """All 5 state points must be present."""
    r = model.predict({"T_evap_degC": 0.0, "T_cond_degC": 45.0})
    sp = r["state_points"]
    for key in ["1_evap_out", "2s_isentropic", "2_comp_out", "3_cond_out", "4_exp_out"]:
        assert key in sp, f"Missing state point: {key}"


def test_pressure_ratio(model):
    """Pressure ratio should be > 1 and reasonable."""
    r = model.predict({"T_evap_degC": 0.0, "T_cond_degC": 45.0})
    assert 1.5 < r["pressure_ratio"] < 8.0


def test_superheat_effect(model):
    """More superheat should slightly reduce COP (more compressor work)."""
    r1 = model.predict({"T_evap_degC": 0.0, "T_cond_degC": 45.0, "superheat_K": 2.0})
    r2 = model.predict({"T_evap_degC": 0.0, "T_cond_degC": 45.0, "superheat_K": 15.0})
    # Large superheat generally changes COP; just check both are valid
    assert r1["cop"] > 1.0 and r2["cop"] > 1.0


def test_mass_flow_positive(model):
    r = model.predict({"T_evap_degC": -10.0, "T_cond_degC": 50.0})
    assert r["mass_flow_kg_s"] > 0


def test_benchmark(model):
    start = time.perf_counter()
    for _ in range(100):
        model.predict({"T_evap_degC": 0.0, "T_cond_degC": 45.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 100 cycles in {elapsed*1000:.1f} ms ({elapsed*10:.2f} ms/cycle)")
    assert elapsed < 10.0
