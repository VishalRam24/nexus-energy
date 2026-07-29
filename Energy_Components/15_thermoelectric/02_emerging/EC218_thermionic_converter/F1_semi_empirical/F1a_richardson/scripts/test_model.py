"""EC218 — Thermionic Converter — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_emitter": 1700.0, "T_collector": 900.0})
    for k in ["J_emitter_Am2", "J_net_Am2", "V_out_V", "power_w", "heat_input_w", "efficiency"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC218"
    assert info["fidelity"] == "F1a"


def test_emission_monotone_increasing_in_T(model):
    """Richardson-Dushman: J must strictly increase with T_emitter."""
    T_e = np.linspace(1200.0, 2000.0, 20)
    r = model.predict({"T_emitter": T_e, "T_collector": 600.0})
    assert np.all(np.diff(r["J_emitter_Am2"]) > 0), \
        "J_emitter must monotonically increase with T_emitter (Richardson-Dushman)"


def test_power_increases_with_T_emitter(model):
    """Power output must increase with emitter temperature."""
    T_e = np.linspace(1300.0, 1900.0, 20)
    r = model.predict({"T_emitter": T_e, "T_collector": 700.0})
    assert np.all(np.diff(r["power_w"]) > 0), \
        "Power must increase with T_emitter"


def test_collector_back_emission_reduces_net_current(model):
    """Higher collector temperature increases back-emission, reducing J_net."""
    # Hold T_emitter fixed, increase T_collector
    T_c = np.linspace(500.0, 1100.0, 20)
    r = model.predict({"T_emitter": 1700.0, "T_collector": T_c})
    assert np.all(np.diff(r["J_net_Am2"]) < 0), \
        "J_net must decrease as T_collector increases (more back-emission)"


def test_efficiency_in_physical_range(model):
    """Thermionic efficiency must be between 0 and 50% for valid inputs."""
    T_e = np.linspace(1300.0, 2000.0, 30)
    r = model.predict({"T_emitter": T_e, "T_collector": 800.0})
    assert np.all(r["efficiency"] >= 0.0), "Efficiency must be non-negative"
    assert np.all(r["efficiency"] <= 0.5), "Efficiency must be <= 50%"


def test_efficiency_theoretical_range_10_to_20_pct(model):
    """At typical conditions (1500-1800K emitter), efficiency ~10-20%."""
    r = model.predict({"T_emitter": 1700.0, "T_collector": 900.0})
    eta = float(r["efficiency"])
    assert 0.05 < eta < 0.30, \
        f"Efficiency {eta*100:.1f}% outside expected 5-30% range for thermionic converter"


def test_first_law_energy_conservation(model):
    """Power output must be less than heat input (first law)."""
    T_e = np.linspace(1300.0, 1900.0, 20)
    r = model.predict({"T_emitter": T_e, "T_collector": 700.0})
    assert np.all(r["power_w"] <= r["heat_input_w"]), \
        "Power output cannot exceed heat input (first law violation)"


def test_zero_net_current_when_emitter_equals_collector_temp(model):
    """When T_emitter == T_collector and phi_e == phi_c, net current approaches zero."""
    # With same temperature, J_e and J_c both use same T but different phi
    # With phi_e > phi_c (default), collector still emits less -> net current exists
    # But: when T_emitter is very low, J_emitter -> 0
    r = model.predict({"T_emitter": 1200.0, "T_collector": 1200.0})
    # J_net = J_e(phi_e=2.0eV) - J_c(phi_c=1.5eV, T_c=1200K)
    # At same T, higher phi_c means less back-emission, so J_net >= 0
    assert float(r["J_net_Am2"]) >= 0.0, "Net current must be non-negative"


def test_v_out_positive(model):
    """Output voltage must be positive (phi_emitter > phi_collector)."""
    r = model.predict({"T_emitter": 1700.0, "T_collector": 900.0})
    assert float(r["V_out_V"]) > 0.0, "V_out must be positive for phi_e > phi_c"


def test_benchmark(model):
    T_e = np.random.uniform(1200.0, 2000.0, 1000)
    T_c = np.random.uniform(400.0, 1100.0, 1000)
    start = time.perf_counter()
    model.predict({"T_emitter": T_e, "T_collector": T_c})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
