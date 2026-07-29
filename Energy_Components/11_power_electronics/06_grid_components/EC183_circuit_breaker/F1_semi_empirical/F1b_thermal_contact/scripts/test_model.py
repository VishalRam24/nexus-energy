"""EC183 -- Circuit Breaker -- F1b Thermal Contact -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_keys(model):
    r = model.predict({"I_A": 500.0})
    for k in ["P_cond_W", "P_aux_W", "P_total_W", "R_contact_Ohm",
              "F_skin", "I_max_thermal_A", "thermal_margin",
              "is_overloaded", "can_interrupt", "E_fault_J"]:
        assert k in r, f"missing key {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC183"
    assert info["fidelity"] == "F1b"


# --- Physics: R(T) ---

def test_resistance_increases_with_temperature(model):
    """R(T) must increase with T_contact (positive alpha_Cu)."""
    r_cold = float(model.predict({"I_A": 100.0, "T_contact": 20.0})["R_contact_Ohm"])
    r_hot = float(model.predict({"I_A": 100.0, "T_contact": 100.0})["R_contact_Ohm"])
    assert r_hot > r_cold, f"R must increase: R(20C)={r_cold:.2e}, R(100C)={r_hot:.2e}"


def test_losses_increase_with_temperature(model):
    """Higher T_contact -> higher R -> higher conduction losses."""
    p20 = float(model.predict({"I_A": 630.0, "T_contact": 20.0})["P_cond_W"])
    p100 = float(model.predict({"I_A": 630.0, "T_contact": 100.0})["P_cond_W"])
    assert p100 > p20


def test_losses_zero_when_open(model):
    """Open breaker: conduction losses must be zero."""
    r = model.predict({"I_A": 630.0, "state": "open"})
    assert float(r["P_cond_W"]) == 0.0


def test_aux_power_always_present(model):
    """Auxiliary power drawn regardless of state."""
    for state in ["closed", "open"]:
        r = model.predict({"I_A": 0.0, "state": state})
        assert float(r["P_aux_W"]) > 0.0, f"P_aux must be >0 in {state} state"


def test_total_power_equals_cond_plus_aux(model):
    """P_total = P_cond + P_aux."""
    I_vals = [0.0, 200.0, 630.0]
    for I in I_vals:
        r = model.predict({"I_A": I, "T_contact": 60.0})
        diff = abs(float(r["P_total_W"]) - float(r["P_cond_W"]) - float(r["P_aux_W"]))
        assert diff < 1e-9, f"Power balance failed at I={I} A"


def test_skin_effect_factor_gte_one(model):
    """F_skin must be >= 1 (AC resistance >= DC resistance)."""
    assert model._model.F_skin >= 1.0


def test_losses_scale_with_current_squared(model):
    """P_cond scales as I^2 (fixed T_contact)."""
    r1 = float(model.predict({"I_A": 300.0, "T_contact": 50.0})["P_cond_W"])
    r2 = float(model.predict({"I_A": 600.0, "T_contact": 50.0})["P_cond_W"])
    # r2 / r1 should be ~4 (= (600/300)^2)
    assert abs(r2 / r1 - 4.0) < 0.01, f"I^2 scaling: expected ratio~4, got {r2/r1:.3f}"


# --- Ampacity ---

def test_ampacity_at_rated_ambient(model):
    """Ampacity at T_ref should equal I_rated."""
    # T_ref = 20 C: sqrt((105-20)/(105-20)) = 1.0 => I_max = I_rated
    r = model.predict({"I_A": 0.0, "T_ambient": 20.0})
    assert abs(float(r["I_max_thermal_A"]) - 630.0) < 1.0


def test_ampacity_decreases_with_ambient(model):
    """Ampacity decreases at higher ambient temperature."""
    I_20 = float(model.predict({"I_A": 0.0, "T_ambient": 20.0})["I_max_thermal_A"])
    I_50 = float(model.predict({"I_A": 0.0, "T_ambient": 50.0})["I_max_thermal_A"])
    assert I_50 < I_20, f"Ampacity must decrease: I(20C)={I_20:.1f}, I(50C)={I_50:.1f}"


def test_overloaded_flag(model):
    """Overload flag set when I > I_max_thermal."""
    r = model.predict({"I_A": 800.0, "T_ambient": 20.0})
    assert bool(r["is_overloaded"]) is True


def test_not_overloaded_at_rated(model):
    """No overload at rated current, rated ambient."""
    r = model.predict({"I_A": 630.0, "T_ambient": 20.0})
    assert bool(r["is_overloaded"]) is False


# --- Interrupting Rating ---

def test_can_interrupt_within_rating(model):
    r = model.predict({"I_fault_kA": 20.0})
    assert bool(r["can_interrupt"]) is True


def test_cannot_interrupt_above_rating(model):
    r = model.predict({"I_fault_kA": 30.0})
    assert bool(r["can_interrupt"]) is False


def test_fault_energy_positive(model):
    r = model.predict({"I_fault_kA": 10.0})
    assert float(r["E_fault_J"]) > 0.0


# --- Vectorised ---

def test_vectorised(model):
    I = np.linspace(0, 700, 50)
    T = np.linspace(20, 100, 50)
    r = model.predict({"I_A": I, "T_contact": T})
    assert len(r["P_cond_W"]) == 50


# --- Benchmark ---

def test_benchmark(model):
    I = np.random.uniform(0, 700, 1000)
    T = np.random.uniform(20, 100, 1000)
    start = time.perf_counter()
    model.predict({"I_A": I, "T_contact": T})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 2.0
