"""EC098 -- ORC -- F1b Part-Load + Condenser Ambient -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_heat_source": 150.0, "T_condenser": 30.0, "PLR": 1.0})
    for k in ["efficiency", "power_output_kw", "heat_rejection_kw",
              "specific_work_kj_kg"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC098"
    assert info["fidelity"] == "F1b"


# --- Efficiency ---

def test_efficiency_below_carnot(model):
    """ORC efficiency must always be below Carnot limit."""
    T_hot = 150.0
    T_cond = 30.0
    eta_carnot = 1.0 - (T_cond + 273.15) / (T_hot + 273.15)
    r = model.predict({"T_heat_source": T_hot, "T_condenser": T_cond, "PLR": 1.0})
    eta = float(r["efficiency"])
    assert eta < eta_carnot, f"eta={eta:.4f} >= Carnot={eta_carnot:.4f}"


def test_efficiency_positive(model):
    """Efficiency must be positive across valid range."""
    PLR = np.linspace(0.3, 1.0, 50)
    r = model.predict({"T_heat_source": 150.0, "T_condenser": 30.0, "PLR": PLR})
    assert np.all(r["efficiency"] > 0)


def test_efficiency_drops_at_part_load(model):
    r_full = model.predict({"T_heat_source": 150.0, "T_condenser": 30.0, "PLR": 1.0})
    r_part = model.predict({"T_heat_source": 150.0, "T_condenser": 30.0, "PLR": 0.3})
    assert float(r_part["efficiency"]) < float(r_full["efficiency"])


def test_efficiency_at_50pct_load(model):
    """At 50% load, f_PLR ~ 0.85 (typical ORC off-design)."""
    r_full = model.predict({"T_heat_source": 150.0, "T_condenser": 30.0, "PLR": 1.0})
    r_half = model.predict({"T_heat_source": 150.0, "T_condenser": 30.0, "PLR": 0.5})
    ratio = float(r_half["efficiency"]) / float(r_full["efficiency"])
    assert 0.70 < ratio < 0.95, f"50% load ratio = {ratio:.3f}"


# --- Condenser temperature sensitivity ---

def test_higher_condenser_temp_lowers_efficiency(model):
    """ORC very sensitive to condenser temperature -- higher T_cond = lower eta."""
    r_cool = model.predict({"T_heat_source": 150.0, "T_condenser": 20.0, "PLR": 1.0})
    r_hot  = model.predict({"T_heat_source": 150.0, "T_condenser": 45.0, "PLR": 1.0})
    assert float(r_cool["efficiency"]) > float(r_hot["efficiency"])


def test_condenser_sensitivity_significant(model):
    """A 15K condenser rise should reduce efficiency by >10% (ORC is very sensitive)."""
    r_30 = model.predict({"T_heat_source": 150.0, "T_condenser": 30.0, "PLR": 1.0})
    r_45 = model.predict({"T_heat_source": 150.0, "T_condenser": 45.0, "PLR": 1.0})
    eta_drop = 1.0 - float(r_45["efficiency"]) / float(r_30["efficiency"])
    assert eta_drop > 0.10, f"Only {eta_drop*100:.1f}% drop for 15K rise"


# --- Heat source temperature ---

def test_higher_heat_source_improves_efficiency(model):
    r_low  = model.predict({"T_heat_source": 100.0, "T_condenser": 30.0, "PLR": 1.0})
    r_high = model.predict({"T_heat_source": 200.0, "T_condenser": 30.0, "PLR": 1.0})
    assert float(r_high["efficiency"]) > float(r_low["efficiency"])


# --- Energy balance ---

def test_energy_balance(model):
    """P_out + Q_reject = Q_hot (first law)."""
    r = model.predict({"T_heat_source": 150.0, "T_condenser": 30.0, "PLR": 0.8,
                        "heat_input_kw": 500.0})
    P_out = float(r["power_output_kw"])
    Q_rej = float(r["heat_rejection_kw"])
    assert abs((P_out + Q_rej) - 500.0) < 0.1


# --- Power output ---

def test_power_at_rated(model):
    """At design conditions (150C/30C, PLR=1), power should be ~100 kW."""
    r = model.predict({"T_heat_source": 150.0, "T_condenser": 30.0, "PLR": 1.0})
    P = float(r["power_output_kw"])
    assert abs(P - 100.0) / 100.0 < 0.10, f"P={P:.1f} kW, expected ~100"


# --- Heat rate increases at part load ---

def test_heat_rate_increases_at_part_load(model):
    """Equivalent heat rate 3600/eta should increase at part load."""
    r_full = model.predict({"T_heat_source": 150.0, "T_condenser": 30.0, "PLR": 1.0})
    r_part = model.predict({"T_heat_source": 150.0, "T_condenser": 30.0, "PLR": 0.5})
    hr_full = 3600 / float(r_full["efficiency"])
    hr_part = 3600 / float(r_part["efficiency"])
    assert hr_part > hr_full


# --- Edge cases ---

def test_minimum_plr(model):
    r = model.predict({"T_heat_source": 150.0, "T_condenser": 30.0, "PLR": 0.3})
    assert float(r["efficiency"]) > 0
    assert float(r["power_output_kw"]) > 0


def test_extreme_hot_condenser(model):
    """At very high condenser temp (55C), efficiency should still be positive
    if heat source is hot enough."""
    r = model.predict({"T_heat_source": 200.0, "T_condenser": 55.0, "PLR": 1.0})
    assert float(r["efficiency"]) > 0


# --- Benchmark ---

def test_benchmark(model):
    PLR = np.random.uniform(0.3, 1.0, 1000)
    T_cond = np.random.uniform(20, 50, 1000)
    start = time.perf_counter()
    model.predict({"T_heat_source": 150.0, "T_condenser": T_cond, "PLR": PLR})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
