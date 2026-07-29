"""EC097 — Rankine Steam Turbine — F1b Ambient + Part-Load — Test Suite

Tests MUST fail the model, not accommodate it.
"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_keys(model):
    r = model.predict({"PLR": 0.8})
    for k in ["efficiency_gross", "efficiency_net", "power_output_mw",
              "heat_input_mw", "heat_rejection_mw", "condenser_pressure_kpa",
              "f_condenser", "f_partload"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC097"
    assert info["fidelity"] == "F1b"


# --- Part-load efficiency ---

def test_efficiency_drops_at_part_load(model):
    """Efficiency at PLR=0.3 must be lower than at PLR=1.0."""
    r_full = model.predict({"PLR": 1.0})
    r_part = model.predict({"PLR": 0.3})
    assert float(r_part["efficiency_net"]) < float(r_full["efficiency_net"]), \
        "Efficiency must degrade at part load"


def test_efficiency_monotone_with_plr(model):
    """Efficiency generally increases with PLR (quadratic penalty: f_PLR = 1 - a*(1-PLR)^2)."""
    plr = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    r = model.predict({"PLR": plr})
    eta = np.asarray(r["efficiency_net"])
    assert np.all(np.diff(eta) > 0), "Efficiency should increase monotonically with PLR"


def test_f_partload_is_one_at_full_load(model):
    """f_PLR = 1 - a*(1-1)^2 = 1.0 at PLR=1."""
    r = model.predict({"PLR": 1.0})
    assert float(r["f_partload"]) == pytest.approx(1.0, abs=1e-9)


def test_efficiency_below_rated_at_low_plr(model):
    r = model.predict({"PLR": 0.4})
    assert float(r["efficiency_net"]) < 0.38


# --- Condenser temperature effect ---

def test_higher_T_cond_reduces_efficiency(model):
    """Warmer condenser reduces cycle efficiency (higher T_cold = lower Carnot)."""
    r_cool = model.predict({"PLR": 1.0, "T_condenser": 20.0})
    r_hot  = model.predict({"PLR": 1.0, "T_condenser": 48.0})
    assert float(r_cool["efficiency_net"]) > float(r_hot["efficiency_net"]), \
        "Hot condenser must reduce efficiency"


def test_T_cond_effect_significant(model):
    """A 15 K rise in condenser temp should cause measurable efficiency drop."""
    r1 = model.predict({"PLR": 1.0, "T_condenser": 25.0})
    r2 = model.predict({"PLR": 1.0, "T_condenser": 40.0})
    drop = float(r1["efficiency_net"]) - float(r2["efficiency_net"])
    assert drop > 0.005, f"15 K condenser rise should drop eta by >0.005, got {drop:.4f}"


def test_f_condenser_is_one_at_design(model):
    """f_cond = 1.0 at design condenser temperature."""
    T_design = model._raw["turbine"]["T_cond_design_c"]["value"]
    r = model.predict({"PLR": 1.0, "T_condenser": T_design})
    assert float(r["f_condenser"]) == pytest.approx(1.0, abs=1e-9)


def test_condenser_pressure_increases_with_temp(model):
    """Saturation pressure must increase with temperature (Clausius-Clapeyron)."""
    T_arr = np.array([20.0, 30.0, 40.0, 50.0])
    r = model.predict({"PLR": 1.0, "T_condenser": T_arr})
    P = np.asarray(r["condenser_pressure_kpa"])
    assert np.all(np.diff(P) > 0), "Condenser pressure must increase with temperature"


# --- Carnot limit ---

def test_efficiency_below_carnot(model):
    """Net efficiency must be below Carnot efficiency."""
    T_steam = model._raw["turbine"]["T_steam_c"]["value"]
    T_cond  = model._raw["turbine"]["T_cond_design_c"]["value"]
    eta_carnot = 1.0 - (T_cond + 273.15) / (T_steam + 273.15)
    r = model.predict({"PLR": 1.0})
    assert float(r["efficiency_gross"]) < eta_carnot, \
        f"Gross efficiency {float(r['efficiency_gross']):.3f} >= Carnot {eta_carnot:.3f}"


# --- Net < gross ---

def test_net_below_gross(model):
    plr = np.linspace(0.2, 1.0, 30)
    r = model.predict({"PLR": plr})
    assert np.all(np.asarray(r["efficiency_net"]) < np.asarray(r["efficiency_gross"]) + 1e-9)


# --- Energy balance ---

def test_energy_balance(model):
    """Q_in = P_net + Q_reject (first law, within 0.5%)."""
    r = model.predict({"PLR": 0.8, "T_condenser": 33.0})
    Q_in  = float(r["heat_input_mw"])
    P_net = float(r["power_output_mw"])
    Q_rej = float(r["heat_rejection_mw"])
    if Q_in > 0:
        ratio = abs(Q_in - P_net - Q_rej) / Q_in
        assert ratio < 0.005, f"Energy balance error: {ratio*100:.2f}%"


# --- Power output ---

def test_power_at_rated(model):
    """At PLR=1.0, power output = P_rated = 100 MW."""
    r = model.predict({"PLR": 1.0})
    assert abs(float(r["power_output_mw"]) - 100.0) < 0.01


# --- Below minimum PLR ---

def test_zero_efficiency_below_plr_min(model):
    """Below PLR_min, efficiency should be zero."""
    r = model.predict({"PLR": 0.10})
    assert float(r["efficiency_gross"]) == 0.0


# --- Benchmark ---

def test_benchmark(model):
    plr  = np.random.uniform(0.2, 1.0, 1000)
    T_c  = np.random.uniform(20, 50, 1000)
    start = time.perf_counter()
    model.predict({"PLR": plr, "T_condenser": T_c})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
